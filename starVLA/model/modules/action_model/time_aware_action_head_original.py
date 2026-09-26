from collections import OrderedDict
from typing import Optional
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from transformers import PreTrainedModel, PretrainedConfig

class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, time_embedding_dim: Optional[int] = None):
        super().__init__()

        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # elementwise_affine=False: 不创建可学习参数
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))



@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)

    initializer_range: float = field(default=0.02)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class PerceiverAttentionBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, time_embedding_dim: Optional[int] = None
    ):
        super().__init__()
        # head_dim = embed_dim // num_heads
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        normed_latents = self.ln_1(latents, timestep_embedding)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1),
        )
        latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
        return latents


class TimestepEncoder(nn.Module):
    def __init__(self, time_channel, time_embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.position = Timesteps(
            time_channel, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel,
            time_embed_dim=time_embedding_dim,
        )
    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        time_feature = self.position(timesteps).to(dtype=dtype)
        time_feature = time_feature.unsqueeze(1)
        time_embedding = self.time_embedding(time_feature)  # (N,1,D)
        return time_embedding


class TimeAwareActionHead_Config(PretrainedConfig):
    model_type = "TimeAwareActionHead"
    def __init__(
        self,
        time_channel: int = 320,
        time_embedding_dim: int = 768,
        time_out_dim: int = None,
        action_dim=768,
        vl_input_dim=2048,
        heads: int = 16,
        layers: int = 16,
        output_dim=1024,
        compute_dtype=torch.float32,
        initializer_range=0.02,
        **kwargs,
    ):
        # head_dim = embed_dim(action_dim) // num_heads
        self.time_channel = time_channel
        self.time_embedding_dim = time_embedding_dim
        self.time_out_dim = time_out_dim
        self.action_dim = action_dim
        self.vl_input_dim = vl_input_dim
        self.heads = heads
        self.layers = layers
        self.output_dim = output_dim
        self.compute_dtype = compute_dtype
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    def convert_to_dict(self):
        return {
            "time_channel": self.time_channel,
            "time_embedding_dim": self.time_embedding_dim,
            "time_out_dim": self.time_out_dim,
            "action_dim": self.action_dim,
            "vl_input_dim": self.vl_input_dim,
            "heads": self.heads,
            "layers": self.layers,
            "output_dim": self.output_dim,
            # "compute_dtype": self.compute_dtype, # issue: TypeError: Object of type dtype is not JSON serializable
            "initializer_range": self.initializer_range,
        }

class TimeAwareActionHead(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = TimeAwareActionHead_Config
    def __init__(self,
        config: TimeAwareActionHead_Config,
    ):
        super().__init__(config)
        self.config = config
        self.vl_input_dim = self.config.vl_input_dim
        self.output_dim = self.config.output_dim
        self.action_dim = self.config.action_dim

        self.time_encoder = TimestepEncoder(
            time_channel=self.config.time_channel,
            time_embedding_dim=self.config.time_embedding_dim,
            compute_dtype=self.config.compute_dtype
        )

        self.time_aware_linear = nn.Linear(
            self.config.time_embedding_dim, self.action_dim, bias=True
        )

        if self.vl_input_dim is not None:
            self.proj_in = nn.Linear(self.vl_input_dim, self.action_dim)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    self.action_dim, self.config.heads, time_embedding_dim=self.config.time_embedding_dim,
                )
                for _ in range(self.config.layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(self.action_dim, self.output_dim), nn.LayerNorm(self.output_dim)
            )

        print(
            "Total number of TimeAwareActionHead parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def init_weights(self):
        for module in self.children():
            module.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        latents: torch.Tensor,  # Shape: (B, T, D)
        visual_language_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
    ):
        time_embedding = self.time_encoder(timestep)

        latents = latents + self.time_aware_linear(
            torch.nn.functional.silu(time_embedding)
        )
        # import ipdb;ipdb.set_trace()

        if self.vl_input_dim is not None:
            visual_language_states = self.proj_in(visual_language_states)

        for l_block in self.perceiver_blocks:
            latents = l_block(
                x=visual_language_states,
                latents=latents,
                timestep_embedding=time_embedding,
            )

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        # print("哈哈哈啊哈哈")

        return latents


# register
from transformers import AutoConfig,AutoModel
AutoConfig.register("TimeAwareActionHead", TimeAwareActionHead_Config)
AutoModel.register(TimeAwareActionHead_Config, TimeAwareActionHead)