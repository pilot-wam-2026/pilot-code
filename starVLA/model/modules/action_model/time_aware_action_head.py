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
        self, d_model: int, n_heads: int, time_embedding_dim: Optional[int] = None,
        num_non_action_tokens: int = 65,
    ):
        super().__init__()
        # head_dim = embed_dim // num_heads
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.num_non_action_tokens = num_non_action_tokens

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        # ln_1: 对latents归一化 — 前num_non_action_tokens个token做普通LN，后16个token做AdaLN
        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_1_plain = nn.LayerNorm(d_model, eps=1e-6)
        # ln_2: 对x(vl_tokens)做普通LN，不做AdaLN
        self.ln_2_plain = nn.LayerNorm(d_model, eps=1e-6)
        # ln_ff: 对latents归一化 — 前num_non_action_tokens个token做普通LN，后16个token做AdaLN
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff_plain = nn.LayerNorm(d_model, eps=1e-6)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    # 我们希望future_tokens能够和action_tokens完全解耦，具体来说：
    # （1）self.ln_1和self.ln_ff在对latents做归一化时，拆分为两组，对前65个token做普通LN、对后16个token做AdaLN；
    # （2）不对x(vl_tokens)做AdaLN，而是改做普通LN；
    # （3）分离attention，future_tokens只与state_tokens、future_tokens和vl_tokens做cross-attn，
    #      而action_tokens可以对以上所有tokens做cross-attn
    def forward_for_WAM_VJEPA2AC(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        # print('################# PerceiverAttentionBlock.forward_for_WAM_VJEPA2AC-1')
        num_action_tokens = latents.shape[1] - self.num_non_action_tokens

        # (1) ln_1: 对latents做归一化 — 前65个token做普通LN，后16个token做AdaLN
        # print(latents.shape)                # torch.Size([32, 81, 768])
        normed_non_action = self.ln_1_plain(latents[:, :self.num_non_action_tokens])
        # print(normed_non_action.shape)      # torch.Size([32, 65, 768])
        normed_action = self.ln_1(latents[:, self.num_non_action_tokens:], timestep_embedding)
        # print(normed_action.shape)          # torch.Size([32, 16, 768])
        normed_latents = torch.cat([normed_non_action, normed_action], dim=1)
        # print(normed_latents.shape)         # torch.Size([32, 81, 768])

        # (2) ln_2: 对x(vl_tokens)做普通LN
        # print(x.shape)                      # torch.Size([32, 196, 768])
        normed_x = self.ln_2_plain(x)
        # print(normed_x.shape)               # torch.Size([32, 196, 768])

        # (3) 分离attention
        # future_tokens只attend [state_tokens, future_tokens, vl_tokens]
        # action_tokens可以attend [state_tokens, future_tokens, action_tokens, vl_tokens]
        non_action_kv = torch.cat([normed_non_action, normed_x], dim=1)
        # print(non_action_kv.shape)          # torch.Size([32, 261, 768])
        full_kv = torch.cat([normed_latents, normed_x], dim=1)
        # print(full_kv.shape)                # torch.Size([32, 277, 768])

        non_action_attn = self.attention(q=normed_non_action, kv=non_action_kv)
        action_attn = self.attention(q=normed_action, kv=full_kv)
        # print(non_action_attn.shape)        # torch.Size([32, 65, 768])
        # print(action_attn.shape)            # torch.Size([32, 16, 768])

        latents_non_action = latents[:, :self.num_non_action_tokens] + non_action_attn
        # print(latents_non_action.shape)     # torch.Size([32, 65, 768])
        latents_action = latents[:, self.num_non_action_tokens:] + action_attn
        # print(latents_action.shape)         # torch.Size([32, 16, 768])
        latents = torch.cat([latents_non_action, latents_action], dim=1)
        # print(latents.shape)                # torch.Size([32, 81, 768])

        # (4) ln_ff: 对latents做归一化 — 前65个token做普通LN，后16个token做AdaLN
        ff_non_action = self.mlp(self.ln_ff_plain(latents[:, :self.num_non_action_tokens]))
        # print(ff_non_action.shape)          # torch.Size([32, 65, 768])
        ff_action = self.mlp(self.ln_ff(latents[:, self.num_non_action_tokens:], timestep_embedding))
        # print(ff_action.shape)              # torch.Size([32, 16, 768])
        latents = torch.cat([
            latents[:, :self.num_non_action_tokens] + ff_non_action,
            latents[:, self.num_non_action_tokens:] + ff_action,
        ], dim=1)
        # print(latents.shape)                # torch.Size([32, 81, 768])

        # print('################# PerceiverAttentionBlock.forward_for_WAM_VJEPA2AC-2')
        return latents
    
    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        print('Warning: this function should not used in WAM-VJEPA !!!!!! '
        '(in /path/to/workspace/projects/WM4A/starVLA/model/modules/action_model/time_aware_action_head.py)')
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
        num_target_vision_tokens=64,
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
        self.num_target_vision_tokens = num_target_vision_tokens
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
            "num_target_vision_tokens": self.num_target_vision_tokens,
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
        self.num_target_vision_tokens = self.config.num_target_vision_tokens

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
                    d_model=self.action_dim, 
                    n_heads=self.config.heads, 
                    time_embedding_dim=self.config.time_embedding_dim,
                    num_non_action_tokens=self.num_target_vision_tokens+1,
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
        # print('###### TimeAwareActionHead.forward-1')
        # print(timestep.shape)       # torch.Size([32])
        time_embedding = self.time_encoder(timestep)
        # print(time_embedding.shape) # torch.Size([32, 1, 768])
        time_bais = self.time_aware_linear(torch.nn.functional.silu(time_embedding))
        # print(time_bais.shape)      # torch.Size([32, 1, 768])

        # 这里的81个tokens是由三组tokens拼接得到的，
        # 首先是1*state_token，随后是64*future_token，最后是16*action_token，
        # 我们认为这个time_baise只应该施加在action_tokens上，
        # print(latents.shape)        # torch.Size([32, 81, 1024])
        latents[:, self.num_target_vision_tokens+1:] = latents[:, self.num_target_vision_tokens+1:] + time_bais
        # print(latents.shape)        # torch.Size([32, 81, 1024])

        if self.vl_input_dim is not None:
            visual_language_states = self.proj_in(visual_language_states)

        # 我们重构了PerceiverAttentionBlock中的注意力机制，
        # 确保future_tokens不会看到被time_step影响的action_tokens
        for l_block in self.perceiver_blocks:
            latents = l_block.forward_for_WAM_VJEPA2AC(
                x=visual_language_states,
                latents=latents,
                timestep_embedding=time_embedding,
            )

        if self.output_dim is not None:
            latents = self.proj_out(latents)
        # print('###### TimeAwareActionHead.forward-2')
        return latents


# register
from transformers import AutoConfig,AutoModel
AutoConfig.register("TimeAwareActionHead", TimeAwareActionHead_Config)
AutoModel.register(TimeAwareActionHead_Config, TimeAwareActionHead)