# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].
# Action repeat is inspired by CogACT

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.time_aware_action_head import TimeAwareActionHead, TimeAwareActionHead_Config

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple


class CategorySpecificLinear(nn.Module):

    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):

    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):

    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):

        B, T, _ = actions.shape


        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        tau_emb = self.pos_encoding(timesteps).to(dtype=actions.dtype)



        a_emb = self.layer1(actions)


        x = torch.cat([a_emb, tau_emb], dim=-1)

        x = swish(self.layer2(x))

        x = self.layer3(x)


        return x


class MultiEmbodimentActionEncoder(nn.Module):

    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)     # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)     # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)     # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(default=1000, metadata={"help": "Number of timestep discretization buckets."})
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(default=True, metadata={"help": "Whether to tune the diffusion model."})
    load_pretrained_det_decode_layer_path: str = field(default=None,
                                                       metadata={"help": "Path to pretrained detection model."})
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=64, metadata={"help": "Number of target vision tokens."})
    use_category_specific: bool = field(default=False,
                                        metadata={"help": "Whether to use category-specific encoders and decoders."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {
        "input_embedding_dim": 768,
        "attention_head_dim": 64,
        "num_attention_heads": 12
    },
    "DiT-L": {
        "input_embedding_dim": 1536,
        "attention_head_dim": 48,
        "num_attention_heads": 32
    },
}


# 训练时
# future_image支路，self.backbone.transformer()
# 训练时
# action支路，self.backbone.build_inputs、self.backbone.forward()、self.action_model.forward()
# 推理时
# future_image支路，self.backbone.generate()
# 推理时
# action支路，self.backbone.build_inputs()、self.backbone.forward()、self.action_model.predict_action()
class FlowmatchingActionHead(nn.Module):
    """Original FlowmatchingActionHead using standard MLP encoders and decoders."""

    def __init__(self, full_config):

        super().__init__()
        config = full_config.framework.action_model
        self.hidden_size = config.hidden_size
        self.full_config = full_config
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        vl_input_dim = diffusion_model_cfg.get('cross_attention_dim', 2048)
        output_dim=diffusion_model_cfg['output_dim']





        action_cfg = TimeAwareActionHead_Config(
            action_dim=self.input_embedding_dim,
            vl_input_dim=vl_input_dim,
            output_dim=output_dim,
            num_target_vision_tokens=config.num_target_vision_tokens,
        )
        self.model = TimeAwareActionHead(config=action_cfg)


        self.action_dim = config.action_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps




        self.state_encoder = MLP(
            input_dim=config.state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        ) if config.state_dim else None




        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )



        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )




        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)



        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)




        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config



    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self,
                vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None,
                embodiment_tag: torch.Tensor = None, encoder_attention_mask=None, action_mask=None, use_vjepa2ac=False, repeated_diffusion_steps=-1
                ):

        device = vl_embs.device

        # 1. 对action加噪得到noisy_trajectory，并生成降噪目标velocity
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)

        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)

        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions

        velocity = actions - noise


        # 2. 编码state
        state_features = self.state_encoder(state) if state is not None else None



        # 3. 将t离散化并，对noised_action做时间编码
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()


        action_features = self.action_encoder(noisy_trajectory, t_discretized)


        # 4. 位置编码

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # 5. 将state-token、future-tokens和noised-action-tokens拼接得到sa-tokens
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(action_features.shape[0], -1, -1)


        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)


        # 6. PerceiverHead前向传播
        # (1) Perceiver-Head和DiT-Head的区别在于前者是cross_attn+ffn，而后者兼具self_attn+cross_attn+ffn，
        # (2) 此外，与MoT使用逐层的vl-tokens作为qkv，然后和action-tokens拼接一起计算attn不同，
        #     Perceiver首先分别对sa-tokens和vl-tokens做AdaLN，然后将sa-tokens作为q、将concat(sa-tokens,vl-tokens)作为kv，
        #     这里的sa-tokens就是state-tokens、future-tokens和action-tokens拼接起来的，
        model_output = self.model(
            latents=sa_embs,
            visual_language_states=vl_embs,
            timestep=t_discretized,
        )




        pred = self.action_decoder(model_output[:, -actions.shape[1]:, :])

        pred_actions = pred


        # 7. 计算action_loss
        # 若提供了action_mask则只在有效动作维度上计算loss、若未提供action_mask则在所有动作维度上计算loss；

        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(device=pred_actions.device, dtype=pred_actions.dtype)
            else:
                action_mask = action_mask.to(device=pred_actions.device, dtype=pred_actions.dtype)

            loss = ((pred_actions - velocity)**2) * action_mask
            loss = loss.sum() / action_mask.sum()
        else:
            # Fallback to mean loss if no mask is provided
            loss = ((pred_actions - velocity)**2).mean()

        # 8. 分离target_emb
        # 其中判断条件为，列表中的任意两个tensor_a,b之间差距小于阈值：|a-b|≤atol+rtol*max(|a|,|b|)
        target_emb = model_output[:, 1:1+self.future_tokens.weight.shape[0], :]

        target_emb_groups = target_emb.chunk(repeated_diffusion_steps, dim=0)
        all_equal = all(torch.allclose(target_emb_groups[0], target_emb_groups[i], atol=1e-5, rtol=0) for i in range(1, repeated_diffusion_steps))





        assert all_equal, "/path/to/local-resource is ERROR"
        target_emb_stacked = torch.stack(target_emb_groups, dim=0)  # 形状: (N, *)
        target_emb = torch.mean(target_emb_stacked, dim=0)




        if use_vjepa2ac:
            return loss, target_emb
        else:
            return loss

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None, embodiment_tag: torch.Tensor = None) -> torch.Tensor:


        # 1. 生成随机噪声
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )





        # 2. 编码state
        state_features = self.state_encoder(state) if state is not None else None



        # 3. 迭代降噪
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps


        for t in range(num_steps):

            t_cont = t / float(num_steps)

            t_discretized = int(t_cont * self.num_timestep_buckets)


            # 3.1 将t离散化并，对noised_action做时间编码
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)


            action_features = self.action_encoder(actions, timesteps_tensor)


            # 3.2 位置编码

            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # 3.3 将state-token、future-tokens和noised-action-tokens拼接得到sa-tokens
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
                if state_features is not None else torch.cat((future_tokens, action_features), dim=1)

            # 3.4 PerceiverHead前向传播
            model_output = self.model(
                latents=sa_embs,
                visual_language_states=vl_embs,
                timestep=timesteps_tensor,
            )

            pred = self.action_decoder(model_output)


            pred_velocity = pred[:, -self.action_horizon:]


            # 3.5 降噪
            actions = actions + dt * pred_velocity



        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class CategorySpecificFlowmatchingActionHead(nn.Module):
    """Category-specific FlowmatchingActionHead using category-specific encoders and decoders."""

    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.hidden_size = config.hidden_size
        self.full_config = full_config
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]

        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        vl_input_dim = diffusion_model_cfg.get('cross_attention_dim', 2048)
        action_cfg = TimeAwareActionHead_Config(action_dim=self.input_embedding_dim, vl_input_dim=vl_input_dim)

        self.model = TimeAwareActionHead(config=action_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        ) if config.state_dim else None

        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self,
                vl_embs: torch.Tensor,
                actions: torch.Tensor,
                state: torch.Tensor = None,
                embodiment_tag: torch.Tensor = None,
                encoder_attention_mask=None,
                action_mask=None):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        embodiment_tag: shape (B,) - required for category-specific operations
        action_mask: shape (B, future_action_window_size, D_action) - optional mask for valid action dimensions
        """
        device = vl_embs.device

        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]     # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_tag)

        # embed state
        state_features = self.state_encoder(state, embodiment_tag) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)

        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            latents=sa_embs,
            visual_language_states=vl_embs,
            timestep=t_discretized,
        )
        pred = self.action_decoder(model_output, embodiment_tag)
        pred_actions = pred[:, -actions.shape[1]:]

        # Use action_mask if provided, otherwise compute mean loss over all dimensions
        if action_mask is not None:
            # Ensure action_mask is on the same device and has the same dtype as pred_actions
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(device=pred_actions.device, dtype=pred_actions.dtype)
            else:
                action_mask = action_mask.to(device=pred_actions.device, dtype=pred_actions.dtype)

            # Compute masked loss: only consider valid action dimensions
            loss = ((pred_actions - velocity)**2) * action_mask
            loss = loss.sum() / action_mask.sum()
        else:
            # Fallback to mean loss if no mask is provided
            loss = ((pred_actions - velocity)**2).mean()
        return loss

    @torch.no_grad()
    def predict_action(self,
                       vl_embs: torch.Tensor,
                       state: torch.Tensor = None,
                       embodiment_tag: torch.Tensor = 24) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        assert embodiment_tag is not None, "Embodiment tag is required for prediction"
        embodiment_tag = torch.tensor(embodiment_tag, device=device, dtype=torch.long) #
        # 如果是标量，扩展到 batch_size
        if embodiment_tag.dim() == 0:
            embodiment_tag = embodiment_tag.repeat(batch_size)

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state, embodiment_tag) if state is not None else None

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)     # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_tag)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
                if state_features is not None else torch.cat((future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                latents=sa_embs,
                visual_language_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_tag)

            pred_velocity = pred[:, -self.action_horizon:]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead or CategorySpecificFlowmatchingActionHead: Initialized action model.
    """
    action_model_config = config.framework.action_model
    use_category_specific = getattr(action_model_config, 'use_category_specific', False)

    print('use_category_specific', use_category_specific)
    if use_category_specific:
        return CategorySpecificFlowmatchingActionHead(full_config=config)
    else:
        return FlowmatchingActionHead(full_config=config)
