# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""DiT4DiT-compatible Perceiver action heads.

These heads keep the starVLA Perceiver action architecture, but use DiT4DiT's
action flow time convention: t=0 is clean action and t=1 is noise.
"""

import numpy as np
import torch

from starVLA.model.modules.action_model.PerceiverHead import (
    CategorySpecificFlowmatchingActionHead,
    FlowmatchingActionHead,
)


class DiT4DiTFlowmatchingActionHead(FlowmatchingActionHead):
    """DiT4DiT-compatible action flow direction."""

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return sample / self.config.noise_s

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        embodiment_tag: torch.Tensor = None,
        encoder_attention_mask=None,
        action_mask=None,
    ):
        device = vl_embs.device

        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * actions + t * noise
        velocity = noise - actions

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = self.state_encoder(state) if state is not None else None

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        model_output = self.model(
            latents=sa_embs,
            visual_language_states=vl_embs,
            timestep=t_discretized,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1]:]

        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(device=pred_actions.device, dtype=pred_actions.dtype)
            else:
                action_mask = action_mask.to(device=pred_actions.device, dtype=pred_actions.dtype)
            loss = ((pred_actions - velocity) ** 2) * action_mask
            return loss.sum() / action_mask.sum()
        return ((pred_actions - velocity) ** 2).mean()

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        embodiment_tag: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        for t in range(num_steps):
            t_cont = 1.0 - t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            model_output = self.model(
                latents=sa_embs,
                visual_language_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)

            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions - dt * pred_velocity
        return actions


class DiT4DiTCategorySpecificFlowmatchingActionHead(CategorySpecificFlowmatchingActionHead):
    """Category-specific DiT4DiT-compatible action flow direction."""

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return sample / self.config.noise_s

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        embodiment_tag: torch.Tensor = None,
        encoder_attention_mask=None,
        action_mask=None,
    ):
        device = vl_embs.device

        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * actions + t * noise
        velocity = noise - actions

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_tag)

        state_features = self.state_encoder(state, embodiment_tag) if state is not None else None

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        model_output = self.model(
            latents=sa_embs,
            visual_language_states=vl_embs,
            timestep=t_discretized,
        )
        pred = self.action_decoder(model_output, embodiment_tag)
        pred_actions = pred[:, -actions.shape[1]:]

        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).to(device=pred_actions.device, dtype=pred_actions.dtype)
            else:
                action_mask = action_mask.to(device=pred_actions.device, dtype=pred_actions.dtype)
            loss = ((pred_actions - velocity) ** 2) * action_mask
            return loss.sum() / action_mask.sum()
        return ((pred_actions - velocity) ** 2).mean()

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        embodiment_tag: torch.Tensor = 24,
    ) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        assert embodiment_tag is not None, "Embodiment tag is required for prediction"
        embodiment_tag = torch.tensor(embodiment_tag, device=device, dtype=torch.long)
        if embodiment_tag.dim() == 0:
            embodiment_tag = embodiment_tag.repeat(batch_size)

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state, embodiment_tag) if state is not None else None

        for t in range(num_steps):
            t_cont = 1.0 - t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_tag)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            model_output = self.model(
                latents=sa_embs,
                visual_language_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_tag)

            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions - dt * pred_velocity
        return actions


def get_action_model(config=None):
    action_model_config = config.framework.action_model
    use_category_specific = getattr(action_model_config, "use_category_specific", False)

    print("use_category_specific", use_category_specific, "head_variant", "dit4dit")
    if use_category_specific:
        return DiT4DiTCategorySpecificFlowmatchingActionHead(full_config=config)
    return DiT4DiTFlowmatchingActionHead(full_config=config)
