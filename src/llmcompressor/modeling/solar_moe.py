# coding=utf-8
# Copyright 2025 The Upstage AI and HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from llmcompressor.modeling.moe_context import MoECalibrationModule
from llmcompressor.utils.dev import skip_weights_initialize

if TYPE_CHECKING:
    from transformers.models.solar_open.modeling_solar_open import SolarOpenMoE


@MoECalibrationModule.register("SolarOpenMoE")
class CalibrationSolarOpenMoE(MoECalibrationModule):
    """
    Calibration version of SolarOpenMoE that unfuses 3D expert parameters into
    individual MLP modules (nn.Linear) so they can be individually quantized.
    Sends all tokens to all experts during calibration.

    SolarOpenMoE differs from Qwen3_5MoeSparseMoeBlock only in:
      - routing: sigmoid + grouped top-k (DeepSeek style) instead of softmax
      - shared experts: a plain additive ``shared_experts`` MLP (no gate)

    is_permanent = True because the unfused structure must persist for
    quantization to target the individual nn.Linear expert weights.
    """

    is_permanent = True

    def __init__(
        self,
        original: SolarOpenMoE,
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        text_config = getattr(config, "text_config", config)

        self.calibrate_all_experts = calibrate_all_experts

        # Routing parameters
        self.n_routed_experts = text_config.n_routed_experts
        self.n_group = text_config.n_group
        self.topk_group = text_config.topk_group
        self.norm_topk_prob = text_config.norm_topk_prob
        self.routed_scaling_factor = text_config.routed_scaling_factor
        self.top_k = text_config.num_experts_per_tok
        self.hidden_size = text_config.hidden_size

        # Keep the original router as-is: it holds both the gate weight and the
        # `e_score_correction_bias` buffer that `route_tokens_to_experts` needs.
        self.gate = original.gate
        self.shared_experts = original.shared_experts
        self.experts = SequentialSolarOpenExperts(text_config, original.experts)

    def route_tokens_to_experts(self, router_logits):
        # Copied from SolarOpenMoE.route_tokens_to_experts
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(
                -1, self.n_group, self.n_routed_experts // self.n_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(
            ~score_mask.bool(), 0.0
        )
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[
            1
        ]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]

        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)

        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        # expert mask: (num_experts, top_k, num_tokens)
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=self.n_routed_experts
        ).permute(2, 1, 0)

        final_hidden_states = torch.zeros_like(
            hidden_states_reshaped, dtype=topk_weights.dtype
        )

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, token_idx = torch.where(expert_mask[expert_idx])

            if self.calibrate_all_experts:
                # all tokens pass through every expert (Hessian sees full data)
                expert_out = expert_layer(hidden_states_reshaped)[token_idx]
            else:
                # normal routing: only the tokens assigned to this expert
                expert_out = expert_layer(hidden_states_reshaped[token_idx])

            if len(token_idx) > 0:
                current_hidden_states = expert_out * topk_weights[token_idx, idx, None]
                final_hidden_states.index_add_(
                    0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
                )

        final_hidden_states = final_hidden_states.type(hidden_states.dtype).view(
            *orig_shape
        )
        # shared experts (plain additive, no gate)
        final_hidden_states = final_hidden_states + self.shared_experts(residuals)
        return final_hidden_states

    def restore(self, original: torch.nn.Module) -> torch.nn.Module:
        return self


class SequentialSolarOpenExperts(torch.nn.ModuleList):
    """
    Unfuses 3D expert parameter tensors into individual SolarOpenMLP modules so
    that each expert's weights are nn.Linear and can be targeted by quantization
    with targets="Linear".

    Fused layout (see ``SolarOpenNaiveMoe.forward``):
      - ``gate_up_proj[i]``: ``(2 * moe_intermediate, hidden)``; used as a linear
        weight, then ``chunk(2, dim=-1)`` on the output => rows ``[:I]`` are gate,
        rows ``[I:]`` are up.
      - ``down_proj[i]``: ``(hidden, moe_intermediate)``; used directly.
    Both are already in ``(out_features, in_features)`` order, so no transpose.
    """

    def __init__(self, config, original):
        from transformers.models.solar_open.modeling_solar_open import SolarOpenMLP

        self.num_experts = original.gate_up_proj.shape[0]
        intermediate_size = config.moe_intermediate_size

        with skip_weights_initialize():
            super().__init__(
                [
                    SolarOpenMLP(config, intermediate_size=intermediate_size)
                    for _ in range(self.num_experts)
                ]
            )

        gate_up_data = original.gate_up_proj.data  # [num_experts, 2*inter, hidden]
        down_data = original.down_proj.data  # [num_experts, hidden, inter]

        for i in range(self.num_experts):
            gate_up = gate_up_data[i]  # [2*intermediate, hidden]
            down = down_data[i]  # [hidden, intermediate]

            # gate_up_proj stores [gate; up] stacked along dim 0
            # nn.Linear weight is [out_features, in_features]
            self[i].gate_proj.weight.data = (
                gate_up[:intermediate_size, :].clone().contiguous()
            )
            self[i].up_proj.weight.data = (
                gate_up[intermediate_size:, :].clone().contiguous()
            )
            self[i].down_proj.weight.data = down.clone().contiguous()
