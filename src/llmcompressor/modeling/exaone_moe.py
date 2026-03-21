# Copyright 2026 The LG AI Research and HuggingFace Inc. team.
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

import torch

from transformers.models.exaone_moe.configuration_exaone_moe import ExaoneMoeConfig
from transformers.models.exaone_moe.modeling_exaone_moe import (
    ExaoneMoeSparseMoEBlock as OriginalExaoneMoeSparseMoEBlock,
)

from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("ExaoneMoeSparseMoEBlock")
class CalibrationExaoneMoeSparseMoEBlock(MoECalibrationModule):
    """
    Calibration version of ExaoneMoeSparseMoEBlock that sends all tokens
    to all experts for proper Hessian collection during quantization.
    """

    is_permanent = True

    def __init__(
        self,
        original: OriginalExaoneMoeSparseMoEBlock,
        config: ExaoneMoeConfig,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = config
        self.gate = original.gate
        self.experts = original.experts
        self.shared_experts = original.shared_experts
        self.calibrate_all_experts = calibrate_all_experts

        self.n_routed_experts = config.num_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(
                -1, self.n_group, self.n_routed_experts // self.n_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[1]
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
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)

        # Begin MoE
        final_hidden_states = torch.zeros_like(
            hidden_states, dtype=topk_weights.dtype
        )
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=len(self.experts)
        )
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx, expert in enumerate(self.experts):
            token_indices, weight_indices = torch.where(expert_mask[expert_idx])
            has_tokens = token_indices.numel() > 0

            if self.calibrate_all_experts:
                expert_output = expert(hidden_states)

                if has_tokens:
                    expert_weights = topk_weights[token_indices, weight_indices]
                    routed_output = (
                        expert_output[token_indices]
                        * expert_weights.unsqueeze(-1)
                    )
                    final_hidden_states.index_add_(
                        0, token_indices, routed_output
                    )
            else:
                if has_tokens:
                    expert_output = expert(hidden_states[token_indices])
                    expert_weights = topk_weights[token_indices, weight_indices]
                    routed_output = (
                        expert_output * expert_weights.unsqueeze(-1)
                    )
                    final_hidden_states.index_add_(
                        0, token_indices, routed_output
                    )
        # End MoE

        hidden_states = final_hidden_states.type(hidden_states.dtype).view(
            *orig_shape
        )
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states
