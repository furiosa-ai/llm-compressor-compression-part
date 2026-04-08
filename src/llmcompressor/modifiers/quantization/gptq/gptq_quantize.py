import math
from copy import copy

import torch
import transformers
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    fake_quantize,
)
from compressed_tensors.utils import update_offload_parameter
from loguru import logger

from llmcompressor.modifiers.utils import SPARSITY_THRESHOLD
from llmcompressor.observers.base import Observer
from llmcompressor.pytorch.utils.helpers import tensor_sparsity

GPTQ_PRECISION = torch.float32

__all__ = [
    "make_empty_hessian",
    "accumulate_hessian",
    "accumulate_imatrix",
    "refine_scale_with_importance",
    "quantize_weight",
]


def make_empty_hessian(
    module: torch.nn.Module, device: torch.device | None = None
) -> torch.Tensor:
    weight = module.weight
    num_columns = weight.shape[1]
    device = device if device is not None else weight.device
    return torch.zeros((num_columns, num_columns), device=device, dtype=GPTQ_PRECISION)


def accumulate_hessian(
    inp: torch.Tensor,
    module: torch.nn.Module,
    H: torch.Tensor | None,
    num_samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inp = inp.to(device=H.device)
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)

    num_added = inp.shape[0]

    match module:
        case torch.nn.Linear() | transformers.Conv1D():
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        case torch.nn.Conv2d():
            unfold = torch.nn.Unfold(
                module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

    num_samples += num_added

    inp = inp.to(dtype=GPTQ_PRECISION)
    inp = math.sqrt(2) * inp
    H += inp.matmul(inp.t())

    return H, num_samples


def accumulate_imatrix(
    inp: torch.Tensor,
    module: torch.nn.Module,
    imatrix: torch.Tensor | None,
) -> torch.Tensor:
    """
    Accumulate importance matrix: channel-wise sum of squared input activations.
    Inspired by SignRound V2 / llama.cpp importance matrix.

    :param inp: input activation tensor
    :param module: module being calibrated
    :param imatrix: existing importance matrix to accumulate into, or None
    :return: updated importance matrix of shape (num_input_channels,)
    """
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)

    # Handle different module types (mirror accumulate_hessian)
    match module:
        case torch.nn.Linear() | transformers.Conv1D():
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
        case torch.nn.Conv2d():
            unfold = torch.nn.Unfold(
                module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2]).flatten(1).t()

    inp = inp.to(dtype=GPTQ_PRECISION)

    # Sum of squares per input channel
    squared = torch.sum(inp**2, dim=0)

    if imatrix is None:
        return squared
    else:
        return imatrix + squared.to(imatrix.device)


def refine_scale_with_importance(
    W: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_args: QuantizationArgs,
    imatrix: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    search_range: tuple[float, float] = (0.5, 1.5),
    search_step: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Refine quantization scale using importance-weighted grid search.
    Inspired by SignRound V2's scale initialization with importance matrix.

    For each group, searches over scale multipliers in [search_range[0], search_range[1]]
    to minimize importance-weighted quantization error.

    :param W: weight tensor (num_rows, num_columns) in GPTQ_PRECISION
    :param scale: initial scale from observer
    :param zero_point: initial zero point from observer
    :param quant_args: quantization arguments
    :param imatrix: importance matrix of shape (num_columns,)
    :param global_scale: optional global scale for TENSOR_GROUP strategy
    :param search_range: (min_factor, max_factor) for grid search
    :param search_step: step size for grid search
    :return: refined (scale, zero_point)
    """
    strategy = quant_args.strategy
    group_size = quant_args.group_size

    if strategy not in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        return scale, zero_point

    if group_size is None or group_size <= 0:
        return scale, zero_point

    if not quant_args.symmetric:
        logger.warning(
            "refine_scale_with_importance: asymmetric quantization requires "
            "zero_point recomputation; skipping refinement"
        )
        return scale, zero_point

    # Ensure imatrix is on the same device as W
    imatrix = imatrix.to(device=W.device)

    num_rows, num_columns = W.shape
    num_groups = num_columns // group_size
    if num_groups == 0:
        return scale, zero_point

    effective_cols = num_groups * group_size

    # Reshape weight and importance into groups
    W_g = W[:, :effective_cols].reshape(num_rows, num_groups, group_size).float()
    imatrix_g = imatrix[:effective_cols].reshape(1, num_groups, group_size).float()

    # Normalize importance to avoid numerical issues
    imatrix_max = imatrix_g.amax(dim=-1, keepdim=True).clamp(min=1e-8)
    imatrix_g = imatrix_g / imatrix_max

    # Compute initial importance-weighted quantization error
    W_q_init = fake_quantize(
        W, scale, zero_point, quant_args, global_scale=global_scale
    )
    W_q_g = W_q_init[:, :effective_cols].reshape(num_rows, num_groups, group_size).float()
    init_loss = ((W_g - W_q_g) ** 2 * imatrix_g).sum(dim=-1).sum(dim=0)  # (num_groups,)
    best_loss = init_loss.clone()

    best_scale = scale.clone()

    # Grid search over multiplicative factors
    factor = search_range[0]
    while factor <= search_range[1] + 1e-9:
        if abs(factor - 1.0) < 1e-9:
            factor += search_step
            continue

        candidate_scale = scale * factor
        W_q = fake_quantize(
            W, candidate_scale, zero_point, quant_args, global_scale=global_scale
        )
        W_q_g = W_q[:, :effective_cols].reshape(num_rows, num_groups, group_size).float()
        loss = ((W_g - W_q_g) ** 2 * imatrix_g).sum(dim=-1).sum(dim=0)  # (num_groups,)

        improved = loss < best_loss
        if improved.any():
            best_scale[:, improved] = candidate_scale[:, improved]
            best_loss[improved] = loss[improved]

        factor += search_step

    num_improved = (best_loss < init_loss).sum().item()
    logger.info(
        f"  Scale refinement: groups improved {num_improved}/{num_groups}"
    )

    return best_scale, zero_point


def quantize_weight(
    module: torch.nn.Module,
    quant_args: QuantizationArgs,
    hessian: torch.Tensor,
    blocksize: int = 128,
    percdamp: float = 0.01,
    imatrix: torch.Tensor | None = None,
) -> tuple[float, dict]:
    """
    Quantize a module weight according to the GPTQ algorithm

    :param module: module with weight being quantized
    :param quant_args: quantization arguments used to find quantization parameters
    :param hessian: preaccumulated hessian for quantization
    :param blocksize: chunk size of quantization updates
    :param percdamp: dampening factor on hessian diagonal
    :param imatrix: optional importance matrix (channel-wise sum of squared activations)
        for importance-weighted scale initialization. If provided, scale is refined
        using grid search before the GPTQ weight update loop.
    :return: tuple of (loss, q_param_dict) where q_param_dict contains
        weight, weight_scale, weight_zero_point, and optionally weight_g_idx
    """
    strategy = quant_args.strategy
    actorder = quant_args.actorder
    global_scale = getattr(module, "weight_global_scale", None)
    final_shape = module.weight.shape
    final_dtype = module.weight.dtype
    W = module.weight.clone()
    H = hessian

    # create observer for calculating quantization parameters
    observer = Observer.load_from_registry(
        quant_args.observer if quant_args.observer else "memoryless_minmax",
        base_name="weight",
        args=quant_args,
        module=module,
    )

    # standardize shape and dtype
    match module:
        case torch.nn.Conv2d():
            W = W.flatten(1)
        case transformers.Conv1D():
            W.transpose_(0, 1)
    W = W.to(dtype=GPTQ_PRECISION)
    num_rows = W.shape[0]
    num_columns = W.shape[1]

    # generate scale, should include tensor group / use global scale
    if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        # mapping from column index to group index
        g_idx = (
            torch.arange(num_columns, device=W.device, dtype=torch.int)
            // quant_args.group_size
        )

        if actorder == ActivationOrdering.GROUP:
            # permute by activation order first, then update groups
            W, H, perm = _apply_activation_ordering(W, H)
            if imatrix is not None:
                imatrix = imatrix[perm]
            update_offload_parameter(module, "weight_g_idx", g_idx)
            scale, zero_point = observer(W)
            if imatrix is not None:
                scale, zero_point = refine_scale_with_importance(
                    W, scale, zero_point, quant_args, imatrix, global_scale
                )

            # use identity g_idx (invert permutation later)

        elif actorder == ActivationOrdering.WEIGHT:
            # update groups first, then permute by activation order
            scale, zero_point = observer(W)
            if imatrix is not None:
                scale, zero_point = refine_scale_with_importance(
                    W, scale, zero_point, quant_args, imatrix, global_scale
                )
            W, H, perm = _apply_activation_ordering(W, H)

            # permute g_idx to maintain identity mapping after unpermutation
            g_idx = g_idx[perm]

        else:
            scale, zero_point = observer(W)
            if imatrix is not None:
                scale, zero_point = refine_scale_with_importance(
                    W, scale, zero_point, quant_args, imatrix, global_scale
                )
    else:
        scale, zero_point = observer(W)

    # sparsity mask
    sparsity = tensor_sparsity(W)
    preserve_zeros = sparsity >= SPARSITY_THRESHOLD
    W_nz_mask = (
        (~torch.isclose(W, torch.zeros(1, device=W.device).float())).float()
        if preserve_zeros
        else None
    )

    losses = torch.zeros(num_rows, device=module.weight.device)

    # mask dead hessian values
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # compute inverse hessian in place to save memory
    try:
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
    except torch._C._LinAlgError:
        logger.warning(
            "Failed to invert hessian due to numerical instability. Consider "
            "increasing GPTQModifier.dampening_frac, increasing the number "
            "of calibration samples, or shuffling the calibration dataset. "
            "Falling back to round-to-nearest for this module."
        )
        Hinv = H = torch.eye(num_columns, dtype=H.dtype, device=H.device)

    # See section 3.4 of https://arxiv.org/abs/2203.07259
    for i1 in range(0, num_columns, blocksize):
        i2 = min(i1 + blocksize, num_columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        if preserve_zeros:
            W1_nz_mask = W_nz_mask[:, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = w.clone()

            # quantize column
            if strategy == QuantizationStrategy.TENSOR:
                q = fake_quantize(
                    q, scale, zero_point, quant_args, global_scale=global_scale
                )
            elif strategy == QuantizationStrategy.CHANNEL:
                q = fake_quantize(
                    q,
                    scale[:, 0],
                    zero_point[:, 0],
                    quant_args,
                    global_scale=global_scale,
                )
            # apply global scale to scale quant scale
            elif strategy in (
                QuantizationStrategy.GROUP,
                QuantizationStrategy.TENSOR_GROUP,
            ):
                # get the group index for the current column
                column_idx = i1 + i
                group_index = g_idx[column_idx]

                # Since we're only applying quantization to a slice, this
                # ends up being a channelwise application
                altered_qargs = copy(quant_args)
                altered_qargs.strategy = QuantizationStrategy.CHANNEL

                q = fake_quantize(
                    q,
                    scale[:, group_index],
                    zero_point[:, group_index],
                    altered_qargs,
                    global_scale=global_scale,
                )
            else:
                raise ValueError(
                    f"Quantization strategy is not supported for GPTQ: {strategy}"
                )

            # propagate column error
            Q1[:, i] = q
            losses1[:, i] = (w - q) ** 2 / d**2

            err1 = (w - q) / d
            w1_err = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            if preserve_zeros:
                W1[:, i:] -= w1_err * W1_nz_mask[:, i:]
            else:
                W1[:, i:] -= w1_err
            Err1[:, i] = err1

        # propagate block error
        W[:, i1:i2] = Q1
        losses += torch.sum(losses1, 1) / 2

        w_err = Err1.matmul(Hinv[i1:i2, i2:])
        if preserve_zeros:
            W[:, i2:] -= w_err * W_nz_mask[:, i2:]
        else:
            W[:, i2:] -= w_err

    has_gidx = False
    if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        if actorder == ActivationOrdering.WEIGHT:
            # restore original permutation
            invperm = torch.argsort(perm)
            W = W[:, invperm]

        elif actorder == ActivationOrdering.GROUP:
            # restore original permutation
            invperm = torch.argsort(perm)
            W = W[:, invperm]
            g_idx = g_idx[invperm]

            # only save g_idx if mapping is not identity
            has_gidx = True

    if not has_gidx:
        g_idx = None

    if isinstance(module, transformers.Conv1D):
        W.transpose_(0, 1)
    W = W.reshape(final_shape).to(final_dtype)

    loss = torch.sum(losses).item()
    q_param_dict = {
        "weight": W,
        "weight_scale": scale.to(dtype=final_dtype),
        "weight_zero_point": zero_point.to(dtype=quant_args.zp_dtype),
    }
    if g_idx is not None:
        q_param_dict["weight_g_idx"] = g_idx
    return (loss, q_param_dict)


def _apply_activation_ordering(
    W: torch.Tensor, H: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Permute weight and hessian in order of greatest output activations

    :param W: weight to permute
    :param H: hessian used to determine activation ordering
    :return: permuted weight, permuted hessian, permutation map
    """
    perm = torch.argsort(torch.diag(H), descending=True)
    return W[:, perm], H[perm][:, perm], perm
