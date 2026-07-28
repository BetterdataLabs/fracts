import gc
import logging
import math
import time
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.optim import AdamW, Optimizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def adjust_batch_size(
    initial_batch_size: int,
    dataloader_sample,
    model: torch.nn.Module,
    device: torch.device,
    min_batch_size: int = 1,
    max_batch_size: int = 512,
    memory_threshold: float = 0.80,
) -> int:
    """
    Automatically adjust batch size based on available GPU memory.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available, using initial batch size")
        return initial_batch_size

    # Store original states before any testing
    original_training_mode = model.training

    # Save original model state dict for complete restoration
    original_state_dict = {
        name: param.clone() for name, param in model.named_parameters()
    }
    original_buffer_dict = {
        name: buffer.clone() for name, buffer in model.named_buffers()
    }

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    total_memory, _ = _get_gpu_memory_info()
    if total_memory is None:
        logger.info("Could not get GPU memory info, using initial batch size")
        return initial_batch_size

    # Create a single test optimizer that we'll reuse
    test_optimizer = AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.95))

    def reset_model_state():
        """Reset model to original state."""
        # Clear all gradients
        model.zero_grad(set_to_none=True)
        test_optimizer.zero_grad(set_to_none=True)

        # Reset model parameters to original values
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(original_state_dict[name])
            for name, buffer in model.named_buffers():
                buffer.copy_(original_buffer_dict[name])

    # Get baseline memory usage (model + optimizer only)
    cleanup_test_resources()
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
    logger.info(
        f"Baseline memory usage (model + optimizer): {baseline_allocated:.1f} MB"
    )

    # Test function to check if batch size works
    def test_batch_size(test_batch_size: int) -> bool:
        try:
            cleanup_test_resources()
            reset_model_state()

            # Set model to training mode for testing
            model.train()

            # Create test batch with the specified size
            static_ids, static_data, ts_data, len_indicator = dataloader_sample

            current_batch = len(static_data)
            # Replicate data to match test batch size
            if current_batch < test_batch_size:
                repeat_factor = (test_batch_size + current_batch - 1) // current_batch

                static_data = static_data.clone().detach()
                ts_data = ts_data.clone().detach()
                len_indicator = len_indicator.clone().detach()

                static_data = static_data.repeat(repeat_factor, 1)[:test_batch_size]
                ts_data = ts_data.repeat(repeat_factor, 1, 1)[:test_batch_size]
                len_indicator = len_indicator.repeat(repeat_factor, 1)[:test_batch_size]
            else:
                static_data = static_data[:test_batch_size].clone().detach()
                ts_data = ts_data[:test_batch_size].clone().detach()
                len_indicator = len_indicator[:test_batch_size].clone().detach()

            # Move data to device
            static_data = static_data.to(device, non_blocking=False)
            ts_data = ts_data.to(device, non_blocking=False)
            len_indicator = len_indicator.to(device, non_blocking=False)

            # Measure memory before forward/backward pass
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated(device) / (1024**2)  # MB

            # Test forward pass
            with torch.cuda.amp.autocast():
                loss = model(ts_data, len_indicator, static_data)

            # Backward pass to test gradient memory requirements
            loss.loss.backward()

            # Force synchronization to ensure all operations complete
            torch.cuda.synchronize()

            # Measure peak memory usage during forward/backward
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB

            # Calculate actual memory used by this batch (peak - baseline)
            batch_memory_used = peak_memory - baseline_allocated

            # Calculate total memory usage ratio (peak / total)
            memory_usage_ratio = peak_memory / total_memory

            logger.debug(
                f"Batch size {test_batch_size}: "
                f"Peak memory: {peak_memory:.1f} MB, "
                f"Batch memory: {batch_memory_used:.1f} MB, "
                f"Usage ratio: {memory_usage_ratio:.2%}"
            )

            # Reset peak memory counter for next test
            torch.cuda.reset_peak_memory_stats(device)

            # Clean up test data
            del loss, static_data, ts_data, len_indicator

            # Return success if under threshold
            return memory_usage_ratio < memory_threshold

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.debug(f"Batch size {test_batch_size} failed with OOM: {str(e)}")
            cleanup_test_resources()
            torch.cuda.reset_peak_memory_stats(device)
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error testing batch size {test_batch_size}: {str(e)}"
            )
            cleanup_test_resources()
            torch.cuda.reset_peak_memory_stats(device)
            return False

    try:
        # Reset peak memory stats before starting
        torch.cuda.reset_peak_memory_stats(device)

        # Search for optimal batch size
        logger.info(
            f"Searching for optimal batch size between {min_batch_size} and {max_batch_size}"
        )
        logger.info(
            f"Total GPU memory: {total_memory:.1f} MB, Target threshold: {memory_threshold:.1%}"
        )

        cleanup_test_resources()

        # First, ensure even the minimum batch size works
        if not test_batch_size(min_batch_size):
            logger.error(f"Even minimum batch size {min_batch_size} causes OOM.")
            return min_batch_size

        # Binary search for optimal batch size
        low, high = min_batch_size, max_batch_size
        optimal_batch_size = min_batch_size

        while low <= high:
            mid = (low + high) // 2
            cleanup_test_resources()
            time.sleep(0.2)  # Brief pause for GPU to settle

            logger.info(f"Testing batch size: {mid}")
            if test_batch_size(mid):
                optimal_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1
                cleanup_test_resources()
                time.sleep(0.5)  # Longer wait after OOM

        logger.info(f"Optimal batch size found: {optimal_batch_size}")

        return optimal_batch_size
    finally:
        # Complete restoration to original state
        reset_model_state()
        model.train(original_training_mode)

        # Clean up optimizer and final cleanup
        del test_optimizer
        cleanup_test_resources()

        # Final cleanup of stored states
        del original_state_dict, original_buffer_dict

        logger.debug("Model fully restored to original state after batch size testing")


def _get_gpu_memory_info():
    """Get GPU memory information in MB."""
    if not torch.cuda.is_available():
        return None, None

    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / (
        1024**2
    )  # MB
    allocated_memory = torch.cuda.memory_allocated(device) / (1024**2)  # MB
    free_memory = total_memory - allocated_memory

    return total_memory, free_memory


def cleanup_test_resources():
    """Clean up GPU memory after each test."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


def add_weight_decay(model: nn.Module, weight_decay: float = 1e-5, skip_list=()):
    """Inherited from fractal generative model utils.misc.add_weight_decay."""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
            or "diffloss" in name
        ):
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def adjust_learning_rate(
    optimizer: Optimizer,
    epoch: float,
    lr: float,
    min_lr: float,
    warmup_epochs: int,
    epochs: int,
    lr_schedule: Literal["cosine", "constant"],
):
    """Decay the learning rate with half-cycle cosine after warmup. Inherited from fractal generative model
    utils.lr_sched.adjust_learning_rate."""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    else:
        if lr_schedule == "constant":
            lr = lr
        elif lr_schedule == "cosine":
            lr = min_lr + (lr - min_lr) * 0.5 * (
                1.0
                + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs))
            )
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def init_weights(m: nn.Module):
    """Extracted init weights from original fractal generative model."""
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Embedding):
        init.normal_(m.weight, 0.02)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
        if m.weight is not None:
            init.constant_(m.weight, 1.0)


def get_sinusoidal_pos_embed(seq_len: int, embed_dim: int) -> torch.Tensor:
    """
    Create sinusoidal position embedding.

    Parameters
    ----------
    seq_len : int
        The sequence length to create position embeddings.
    dim : int
        The embedded dimension.

    Returns
    -------
    torch.Tensor
        The positional embeddings.
    """

    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float)
        * -(np.log(10000.0) / embed_dim)
    )

    embeddings = torch.zeros(1, seq_len, embed_dim)
    embeddings[0, :, 0::2] = torch.sin(position * div_term)
    embeddings[0, :, 1::2] = torch.cos(position * div_term)
    return embeddings


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: float = 10000, cls_token_num=120
):
    """Directly copied from baseline."""
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack(
        [freqs_cis.real, freqs_cis.imag], dim=-1
    )  # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache


def find_multiple(n: int, k: int):
    """Directly copied from baseline."""
    if n % k == 0:
        return n
    return n + k - (n % k)


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    """Adapted: attn_mask always given and each row may have different mask."""
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(query.size(0), L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = (
            torch.ones(L, S, dtype=torch.bool, device=attn_bias.device)
            .tril(diagonal=0)
            .unsqueeze(0)
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(1).expand(-1, L, -1)
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.unsqueeze(1)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def scaled_dot_product_attention_new(
    query, key, value, attn_mask=None, dropout_p=0.0, scale=None
) -> torch.Tensor:
    """Compute scaled dot product attention with explicit mask handling."""
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    with torch.cuda.amp.autocast(enabled=False):
        q = query.float()
        k = key.float()
        v = value.float()
        attn_weight = q @ k.transpose(-2, -1) * scale_factor

    if attn_mask is None:
        raise ValueError("Attention mask must be provided.")

    if attn_mask.dtype != attn_weight.dtype:
        attn_mask = attn_mask.to(attn_weight.dtype)

    attn_weight += attn_mask.unsqueeze(1)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p > 0:
        attn_weight = torch.nn.functional.dropout(
            attn_weight, p=dropout_p, training=query.requires_grad
        )
    return (attn_weight @ v).to(query.dtype)


if __name__ == "__main__":

    pos_embeddings = get_sinusoidal_pos_embed(2, 128)
    # print pos embeddings of two tokens
    for i in range(2):
        print(pos_embeddings[0, i])
