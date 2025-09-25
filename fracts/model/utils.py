import math
from typing import Literal

import torch
from torch import nn
from torch.nn import init
from torch.optim import Optimizer
import numpy as np

def add_weight_decay(model: nn.Module, weight_decay: float = 1e-5, skip_list=()):
    """Inherited from fractal generative model utils.misc.add_weight_decay."""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def adjust_learning_rate(optimizer: Optimizer, epoch: float, lr: float, min_lr: float, warmup_epochs: int, epochs: int,
                         lr_schedule: Literal["cosine", "constant"]):
    """Decay the learning rate with half-cycle cosine after warmup. Inherited from fractal generative model
    utils.lr_sched.adjust_learning_rate."""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    else:
        if lr_schedule == "constant":
            lr = lr
        elif lr_schedule == "cosine":
            lr = min_lr + (lr - min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
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
            init.constant_(m.bias, 0.)
    elif isinstance(m, nn.Embedding):
        init.normal_(m.weight, 0.02)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            init.constant_(m.bias, 0.)
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
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * 
                        -(np.log(10000.0) / embed_dim))
    
    embeddings = torch.zeros(1, seq_len, embed_dim)
    embeddings[0, :, 0::2] = torch.sin(position * div_term)
    embeddings[0, :, 1::2] = torch.cos(position * div_term)
    return embeddings
    


def precompute_freqs_cis(seq_len: int, n_elem: int, base: float = 10000, cls_token_num=120):
    """Directly copied from baseline."""
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)  # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache])  # (cls_token_num+seq_len, head_dim // 2, 2)
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
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=attn_bias.device).tril(diagonal=0).unsqueeze(0)
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
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p, training=query.requires_grad)
    return (attn_weight @ v).to(query.dtype)


if __name__ == '__main__':

    pos_embeddings = get_sinusoidal_pos_embed(2, 128)
    #print pos embeddings of two tokens
    for i in range(2):
        print(pos_embeddings[0, i])