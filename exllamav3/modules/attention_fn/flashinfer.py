"""FlashInfer paged-attention dispatch for the pre-Ampere EXL3 fallback path.

The standard FlashInfer paged prefill wrapper directly accepts ExLlama's fp16
``(pages, 256, kv_heads, head_dim)`` cache layout. QSA sparse attention bypasses the
normal dispatcher before reaching this module, and decode remains on BC/Triton.
"""

import os

import torch

from ...constants import PAGE_SIZE
from ...ext import exllamav3_ext as ext
from .common import AttnArgs

try:
    import flashinfer

    has_flashinfer = True
except (ModuleNotFoundError, ImportError):
    flashinfer = None
    has_flashinfer = False


_states = {}


def _enabled(device: torch.device) -> bool:
    """Enable automatically on pre-Ampere; force with EXL3_FLASHINFER=1."""
    mode = os.environ.get("EXL3_FLASHINFER", "auto").lower()
    if mode in ("0", "false", "off", "no"):
        return False
    if mode in ("1", "true", "on", "yes"):
        return True
    return torch.cuda.get_device_capability(device)[0] < 8


def _state_for(device: torch.device):
    key = device.index
    state = _states.get(key)
    if state is None:
        workspace_mb = int(os.environ.get("EXL3_FLASHINFER_WORKSPACE_MB", "16"))
        if workspace_mb <= 0:
            raise ValueError("EXL3_FLASHINFER_WORKSPACE_MB must be positive")
        workspace = torch.empty(
            workspace_mb * 1024 * 1024,
            dtype = torch.uint8,
            device = device,
        )
        state = _states[key] = (
            workspace,
            # This path is intentionally FlashAttention-2, not FlashInfer's evolving
            # automatic backend selection. FA2 is the supported fast backend on SM75.
            flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend = "fa2"),
        )
    return state


def fn_flashinfer_paged_prefill(args: AttnArgs) -> torch.Tensor | None:
    """Run dense fp16 cached prefill through FlashInfer when its layout is exact."""
    if (
        not has_flashinfer or
        not _enabled(args.q.device) or
        args.is_varlen() or
        not args.has_kv_cache() or
        args.q_len <= 16 or
        args.dim not in (64, 128, 256) or
        args.q.dtype != torch.float16 or
        args.k.dtype != torch.float16 or
        args.v.dtype != torch.float16 or
        args.k_cache.dtype != torch.float16 or
        args.v_cache.dtype != torch.float16 or
        args.k_cache.ndim != 4 or
        args.v_cache.ndim != 4 or
        args.k_cache.shape[1] != PAGE_SIZE or
        args.v_cache.shape[1] != PAGE_SIZE or
        args.non_causal_spans is not None or
        args.sinks is not None or
        args.window_size not in (None, -1) or
        args.softcap not in (None, 0, 0.0) or
        not args.q.is_contiguous() or
        not args.k.is_contiguous() or
        not args.v.is_contiguous()
    ):
        return None

    # FlashInfer reads the appended K/V from the paged cache, whereas ExLlama's
    # Triton path consumes the append tensors directly before committing them. Commit
    # here; CacheLayer_fp16.update_kv is intentionally a no-op after dispatch.
    try:
        _, wrapper = _state_for(args.q.device)
        total_lens = args.cache_seqlens + args.q_len
        pages_per_seq = torch.div(total_lens + PAGE_SIZE - 1, PAGE_SIZE, rounding_mode = "floor")
        last_page_len = total_lens - (pages_per_seq - 1) * PAGE_SIZE
        page_slots = torch.arange(
            args.block_table.shape[1],
            dtype = args.block_table.dtype,
            device = args.block_table.device,
        )
        page_mask = page_slots.unsqueeze(0) < pages_per_seq.unsqueeze(1)
        paged_kv_indices = args.block_table[page_mask].contiguous()
        paged_kv_indptr = torch.cat((
            torch.zeros((1,), dtype = args.cache_seqlens.dtype, device = args.q.device),
            torch.cumsum(pages_per_seq, dim = 0, dtype = torch.int32),
        ))
        qo_indptr = torch.arange(
            args.bsz + 1,
            dtype = args.cache_seqlens.dtype,
            device = args.q.device,
        ) * args.q_len
        wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            last_page_len,
            args.num_q_heads,
            args.num_kv_heads,
            args.dim,
            PAGE_SIZE,
            causal = args.causal,
            sm_scale = args.sm_scale,
        )
    except torch.cuda.OutOfMemoryError:
        return None

    ext.paged_kv_cache_update(
        args.k,
        args.v,
        args.k_cache,
        args.v_cache,
        args.block_table,
        args.cache_seqlens,
    )
    o = wrapper.run(args.q.reshape(-1, args.num_q_heads, args.dim), (args.k_cache, args.v_cache))
    return o.view(args.bsz, args.q_len, args.num_q_heads, args.dim)
