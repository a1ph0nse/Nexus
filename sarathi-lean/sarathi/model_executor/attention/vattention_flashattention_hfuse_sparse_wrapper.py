from typing import List, Optional, Tuple

import torch

from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.vattention_flashattention_hfuse_wrapper import (
    VAttentionFlashAttentionHFUSEWrapper,
)
from sarathi.cache_ops import cache_flat

try:
    import pod_attn as fused
except Exception as e:
    print('unable to import module pod_attn')


class VAttentionFlashAttentionHFUSESparseWrapper(VAttentionFlashAttentionHFUSEWrapper):
    """
    vAttention HFUSE wrapper with Sliding Window Attention (SWA).
    Uses window_size=(4096, 0) to restrict attention to 4096 tokens to the left.
    """
    _inst = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        assert self.is_metadata_initialized, "Metadata is not initialized."

        if self.is_profiling_iteration:
            return torch.zeros_like(query)

        token_offset = 0
        output = torch.empty_like(query, device=self.device)
        idx = 0

        p_query, p_kcache, p_vcache = None, None, None
        p_query_len, p_cache_len = 0, 0
        if len(self.prefill_cache_lens) == 1:
            p_cache_len = self.current_total_len_device_lst[0]
            p_query_len = self.prefill_query_lens[0]
            cache_idx_p = self.batch_index[idx]
            p_query = query[: self.prefill_query_lens[0]].reshape(1, -1, self.num_q_heads, self.head_dim)
            p_k = key[: self.prefill_query_lens[0]].reshape(1, -1, self.num_kv_heads, self.head_dim)
            p_v = value[: self.prefill_query_lens[0]].reshape(1, -1, self.num_kv_heads, self.head_dim)
            p_kcache = kv_cache[0][cache_idx_p].reshape(1, -1, self.num_kv_heads, self.head_dim)
            p_vcache = kv_cache[1][cache_idx_p].reshape(1, -1, self.num_kv_heads, self.head_dim)
            token_offset = self.prefill_query_lens[0]
            with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
                cache_flat(p_k.squeeze(0),
                        p_v.squeeze(0),
                        p_kcache.squeeze(0),
                        p_vcache.squeeze(0),
                        "auto")
        elif len(self.prefill_cache_lens) > 1:
            raise ValueError("Multiple prefill cache lengths not supported")

        d_query, d_k, d_v = None, None, None
        if self.decode_batch_size != 0:
            with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
                d_query = query[
                    token_offset : token_offset + self.decode_batch_size
                ].reshape(-1, 1, self.num_q_heads, self.head_dim)
                d_k = key[token_offset : token_offset + self.decode_batch_size].reshape(
                    -1, 1, self.num_kv_heads, self.head_dim
                )
                d_v = value[
                    token_offset : token_offset + self.decode_batch_size
                ].reshape(-1, 1, self.num_kv_heads, self.head_dim)

        with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
            output_p, output_d = fused.true_fused_attn_with_kvcache(
                p_query,
                p_kcache,
                p_vcache,
                d_query,
                kv_cache[0],
                kv_cache[1],
                d_k,
                d_v,
                causal=True,
                cache_seqlens_p=p_cache_len,
                cache_seqlens_d=self.decode_cache_lens,
                cache_batch_idx=self.batch_index_gen,
                fused_params=self.fused_param,
                window_size=(4096, 0),
                )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            if p_query is not None:
                output[:p_query_len].copy_(
                    output_p.reshape(-1, self.num_q_heads * self.head_dim)
                )
            if d_query is not None:
                output[p_query_len : p_query_len + self.decode_batch_size].copy_(
                    output_d.reshape(-1, self.num_q_heads * self.head_dim)
                )

        return output
