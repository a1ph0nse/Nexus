from typing import Optional, Tuple

import torch

from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.flashattention_pod_wrapper import FlashAttentionPODWrapper
from sarathi.cache_ops import reshape_and_cache_flash

try:
    import pod_attn as fused
except Exception as e:
    print('unable to import module pod_attn')


class FlashAttentionPODSparseWrapper(FlashAttentionPODWrapper):
    """
    FlashAttention POD wrapper with Sliding Window Attention (SWA).
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

        prefill_query = None
        prefill_cache_len, query_len = 0, 0
        prefill_block_table = None
        if len(self.prefill_cache_lens) == 1:
            prefill_cache_len = self.prefill_cache_lens[0]
            prefill_block_table = self.prefill_block_tables[0]
            query_len = self.prefill_query_lens[0]

            with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
                prefill_query = query[token_offset : token_offset + query_len].reshape(
                    1, -1, self.num_q_heads, self.head_dim
                )
                prefill_key = key[token_offset : token_offset + query_len].reshape(
                    1, -1, self.num_kv_heads, self.head_dim
                )
                prefill_value = value[token_offset : token_offset + query_len].reshape(
                    1, -1, self.num_kv_heads, self.head_dim
                )

            with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
                slot_mapping = self.prefix_plus_current_prompt_tokens_slot_mapping[token_offset: token_offset + query_len]
                assert slot_mapping is not None
                reshape_and_cache_flash(prefill_key.squeeze(0),
                                        prefill_value.squeeze(0),
                                        kv_cache[0],
                                        kv_cache[1],
                                        slot_mapping,
                                        "auto",
                                        )
        elif len(self.prefill_cache_lens) > 1:
            raise ValueError("Multiple prefill cache lengths not supported")

        token_offset += query_len

        decode_batch_size = 0 if self.decode_cache_len is None else self.decode_cache_len.size(0)
        decode_query, decode_key, decode_value = None, None, None
        if decode_batch_size != 0:
            with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
                decode_query = query[
                    token_offset : token_offset + decode_batch_size
                ].reshape(-1, 1, self.num_q_heads, self.head_dim)
                decode_key = key[token_offset : token_offset + decode_batch_size].reshape(
                    -1, 1, self.num_kv_heads, self.head_dim
                )
                decode_value = value[
                    token_offset : token_offset + decode_batch_size
                ].reshape(-1, 1, self.num_kv_heads, self.head_dim)

        with self.get_timer(OperationMetrics.ATTN, layer_id):
            prefill_output, decode_output = fused.true_fused_attn_with_kvcache(
                prefill_query,
                kv_cache[0],
                kv_cache[1],
                decode_query,
                kv_cache[0],
                kv_cache[1],
                decode_key,
                decode_value,
                causal=True,
                cache_seqlens_p=prefill_cache_len+query_len,
                cache_seqlens_d=self.decode_cache_len,
                block_table_p=prefill_block_table,
                block_table_d=self.decode_block_table,
                fused_params=self.fused_param,
                window_size=(4096, 0),
                )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            if prefill_query is not None:
                output[: query_len].copy_(
                    prefill_output.reshape(-1, self.num_q_heads * self.head_dim)
                )
            if decode_query is not None:
                output[query_len : query_len + decode_batch_size].copy_(
                    decode_output.reshape(-1, self.num_q_heads * self.head_dim)
                )

        return output
