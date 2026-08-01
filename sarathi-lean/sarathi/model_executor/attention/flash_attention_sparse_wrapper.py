from typing import List, Optional, Tuple

import torch
from flash_attn import flash_attn_with_kvcache

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.logger import init_logger
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.flash_attention_wrapper import FlashAttentionWrapper
from sarathi.cache_ops import reshape_and_cache_flash

logger = init_logger(__name__)


class FlashAttentionSparseWrapper(FlashAttentionWrapper):
    """
    FlashAttention wrapper with Sliding Window Attention (SWA).
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

        # first process the prefill attention
        for prefill_cache_len, prefill_block_table, query_len in zip(
            self.prefill_cache_lens, self.prefill_block_tables, self.prefill_query_lens
        ):
            with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
                seq_query = query[token_offset : token_offset + query_len].reshape(
                    1, -1, self.num_q_heads, self.head_dim
                )
                seq_key = key[token_offset : token_offset + query_len].reshape(
                    1, -1, self.num_kv_heads, self.head_dim
                )
                seq_value = value[token_offset : token_offset + query_len].reshape(
                    1, -1, self.num_kv_heads, self.head_dim
                )

            with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
                slot_mapping = self.prefix_plus_current_prompt_tokens_slot_mapping[token_offset: token_offset + query_len]
                assert slot_mapping is not None
                reshape_and_cache_flash(seq_key.squeeze(0),
                                        seq_value.squeeze(0),
                                        kv_cache[0],
                                        kv_cache[1],
                                        slot_mapping,
                                        "auto",
                                        )

            with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
                seq_output = flash_attn_with_kvcache(
                    seq_query,
                    kv_cache[0],
                    kv_cache[1],
                    cache_seqlens=prefill_cache_len+query_len,
                    block_table=prefill_block_table,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=(4096, 0),
                )

            with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
                output[token_offset : token_offset + query_len].copy_(
                    seq_output.reshape(-1, self.num_q_heads * self.head_dim)
                )

            token_offset += query_len

        if self.decode_cache_len is None:
            return output

        decode_batch_size = self.decode_cache_len.size(0)

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

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            slot_mapping = self.current_tokens_slot_mapping[token_offset: token_offset + decode_batch_size]

        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):
            decode_output = flash_attn_with_kvcache(
                decode_query,
                kv_cache[0],
                kv_cache[1],
                decode_key,
                decode_value,
                cache_seqlens=self.decode_cache_len,
                block_table=self.decode_block_table,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=(4096, 0),
            )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output[token_offset : token_offset + decode_batch_size].copy_(
                decode_output.reshape(-1, self.num_q_heads * self.head_dim)
            )

        return output
