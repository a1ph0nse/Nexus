from typing import List

import torch
from flashinfer import get_batch_indices_positions, get_seq_lens

from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.model_executor.attention.load_attention_wrapper import LoadAttentionWrapper


class LoadAttentionSparseWrapper(LoadAttentionWrapper):
    """
    LoadAttention wrapper with Sliding Window Attention (SWA).
    Uses window_left=4096 to restrict attention to 4096 tokens to the left.
    """
    _inst = None

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        # Reuse the parent's metadata computation logic, then override
        # the _wrapper.begin_forward() call to add window_left=4096.

        qo_indptr: List[int] = [0]
        kv_page_indices: List[int] = []
        kv_last_page_len: List[int] = []
        kv_page_indptr: List[int] = [0]
        kv_len_arr: List[int] = []

        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()

            current_total_len = processed_prompt_len + prompt_chunk_len

            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            qo_indptr.append(qo_indptr[-1] + prompt_chunk_len)
            num_blocks_in_use = (
                current_total_len + self.block_size - 1
            ) // self.block_size
            kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])
            kv_page_indptr.append(kv_page_indptr[-1] + num_blocks_in_use)
            kv_last_page_len.append(
                (current_total_len - 1) % self.block_size + 1
            )
            kv_len_arr.append(prompt_chunk_len)

        self.total_prefill_len = sum(kv_len_arr)

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            context_len = seq_metadata.seq.get_len()
            qo_indptr.append(qo_indptr[-1] + 1)
            kv_page_indices.extend(seq_metadata.block_table)
            kv_page_indptr.append(kv_page_indptr[-1] + len(seq_metadata.block_table))
            kv_last_page_len.append((context_len - 1) % self.block_size + 1)
            kv_len_arr.append(context_len)

        # Convert to tensors.
        self.qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32, device=self.device)
        self.kv_page_indices = torch.tensor(
            kv_page_indices, dtype=torch.int32, device=self.device
        )
        self.kv_page_indptr = torch.tensor(
            kv_page_indptr, dtype=torch.int32, device=self.device
        )
        self.kv_last_page_len = torch.tensor(
            kv_last_page_len, dtype=torch.int32, device=self.device
        )
        self.kv_len_arr_tensor = torch.tensor(
            kv_len_arr, dtype=torch.int32, device=self.device
        )

        self.batch_indices, self.positions = get_batch_indices_positions(
            self.qo_indptr,
            get_seq_lens(self.kv_page_indptr, self.kv_last_page_len, self.block_size),
            self.qo_indptr[-1].item()
        )
        self.total_seqlen = self.qo_indptr.shape[0] - 1
        self.append_seqlen = self.qo_indptr[-1].item()

        self._wrapper.begin_forward(
            self.qo_indptr,
            self.kv_page_indptr,
            self.kv_page_indices,
            self.kv_len_arr_tensor,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            self.head_dim,
            self.block_size,
            window_left=4096,
        )
