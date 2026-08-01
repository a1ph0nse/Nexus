import sys
import torch
import utils

import os
import csv
import matplotlib.pyplot as plt

import json
import argparse

model_configs = {
    #MHA
    'llama-7b-tp1': {'num_heads': 32, 'num_kv_heads': 32, 'head_size': 128, 'num_layers': 32},
    'llama-7b-tp2': {'num_heads': 16, 'num_kv_heads': 16, 'head_size': 128, 'num_layers': 32},
    'llama-7b-tp4': {'num_heads': 8, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
    'llama-7b-tp8': {'num_heads': 4, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    'llama-13b-tp1': {'num_heads': 40, 'num_kv_heads': 40, 'head_size': 128, 'num_layers': 32},
    'llama-13b-tp2': {'num_heads': 20, 'num_kv_heads': 20, 'head_size': 128, 'num_layers': 32},
    #GQA
    'llama-3-8b-tp1': {'num_heads': 32, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
    'llama-3-8b-tp2': {'num_heads': 16, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    'llama-3-8b-tp4': {'num_heads': 8, 'num_kv_heads': 2, 'head_size': 128, 'num_layers': 32},
    'llama-3-8b-tp8': {'num_heads': 4, 'num_kv_heads': 1, 'head_size': 128, 'num_layers': 32},
    'llama-3-70B-tp1': {'num_heads': 64, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
    'llama-3-70B-tp2': {'num_heads': 32, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    'yi-6b-tp1': {'num_heads': 32, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    'yi-6b-tp2': {'num_heads': 16, 'num_kv_heads': 2, 'head_size': 128, 'num_layers': 32},
    'yi-6b-tp4': {'num_heads': 8, 'num_kv_heads': 1, 'head_size': 128, 'num_layers': 32},
    'yi-34b-tp1': {'num_heads': 56, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 60},
    'yi-34b-tp2': {'num_heads': 28, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 60},
    'yi-34b-tp4': {'num_heads': 14, 'num_kv_heads': 2, 'head_size': 128, 'num_layers': 60},
    'progen2-tp1': {'num_heads': 32, 'num_kv_heads': 32, 'head_size': 128, 'num_layers': 32},
    # 'llama-7b-tp2': {'num_heads': 16, 'num_kv_heads': 16, 'head_size': 128, 'num_layers': 32},
}

block_size = 256
is_profile = True
causal = True
disable_split = False
init_flag = False

def bench_attn(seqlen_configs, models, methods):
    for model in models:
        print(f"Profiling Model: {model}...")
        for seqlen_config in seqlen_configs:
            num_heads = model_configs[model]['num_heads']
            num_kv_heads = model_configs[model]['num_kv_heads']
            head_size = model_configs[model]['head_size']

            (
                q_p, k_cache_p, v_cache_p, qo_indptr_p, kv_indptr_p, block_table_p, qo_lens_p, kv_lens_p, last_page_len_p,
                q_d, k_cache_d, v_cache_d, qo_indptr_d, kv_indptr_d, block_table_d, qo_lens_d, kv_lens_d, last_page_len_d
            ) = utils.generate_page_input(seqlen_config, num_heads, num_kv_heads, head_size, block_size)

            block_table_p_fa = utils.modify_block_table(kv_indptr_p, block_table_p)
            block_table_d_fa = utils.modify_block_table(kv_indptr_d, block_table_d)

            kv_cache_p = utils.merge_kv_cache(k_cache_p, v_cache_p)
            kv_cache_d = utils.merge_kv_cache(k_cache_d, v_cache_d)

            (
                q_pd, kv_cache_pd,
                qo_indptr_pd, kv_indptr_pd,
                block_table_pd, kv_lens_pd, last_page_len_pd
            ) = utils.merge_pd(
                q_p, kv_cache_p,
                q_d, kv_cache_d,
                qo_indptr_p, qo_indptr_d,
                kv_indptr_p, kv_indptr_d,
                kv_lens_p, kv_lens_d,
                last_page_len_p, last_page_len_d
                )

            if "fa_serial" in methods:
                fa_p, t_fa_p = utils.bench_fa_prefill_paged(
                    q_p, k_cache_p, v_cache_p,
                    cache_seqlens=kv_lens_p,
                    block_table = block_table_p_fa, causal=causal,
                    profile=is_profile
                )

                fa_d, t_fa_d = utils.bench_fa_decode_paged(
                    q_d, k_cache_d, v_cache_d,
                    cache_seqlens=kv_lens_d,
                    block_table = block_table_d_fa,
                    profile=is_profile
                )

            if "fa_stream" in methods:
                stream_p, stream_d, t_fastream = utils.bench_fa_stream_paged(
                    q_p, k_cache_p, v_cache_p,
                    q_d, k_cache_d, v_cache_d,
                    fused_params=9, cache_seqlens_p=kv_lens_p, cache_seqlens_d=kv_lens_d,
                    block_table_p = block_table_p_fa, block_table_d = block_table_d_fa, causal=causal,
                    profile=is_profile
                )

            if "pod_attn" in methods:
                pod_p, pod_d, t_fapod = utils.bench_fa_pod_paged(
                    q_p, k_cache_p, v_cache_p,
                    q_d, k_cache_d, v_cache_d,
                    fused_params=9, cache_seqlens_p=kv_lens_p, cache_seqlens_d=kv_lens_d,
                    block_table_p = block_table_p_fa, block_table_d = block_table_d_fa, causal=causal,
                    profile=is_profile
                )

            if "fi_serial" in methods:
                # fi batch_prefill calculate p d serial
                fi_p, t_fi_serial_p = utils.bench_flashinfer_batch_prefill_paged(
                    q_p.reshape(-1, num_heads, head_size), kv_cache_p,
                    qo_indptr_p, kv_indptr_p, block_table_p,
                    last_page_len_p, num_head=num_heads, num_kv_head=num_kv_heads, head_dim=head_size, causal=causal, block_size=block_size,
                    profile=is_profile
                )

                fi_d, t_fi_serial_d = utils.bench_flashinfer_batch_prefill_paged(
                    q_d.reshape(-1, num_heads, head_size), kv_cache_d,
                    qo_indptr_d, kv_indptr_d, block_table_d,
                    last_page_len_d, num_head=num_heads, num_kv_head=num_kv_heads, head_dim=head_size, causal=causal, block_size=block_size,
                    profile=is_profile
                )

            if "fi_bpf" in methods:
                # fi batch_prefill calculate p & d
                fi_pf_pd, t_fi_bpf = utils.bench_flashinfer_batch_prefill_paged(
                    q_pd, kv_cache_pd,
                    qo_indptr_pd, kv_indptr_pd, block_table_pd,
                    last_page_len_pd, num_head=num_heads, num_kv_head=num_kv_heads, head_dim=head_size, causal=causal, block_size=block_size,
                    profile=is_profile
                )

            if "load_attn" in methods:
                # load_attn
                la_pd, t_la = utils.bench_flashinfer_load_attn_paged(
                    q_pd, kv_cache_pd,
                    qo_indptr_pd, kv_indptr_pd, block_table_pd,
                    kv_lens_pd, num_head=num_heads, num_kv_head=num_kv_heads, head_dim=head_size, causal=causal, block_size=block_size, disable_split=disable_split,
                    profile=is_profile
                )

            if "hfuse" in methods:
                # hfuse (page table is wrong)
                bs_p = q_p.shape[0]
                bs_d = q_d.shape[0]
                max_cache_seqlen = kv_lens_p.max()
                max_cl = kv_lens_d.max()
                k_p = torch.randn(bs_p, max_cache_seqlen, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
                v_p = torch.randn(bs_p, max_cache_seqlen, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
                k_d = torch.randn(bs_d, max_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
                v_d = torch.randn(bs_d, max_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)

                hfuse_p, hfuse_d, t_fahfuse = utils.bench_fa_pod(
                    q_p, k_p, v_p,
                    q_d, k_d, v_d,
                    fused_params=64, seq_lens_k_p=kv_lens_p, seq_lens_k_d=kv_lens_d, causal=causal, profile=is_profile
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='profile kernel utilization')
    parser.add_argument('method', type=str, default='fa_pod',
                        help='profile method name')
    parser.add_argument('model', type=str, default='llama-7b-tp1',
                        help='profile model name')

    args = parser.parse_args()

    seqlen_configs = []
    # chunk 512
    # fix len
    for i in range(2):
        seqlen_configs.append([(1024 * 4, 1)] * 32 + [(512 * (i + 1), 512)] * 2)
    for i in range(2):
        seqlen_configs.append([(1024 * 8, 1)] * 16 + [(512 * (i + 1), 512)] * 2)
    # var len
    for i in range(4):
        seqlen_configs.append([(1024 * 24, 1)] * 1 + [(1024 * 1, 1)] * 53 + [(512 * (i + 1), 512)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 20, 1)] * 1 + [(1024 * 1, 1)] * 53 + [(512 * (i + 1), 512)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 16, 1)] * 1 + [(1024 * 1, 1)] * 53 + [(512 * (i + 1), 512)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 12, 1)] * 2 + [(1024 * 1, 1)] * 52 + [(512 * (i + 1), 512)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 10, 1)] * 4 + [(1024 * 1, 1)] * 50 + [(512 * (i + 1), 512)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 8, 1)] * 8 + [(1024 * 1, 1)] * 46 + [(512 * (i + 1), 512)] * 1)
    
    # 8 + 16
    # chunk 1024
    # fix len
    for i in range(1):
        seqlen_configs.append([(1024 * 4, 1)] * 32 + [(1024 * (i + 1), 1024)] * 1)
    for i in range(2):
        seqlen_configs.append([(1024 * 8, 1)] * 16 + [(1024 * (i + 1), 1024)] * 1)
    for i in range(1):
        seqlen_configs.append([(1024 * 16, 1)] * 8 + [(1024 * (i + 1), 1024)] * 1)
    # var len
    for i in range(4):
        seqlen_configs.append([(1024 * 24, 1)] * 1 + [(1024 * 1, 1)] * 53 + [(1024 * (i + 1), 1024)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 20, 1)] * 1 + [(1024 * 1, 1)] * 53 + [(1024 * (i + 1), 1024)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 16, 1)] * 1 + [(1024 * 1, 1)] * 53 + [(1024 * (i + 1), 1024)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 12, 1)] * 2 + [(1024 * 1, 1)] * 52 + [(1024 * (i + 1), 1024)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 10, 1)] * 4 + [(1024 * 1, 1)] * 50 + [(1024 * (i + 1), 1024)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 8, 1)] * 8 + [(1024 * 1, 1)] * 46 + [(1024 * (i + 1), 1024)] * 1)
    # for i in range(4):
    #     seqlen_configs.append([(1024 * 4, 1)] * 16 + [(1024 * 1, 1)] * 38 + [(1024 * (i + 1), 1024)] * 1)
    
    # 8 + 16
    # chunk 2048
    # fix len
    for i in range(1):
        seqlen_configs.append([(1024 * 4, 1)] * 32 + [(2048 * (i + 1), 2048)] * 1)
    for i in range(1):
        seqlen_configs.append([(1024 * 8, 1)] * 16 + [(2048 * (i + 1), 2048)] * 1)
    for i in range(2):
        seqlen_configs.append([(1024 * 16, 1)] * 8 + [(2048 * (i + 1), 2048)] * 1)
    # var len
    for i in range(4):
        seqlen_configs.append([(1024 * 24, 1)] * 1 + [(1024 * 1, 1)] * 53 + [(2048 * (i + 1), 2048)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 20, 1)] * 1 + [(1024 * 1, 1)] * 53 + [(2048 * (i + 1), 2048)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 16, 1)] * 1 + [(1024 * 1, 1)] * 53 + [(2048 * (i + 1), 2048)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 12, 1)] * 2 + [(1024 * 1, 1)] * 52 + [(2048 * (i + 1), 2048)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 10, 1)] * 4 + [(1024 * 1, 1)] * 50 + [(2048 * (i + 1), 2048)] * 1)
    for i in range(4):
        seqlen_configs.append([(1024 * 8, 1)] * 8 + [(1024 * 1, 1)] * 46 + [(2048 * (i + 1), 2048)] * 1)
    # models = [
    #     'llama-7b-tp1',
        # 'llama-7b-tp2',
        # 'llama-7b-tp4',
        # 'llama-7b-tp8',
        # 'llama-13b-tp1',
        # 'llama-13b-tp2',
        # 'llama-3-8b-tp1',
        # 'llama-3-8b-tp2',
        # 'llama-3-8b-tp4',
        # 'llama-3-8b-tp8',
        # 'llama-3-70B-tp1',
        # 'llama-3-70B-tp2',
        # 'yi-6b-tp1',
        # 'yi-6b-tp2',
        # 'yi-6b-tp4',
    # ]

    # methods = [
    #     "fa_pod",
    #     "la"
    # ]

    methods = [args.method]
    models = [args.model]

    bench_attn(seqlen_configs, models, methods)
    
