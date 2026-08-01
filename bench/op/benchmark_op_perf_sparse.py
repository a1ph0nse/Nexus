import sys
import torch
import utils

import os
import csv
import matplotlib.pyplot as plt

import json

model_configs = {
    #MHA
    'llama-7b-tp1': {'num_heads': 32, 'num_kv_heads': 32, 'head_size': 128, 'num_layers': 32},
    'llama-7b-tp2': {'num_heads': 16, 'num_kv_heads': 16, 'head_size': 128, 'num_layers': 32},
    'llama-7b-tp4': {'num_heads': 8, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
    # 'llama-7b-tp8': {'num_heads': 4, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    'llama-13b-tp1': {'num_heads': 40, 'num_kv_heads': 40, 'head_size': 128, 'num_layers': 40},
    'llama-13b-tp2': {'num_heads': 20, 'num_kv_heads': 20, 'head_size': 128, 'num_layers': 40},
    #GQA
    'llama-3-8b-tp1': {'num_heads': 32, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
    'llama-3-8b-tp2': {'num_heads': 16, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    'llama-3-8b-tp4': {'num_heads': 8, 'num_kv_heads': 2, 'head_size': 128, 'num_layers': 32},
    # 'llama-3-70B-tp1': {'num_heads': 64, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 80},
    # 'llama-3-70B-tp2': {'num_heads': 32, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 80},
    'yi-6b-tp1': {'num_heads': 32, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    # 'yi-6b-tp2': {'num_heads': 16, 'num_kv_heads': 2, 'head_size': 128, 'num_layers': 32},
    'yi-34b-tp1': {'num_heads': 56, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 60},
    'yi-34b-tp2': {'num_heads': 28, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 60},
    # MQA
    # 'falcon-7b-tp1': {'num_heads': 71, 'num_kv_heads': 1, 'head_size': 128, 'num_layers': 28},
    'PaLM-7b-tp1': {'num_heads': 64, 'num_kv_heads': 1, 'head_size': 128, 'num_layers': 28},
    'chatglm2-6b-tp1': {'num_heads': 32, 'num_kv_heads': 1, 'head_size': 128, 'num_layers': 28},
    'chatglm2-6b-tp2': {'num_heads': 16, 'num_kv_heads': 1, 'head_size': 128, 'num_layers': 28},
    'yi-6b-tp4': {'num_heads': 8, 'num_kv_heads': 1, 'head_size': 128, 'num_layers': 32},
    'llama-3-8b-tp8': {'num_heads': 4, 'num_kv_heads': 1, 'head_size': 128, 'num_layers': 32},
    # sparse
    'Mistral-7b-tp1': {'num_heads': 32, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
    'Mistral-7b-tp2': {'num_heads': 16, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    'Mistral-7b-tp4': {'num_heads': 8, 'num_kv_heads': 2, 'head_size': 128, 'num_layers': 32},
}

print(f"model;causal;block_size;" +
      f"fa_p;fa_d;fa_serial;" + 
      f"fi_p;fi_d;fi_serial;" + 
      f"fa_stream;fi_bpf;fa_hfuse;fa_pod;la;" +
      f"speedup(fa_serial/fi_serial); speedup(fa_serial/fa_stream);speedup(fa_serial/fi_bpf);speedup(fa_serial/fa_hfuse);speedup(fa_serial/fapod);speedup(fa_serial/la)")

block_size = 256

causal = True
disable_split = False
window_left = 4096
window_size = (4096, 0)
init_flag = False

def bench_attn(seqlen_configs, models):
    for model in models:
        torch.cuda.empty_cache()
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

            fa_p, t_fa_p = utils.bench_fa_prefill_paged(
                q_p, k_cache_p, v_cache_p,
                cache_seqlens=kv_lens_p,
                block_table = block_table_p_fa, causal=causal, window_size=window_size)

            fa_d, t_fa_d = utils.bench_fa_decode_paged(
                q_d, k_cache_d, v_cache_d,
                cache_seqlens=kv_lens_d,
                block_table = block_table_d_fa, window_size=window_size)

            stream_p, stream_d, t_fastream = utils.bench_fa_stream_paged(
                q_p, k_cache_p, v_cache_p,
                q_d, k_cache_d, v_cache_d,
                cache_seqlens_p=kv_lens_p, cache_seqlens_d=kv_lens_d,
                block_table_p = block_table_p_fa, block_table_d = block_table_d_fa, causal=causal,
                window_size=window_size
            )

            assert torch.allclose(fa_p, stream_p, atol=1e-3, rtol=1e-3)
            assert torch.allclose(fa_d, stream_d, atol=1e-3, rtol=1e-3)

            pod_p, pod_d, t_fapod = utils.bench_fa_pod_paged(
                q_p, k_cache_p, v_cache_p,
                q_d, k_cache_d, v_cache_d,
                fused_params=9, cache_seqlens_p=kv_lens_p, cache_seqlens_d=kv_lens_d,
                block_table_p = block_table_p_fa, block_table_d = block_table_d_fa, causal=causal,
                window_size=window_size
            )

            # NOTE: pod_attn has known numerical discrepancy under SWA (window_size != (-1,-1))
            # assert torch.allclose(fa_p, pod_p, atol=1e-3, rtol=1e-3)
            # assert torch.allclose(fa_d, pod_d, atol=1e-3, rtol=1e-3)

            # # fi batch_prefill calculate p d serial
            fi_p, t_fi_serial_p = utils.bench_flashinfer_batch_prefill_paged(
                q_p.reshape(-1, num_heads, head_size), kv_cache_p,
                qo_indptr_p, kv_indptr_p, block_table_p,
                last_page_len_p, num_head=num_heads, num_kv_head=num_kv_heads, head_dim=head_size, causal=causal, block_size=block_size,
                window_left=window_left
            )

            fi_d, t_fi_serial_d = utils.bench_flashinfer_batch_prefill_paged(
                q_d.reshape(-1, num_heads, head_size), kv_cache_d,
                qo_indptr_d, kv_indptr_d, block_table_d,
                last_page_len_d, num_head=num_heads, num_kv_head=num_kv_heads, head_dim=head_size, causal=causal, block_size=block_size,
                window_left=window_left
            )

            # NOTE: FI BatchPrefill prefill has known numerical discrepancy under SWA (window_left >= 0)
            # assert torch.allclose(fa_p.reshape(-1, num_heads, head_size), fi_p, atol=1e-3, rtol=1e-3)
            # NOTE: FI BatchPrefill decode has known numerical discrepancy under SWA (window_left >= 0)
            # assert torch.allclose(fa_d.reshape(-1, num_heads, head_size), fi_d, atol=1e-3, rtol=1e-3)

            # fi batch_prefill calculate p & d
            fi_pf_pd, t_fi_bpf = utils.bench_flashinfer_batch_prefill_paged(
                q_pd, kv_cache_pd,
                qo_indptr_pd, kv_indptr_pd, block_table_pd,
                last_page_len_pd, num_head=num_heads, num_kv_head=num_kv_heads, head_dim=head_size, causal=causal, block_size=block_size,
                window_left=window_left
            )

            # NOTE: FI BatchPrefill prefill has known numerical discrepancy under SWA (window_left >= 0)
            # assert torch.allclose(fa_p.reshape(-1, num_heads, head_size), fi_pf_pd[qo_indptr_d[-1].item():, :, :], atol=1e-3, rtol=1e-3)
            # NOTE: FI BatchPrefill decode has known numerical discrepancy under SWA (window_left >= 0)
            # assert torch.allclose(fa_d.reshape(-1, num_heads, head_size), fi_pf_pd[:qo_indptr_d[-1].item(), :, :], atol=1e-3, rtol=1e-3)

            # load_attn
            la_pd, t_la = utils.bench_flashinfer_load_attn_paged(
                q_pd, kv_cache_pd,
                qo_indptr_pd, kv_indptr_pd, block_table_pd,
                kv_lens_pd, num_head=num_heads, num_kv_head=num_kv_heads, head_dim=head_size, causal=causal, block_size=block_size, disable_split=disable_split,
                window_left=window_left
            )

            # NOTE: LoadAttention prefill has known numerical discrepancy under SWA (window_left >= 0)
            # assert torch.allclose(fa_p.reshape(-1, num_heads, head_size), la_pd[qo_indptr_d[-1].item():, :, :], atol=1e-3, rtol=1e-3)
            # NOTE: LoadAttention decode has known numerical discrepancy under SWA (window_left >= 0)
            # assert torch.allclose(fa_d.reshape(-1, num_heads, head_size), la_pd[:qo_indptr_d[-1].item(), :, :], atol=1e-3, rtol=1e-3)

            # hfuse (page table is wrong)
            bs_p = q_p.shape[0]
            bs_d = q_d.shape[0]
            max_cache_seqlen = kv_lens_p.max()
            max_cl = kv_lens_d.max()
            k_p = torch.randn(bs_p, max_cache_seqlen, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
            v_p = torch.randn(bs_p, max_cache_seqlen, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
            k_d = torch.randn(bs_d, max_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
            v_d = torch.randn(bs_d, max_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)

            fa_p, _ = utils.bench_fa_prefill_paged(
                q_p, k_p, v_p,
                cache_seqlens=kv_lens_p,
                causal=causal, profile=True, window_size=window_size)

            fa_d, _ = utils.bench_fa_decode_paged(
                q_d, k_d, v_d,
                cache_seqlens=kv_lens_d,
                profile=True, window_size=window_size)

            hfuse_p, hfuse_d, t_fahfuse = utils.bench_fa_pod(
                q_p, k_p, v_p,
                q_d, k_d, v_d,
                fused_params=64, seq_lens_k_p=kv_lens_p, seq_lens_k_d=kv_lens_d, causal=causal,
                window_size=window_size
            )

            # NOTE: pod_attn (hfuse) prefill has known numerical discrepancy under SWA (window_size != (-1,-1))
            # assert torch.allclose(fa_p, hfuse_p, atol=1e-3, rtol=1e-3)
            # NOTE: pod_attn decode path has known numerical discrepancy under SWA (window_size != (-1,-1))
            # assert torch.allclose(fa_d, hfuse_d, atol=1e-3, rtol=1e-3)


            t_fa_serial = round(t_fa_p + t_fa_d, 3)
            t_fi_serial = round(t_fi_serial_p + t_fi_serial_d, 3)

            speedup_fi_serial = round(t_fa_serial / t_fi_serial, 3)
            speedup_stream = round(t_fa_serial / t_fastream, 3)
            speedup_pod = round(t_fa_serial / t_fapod, 3)
            speedup_bpf = round(t_fa_serial / t_fi_bpf, 3)
            speedup_la = round(t_fa_serial / t_la, 3)
            speedup_hfuse = round(t_fa_serial / t_fahfuse, 3)
            
            print(f"{model};{causal};{block_size};" +
            f"{t_fa_p};{t_fa_d};{t_fa_serial};" +
            f"{t_fi_serial_p};{t_fi_serial_d};{t_fi_serial};" +
            f"{t_fastream};{t_fi_bpf};{t_fahfuse};{t_fapod};{t_la};" +
            f"{speedup_fi_serial};{speedup_stream};{speedup_bpf};{speedup_hfuse};{speedup_pod};{speedup_la};")

            dir_path = 'perf/op_perf'
            os.makedirs(dir_path, exist_ok=True)
            filename = f'./{dir_path}/op_perf_sparse.csv'
            need_first_line = True
            if os.path.exists(filename):
                need_first_line = False

            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                if need_first_line:        
                    writer.writerow(['model', 'seqlen_config', 'causal', 'block_size',
                                        'fa_p', 'fa_d', 'fa_serial',
                                        'fi_p', 'fi_d', 'fi_serial',
                                        'fa_stream', 'fi_bpf', 'fa_hfuse', 'fa_pod', 'la',
                                        'speedup(fa_serial/fi_serial)', 'speedup(fa_serial/fa_stream)', 'speedup(fa_serial/fi_bpf)', 'speedup(fa_serial/fa_hfuse)', 'speedup(fa_serial/fa_pod)', 'speedup(fa_serial/la)'])
                
                csv_data = [
                    model, json.dumps(seqlen_config), causal, block_size,
                    t_fa_p, t_fa_d, t_fa_serial,
                    t_fi_serial_p, t_fi_serial_d, t_fi_serial,
                    t_fastream, t_fi_bpf, t_fahfuse, t_fapod, t_la,
                    speedup_fi_serial, speedup_stream, speedup_bpf, speedup_hfuse, speedup_pod, speedup_la
                    ]
                writer.writerow(csv_data)

if __name__ == "__main__":
    seqlen_configs = []
    # v3 test case
    # fix and varlen
    # chunk 512
    # fix len
    # 72
    # 8 + 16
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
    # for i in range(4):
    #     seqlen_configs.append([(1024 * 4, 1)] * 16 + [(1024 * 1, 1)] * 38 + [(2048 * (i + 1), 2048)] * 1)

    # MHA: 84 * 2 = 168
    # GQA: 84 * 2 = 168
    # MQA: 84 * 2 = 168
    
    models = [
        # MHA
        # 'llama-7b-tp1',
        # 'llama-7b-tp2',
        # 'llama-7b-tp4',
        # 'llama-13b-tp1',
        # 'llama-13b-tp2',
        # GQA
        # 'llama-3-8b-tp1',
        # 'llama-3-8b-tp2',
        # 'yi-6b-tp1',
        # 'yi-34b-tp1',
        # 'yi-34b-tp2',
        # MQA
        # 'PaLM-7b-tp1',
        # 'chatglm2-6b-tp1',
        # 'chatglm2-6b-tp2',
        # 'yi-6b-tp4',
        # 'llama-3-8b-tp8',
        'Mistral-7b-tp1',
        'Mistral-7b-tp2',
        'Mistral-7b-tp4',
    ]

    bench_attn(seqlen_configs, models)

    # op_mapper = {
    #     "MHA" : [
    #         # 'llama-7b-tp1',
    #         # 'llama-7b-tp2',
    #         'llama-7b-tp4',
    #         # 'llama-13b-tp1',
    #         'llama-13b-tp2'
    #     ],
    #     "GQA" : [
    #         'llama-3-8b-tp1',
    #         'llama-3-8b-tp2',
    #         # 'yi-6b-tp1',
    #         # 'yi-34b-tp1',
    #         # 'yi-34b-tp2',
    #     ],
    #     "MQA" : [
    #         # 'PaLM-7b-tp1',
    #         # 'chatglm2-6b-tp1',
    #         # 'chatglm2-6b-tp2',
    #         'yi-6b-tp4',
    #         'llama-3-8b-tp8',
    #     ]
    # }

    # op_names = ["MHA", "GQA", "MQA"]

    # for op_name in op_names:
    #     bench_attn(seqlen_configs, op_mapper[op_name], op_name)
    
