import sys
import torch
import utils

import os
import csv
import json

model_configs = {
    #MHA
    'llama-7b-tp1': {'num_heads': 32, 'num_kv_heads': 32, 'head_size': 128, 'num_layers': 32},
    'llama-7b-tp2': {'num_heads': 16, 'num_kv_heads': 16, 'head_size': 128, 'num_layers': 32},
    'llama-7b-tp4': {'num_heads': 8, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
    'llama-7b-tp8': {'num_heads': 4, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    'llama-13b-tp1': {'num_heads': 40, 'num_kv_heads': 40, 'head_size': 128, 'num_layers': 40},
    'llama-13b-tp2': {'num_heads': 20, 'num_kv_heads': 20, 'head_size': 128, 'num_layers': 40},
    #GQA
    'llama-3-8b-tp1': {'num_heads': 32, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
    'llama-3-8b-tp2': {'num_heads': 16, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    'llama-3-8b-tp4': {'num_heads': 8, 'num_kv_heads': 2, 'head_size': 128, 'num_layers': 32},
    'llama-3-8b-tp8': {'num_heads': 4, 'num_kv_heads': 1, 'head_size': 128, 'num_layers': 32},
    'llama-3-70B-tp1': {'num_heads': 64, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 80},
    'llama-3-70B-tp2': {'num_heads': 32, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 80},
    'yi-6b-tp1': {'num_heads': 32, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    'yi-6b-tp2': {'num_heads': 16, 'num_kv_heads': 2, 'head_size': 128, 'num_layers': 32},
    'yi-6b-tp4': {'num_heads': 8, 'num_kv_heads': 1, 'head_size': 128, 'num_layers': 32},
    'yi-34b-tp1': {'num_heads': 56, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 60},
    'yi-34b-tp2': {'num_heads': 28, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 60},#
    'yi-34b-tp4': {'num_heads': 14, 'num_kv_heads': 2, 'head_size': 128, 'num_layers': 60},#
    # sparse
    'mistral-7b-tp1': {'num_heads': 32, 'num_kv_heads': 8, 'head_size': 128, 'num_layers': 32},
    'mistral-7b-tp2': {'num_heads': 16, 'num_kv_heads': 4, 'head_size': 128, 'num_layers': 32},
    'mistral-7b-tp4': {'num_heads': 8, 'num_kv_heads': 2, 'head_size': 128, 'num_layers': 32},
    'progen2-tp1': {'num_heads': 32, 'num_kv_heads': 32, 'head_size': 128, 'num_layers': 32},
}

print(f"model;causal;block_size;" +
      f"sm_cost_min;sm_cost_max;sm_cost_avg;sm_max_min_diff;sm_lb_metric;" +
      f"cta_cost_min;cta_cost_max;cta_cost_avg;cta_max_min_diff;cta_lb_metric")

block_size = 256
causal = True
disable_split = False
window_left = -1

CSV_HEADER = [
    'model', 'seqlen_config', 'causal', 'block_size',
    'sm_cost_min', 'sm_cost_max', 'sm_cost_avg', 'sm_max_min_diff', 'sm_lb_metric',
    'cta_cost_min', 'cta_cost_max', 'cta_cost_avg', 'cta_max_min_diff', 'cta_lb_metric',
]

def bench_attn(seqlen_configs, models):
    for model in models:
        for seqlen_config in seqlen_configs:
            num_heads = model_configs[model]['num_heads']
            num_kv_heads = model_configs[model]['num_kv_heads']
            head_size = model_configs[model]['head_size']

            # 生成 page 输入 (保留原有逻辑)
            (
                q_p, k_cache_p, v_cache_p, qo_indptr_p, kv_indptr_p, block_table_p, qo_lens_p, kv_lens_p, last_page_len_p,
                q_d, k_cache_d, v_cache_d, qo_indptr_d, kv_indptr_d, block_table_d, qo_lens_d, kv_lens_d, last_page_len_d
            ) = utils.generate_page_input(seqlen_config, num_heads, num_kv_heads, head_size, block_size)

            # 合并 prefill + decode (保留原有逻辑)
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

            # 仅调用 LoadAttention.plan() 获取调度 cost 统计
            sm_min, sm_max, sm_avg, cta_min, cta_max, cta_avg = \
                utils.bench_load_attention_cost(
                    q_pd, kv_cache_pd,
                    qo_indptr_pd, kv_indptr_pd, block_table_pd,
                    kv_lens_pd,
                    num_head=num_heads, num_kv_head=num_kv_heads,
                    head_dim=head_size, causal=causal, block_size=block_size,
                    disable_split=disable_split)

            # 派生指标
            sm_diff = sm_max - sm_min
            sm_lb = sm_avg / sm_diff if sm_diff > 0 else -1.0
            cta_diff = cta_max - cta_min
            cta_lb = cta_avg / cta_diff if cta_diff > 0 else -1.0

            # 终端输出
            print(f"{model};{causal};{block_size};" +
                  f"{sm_min:.2f};{sm_max:.2f};{sm_avg:.2f};{sm_diff:.2f};{sm_lb:.3f};" +
                  f"{cta_min:.2f};{cta_max:.2f};{cta_avg:.2f};{cta_diff:.2f};{cta_lb:.3f}")

            # CSV 写入
            dir_path = 'perf/op_perf'
            os.makedirs(dir_path, exist_ok=True)
            filename = f'./{dir_path}/op_est_cost.csv'
            need_first_line = not os.path.exists(filename)

            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if need_first_line:
                    writer.writerow(CSV_HEADER)
                writer.writerow([
                    model, json.dumps(seqlen_config), causal, block_size,
                    f"{sm_min:.2f}", f"{sm_max:.2f}", f"{sm_avg:.2f}", f"{sm_diff:.2f}", f"{sm_lb:.3f}",
                    f"{cta_min:.2f}", f"{cta_max:.2f}", f"{cta_avg:.2f}", f"{cta_diff:.2f}", f"{cta_lb:.3f}",
                ])

if __name__ == "__main__":
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

    # MHA: 72 * 5 = 360
    # GQA: 72 * 8 = 576
    models = [
        # 'yi-6b-tp1',
        # 'yi-6b-tp2',
        # 'yi-6b-tp4',
        # 'llama-7b-tp1',
        # 'llama-7b-tp2',
        # 'llama-7b-tp4',
        # 'llama-3-8b-tp1',
        # 'llama-3-8b-tp2',
        # 'llama-3-8b-tp8',
        # 'llama-13b-tp1',
        # 'llama-13b-tp2',
        # 'llama-3-70B-tp1',
        # 'llama-3-70B-tp2',
        # 'yi-34b-tp1',
        # 'yi-34b-tp2',
        # 'yi-34b-tp4',
        # 'mistral-7b-tp1',
        # 'mistral-7b-tp2',
        # 'mistral-7b-tp4',
        'progen2-tp1'
    ]
    bench_attn(seqlen_configs, models)
    
