import os
import re
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import pandas as pd

src = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(src))
logs = os.path.join(root, 'perf/op_util')
output_dir = os.path.join(root, 'perf', 'fig', 'op')
os.makedirs(output_dir, exist_ok=True)

record = {}

def get_substring(string, start, end):
    return string[string.find(start)+len(start):string.find(end)]

def prettify_model_name(model):
    return 'Yi-6B' if 'yi-6b' in model else \
            'Llama-2-7B' if 'llama-7b' in model else \
            'Llama-3-8B' if 'llama-3-8b' in model else model

def prettify_attn_name(attn):
    return 'POD_Attn' if 'pod_attn' in attn else \
            'HFuse' if 'hfuse' in attn else \
            'Libra' if 'load_attn' in attn else \
            'FI_Batch' if 'fi_bpf' in attn else attn

MODEL_INDEX_MAP = {
    "Llama-3-8B": 19,    # 模型对应下标
    "Llama-2-7B": 36,
    "Yi-6B": 7
}

def read_record(path):
    # ===================== 关键：放开模型限制，读取所有模型 =====================
    model = prettify_model_name(get_substring(path, 'op_util/op_util_model_', '_method_'))
    if model == "Yi-6B":
        return
    attn = prettify_attn_name(get_substring(path, '_method_', '.csv'))

    df = pd.read_csv(path, skiprows=3, thousands=',')
    df = df.dropna(subset=['ID'])
    df = df.iloc[1:].reset_index(drop=True)

    if model not in MODEL_INDEX_MAP:
        return  # 没有配置下标就跳过
    target_idx = MODEL_INDEX_MAP[model]

    compute = pd.to_numeric(df['sm__throughput.avg.pct_of_peak_sustained_elapsed'], errors='coerce').iloc[target_idx]
    dram = pd.to_numeric(df['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'], errors='coerce').iloc[target_idx]
    sm_max = pd.to_numeric(df['sm__cycles_active.max'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    sm_avg = pd.to_numeric(df['sm__cycles_active.avg'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    sm_min = pd.to_numeric(df['sm__cycles_active.min'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    sm_balance = (sm_max - sm_min) / sm_avg
    warp_max = pd.to_numeric(df['smsp__inst_executed.max'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    warp_avg = pd.to_numeric(df['smsp__inst_executed.avg'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    warp_min = pd.to_numeric(df['smsp__inst_executed.min'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    warp_balance = (warp_max - warp_min) / warp_avg
    stall_long = pd.to_numeric(df['smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]

    # 按 model + attn 存储
    key = (model, attn)
    if key not in record:
        record[key] = {
            'compute_throughput': [],
            'dram_throughput': [],
            'sm_balance': [],
            'stall_long': [],
            'warp_balance': [],
            'sm_max': [],
            'sm_min': [],
            'warp_max': [],
            'warp_min': []

        }
    record[key]['compute_throughput'].append(compute)
    record[key]['dram_throughput'].append(dram)
    record[key]['stall_long'].append(stall_long)
    record[key]['sm_balance'].append(sm_balance)
    record[key]['warp_balance'].append(warp_balance)
    record[key]['sm_balance'].append(sm_balance)
    record[key]['warp_balance'].append(warp_balance)
    record[key]['sm_max'].append(sm_max)
    record[key]['sm_min'].append(sm_min)
    record[key]['warp_max'].append(warp_max)
    record[key]['warp_min'].append(warp_min)

def read_logs():
    for root_dir, dirs, files in os.walk(logs):
        for file in files:
            if not file.endswith('.csv'): continue
            path = os.path.join(root_dir, file)
            read_record(path)

read_logs()

# ===================== 构建按模型整理的 DataFrame =====================
model_list = sorted(list(set(k[0] for k in record.keys())))
attn_list = ['POD_Attn', 'HFuse']

data_dict = {}
for attn in attn_list:
    data_dict[attn] = {}
    for model in model_list:
        val = record.get((model, attn), {}).get('sm_balance', [0])[0]
        data_dict[attn][model] = {
            'sm_balance': record.get((model, attn), {}).get('sm_balance', [0])[0],
            'stall_long': record.get((model, attn), {}).get('stall_long', [0])[0],
            'warp_balance': record.get((model, attn), {}).get('warp_balance', [0])[0]
        }

# ===================== 绘图风格 =====================
plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'Sans Serif'
colors = {'POD_Attn': '#549CD6', 'HFuse': '#F896A3'}


# ===================== 1行3列 + 横坐标模型 + SM/Warp 画 max+min（修复重叠） =====================
def plot_combined_by_model():
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3, wspace=0.15)  # 加大子图间距
    ax1, ax2, ax3 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2])
    axes = [ax1, ax2, ax3]

    metrics = [
        ('sm',      'SM Active Cycles',        ['sm_max', 'sm_min']),
        ('stall',   'Memory Access Stall Ratio', None),
        ('warp',    'Warp Instruction Executed',    ['warp_max', 'warp_min'])
    ]

    model_list = sorted(list(set(k[0] for k in record.keys())))
    width = 0.15
    x = np.arange(len(model_list)) * 0.7

    colors = {
        'POD_min': '#2E86AB',
        'POD_max': '#549CD6',
        'HFuse_min': '#C73E1D',
        'HFuse_max': '#F896A3',
    }

    for ax, (plot_type, ylab, max_min_keys) in zip(axes, metrics):
        if plot_type == 'stall':
            pod_vals = [record.get((m, 'POD_Attn'), {}).get('stall_long', [0])[0] for m in model_list]
            hf_vals  = [record.get((m, 'HFuse'),    {}).get('stall_long', [0])[0] for m in model_list]

            ax.bar(x - width/2, pod_vals, width, color='#549CD6', edgecolor='black', linewidth=0.8)
            ax.bar(x + width/2, hf_vals,  width, color='#F896A3', edgecolor='black', linewidth=0.8)

        else:
            key_max, key_min = max_min_keys
            pod_max = [record.get((m, 'POD_Attn'), {}).get(key_max, [0])[0] for m in model_list]
            pod_min = [record.get((m, 'POD_Attn'), {}).get(key_min, [0])[0] for m in model_list]
            hf_max  = [record.get((m, 'HFuse'),    {}).get(key_max, [0])[0] for m in model_list]
            hf_min  = [record.get((m, 'HFuse'),    {}).get(key_min, [0])[0] for m in model_list]

            # ===================== 核心：完全错开位置 =====================
            ax.bar(x - 1.5*width, pod_max, width, color=colors['POD_max'], edgecolor='black', linewidth=0.8)
            ax.bar(x - 0.5*width, pod_min, width, color=colors['POD_min'], edgecolor='black', linewidth=0.8)
            ax.bar(x + 0.5*width, hf_max,  width, color=colors['HFuse_max'], edgecolor='black', linewidth=0.8)
            ax.bar(x + 1.5*width, hf_min,  width, color=colors['HFuse_min'], edgecolor='black', linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(model_list, fontsize=16, rotation=0, ha='center')
        ax.set_ylabel(ylab, fontsize=20)
        # ax.set_xlabel('Model', fontsize=18)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

        # ===================== Warp 轴科学计数法变短 =====================
        # if plot_type == 'warp':
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    # 图例
    handles = [
        plt.Rectangle((0,0),1,1, color=colors['POD_max'], label='POD (Max)'),
        plt.Rectangle((0,0),1,1, color=colors['POD_min'], label='POD (Min)'),
        plt.Rectangle((0,0),1,1, color=colors['HFuse_max'], label='HFuse (Max)'),
        plt.Rectangle((0,0),1,1, color=colors['HFuse_min'], label='HFuse (Min)'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=18, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
    plt.savefig(os.path.join(output_dir, 'three_metrics_final.pdf'), bbox_inches='tight', dpi=300)
    plt.show()

plot_combined_by_model()