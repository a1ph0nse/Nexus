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

def read_record(path):
    model = prettify_model_name(get_substring(path, 'op_util/op_util_model_', '_method_'))
    attn = prettify_attn_name(get_substring(path, '_method_', '.csv'))
    # if model != 'Llama-3-8B':
    #     return
    if model != 'Yi-6B':
        return
    # if model != 'Llama-2-7B':
    #     return
    df = pd.read_csv(path, skiprows=3, thousands=',')
    df = df.dropna(subset=['ID'])
    df = df.iloc[1:].reset_index(drop=True)

    compute = pd.to_numeric(df['sm__throughput.avg.pct_of_peak_sustained_elapsed'], errors='coerce')
    dram = pd.to_numeric(df['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'], errors='coerce')
    sm_max = pd.to_numeric(df['sm__cycles_active.max'].astype(str).str.replace(',', ''), errors='coerce')
    sm_avg = pd.to_numeric(df['sm__cycles_active.avg'].astype(str).str.replace(',', ''), errors='coerce')
    sm_min = pd.to_numeric(df['sm__cycles_active.min'].astype(str).str.replace(',', ''), errors='coerce')
    sm_balance = (sm_max - sm_min) / sm_avg
    # warp_max = pd.to_numeric(df['smsp__warps_eligible.max.per_cycle_active'].astype(str).str.replace(',', ''), errors='coerce')
    # warp_avg = pd.to_numeric(df['smsp__warps_eligible.avg.per_cycle_active'].astype(str).str.replace(',', ''), errors='coerce')
    # warp_min = pd.to_numeric(df['smsp__warps_eligible.min.per_cycle_active'].astype(str).str.replace(',', ''), errors='coerce')
    # warp_balance = (warp_max - warp_min) / warp_avg # obvious but correct?
    # warp_balance = warp_min # obvious but correct?
    # warp_balance = pd.to_numeric(df['smsp__average_warps_issue_stalled_wait_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce') # ok
    warp_max = pd.to_numeric(df['smsp__inst_executed.max'].astype(str).str.replace(',', ''), errors='coerce')
    warp_max = pd.to_numeric(df['smsp__inst_executed.max'].astype(str).str.replace(',', ''), errors='coerce')
    warp_avg = pd.to_numeric(df['smsp__inst_executed.avg'].astype(str).str.replace(',', ''), errors='coerce')
    warp_min = pd.to_numeric(df['smsp__inst_executed.min'].astype(str).str.replace(',', ''), errors='coerce')
    warp_balance = (warp_max - warp_min) / warp_avg

    stall_long = pd.to_numeric(df['smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce')

    if attn not in record:
        record[attn] = {
            'compute_throughput': [],
            'dram_throughput': [],
            'sm_balance' : [],
            'stall_long' : [],
            'warp_balance' : [],
        }
    record[attn]['compute_throughput'].append(compute)
    record[attn]['dram_throughput'].append(dram)
    record[attn]['stall_long'].append(stall_long)
    record[attn]['sm_balance'].append(sm_balance)
    record[attn]['warp_balance'].append(warp_balance)
        
def read_logs():
    for root, dirs, files in os.walk(logs):
        for file in files:
            if not file.endswith('.csv'):
                continue
            path = os.path.join(root, file)
            read_record(path)

methods = ["FI_Batch", 'POD_Attn', "HFuse", 'Libra']

read_logs()
for attn, metrics in record.items():
    for metric, values in metrics.items():
        if values:
            record[attn][metric] = pd.concat(values, ignore_index=True)

df = pd.DataFrame.from_dict(record).transpose()
df.fillna(0, inplace=True)
target_methods = ["POD_Attn", "HFuse"]
df = df.reindex(target_methods)
# print(record)
# print(df)

# methods = ["FI_Batch", 'POD_Attn', "HFuse", 'Libra']

plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Sans Serif'})

colors = {
    'POD_Attn': '#549CD6',
    'HFuse': '#F896A3',
}

def plot_combined_figure():
    # 品字形布局：2行 | 第一行1个图，第二行2个图
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    # 三个子图
    ax1 = fig.add_subplot(gs[0, :])  # 第一行：占满两列
    ax2 = fig.add_subplot(gs[1, 0])  # 第二行左
    ax3 = fig.add_subplot(gs[1, 1])  # 第二行右
    axes = [ax1, ax2, ax3]

    # 三个指标
    metrics = [
        ('sm_balance', 'SM Balance'),
        ('stall_long', 'Stall Longscoreboard'),
        ('warp_balance', 'Warp Balance')
    ]

    # 绘图参数
    start = 0
    n_cases = len(df.loc['HFuse', 'sm_balance'])
    # n_cases = 20
    
    start = 16
    n_cases = 4
    width = 0.35
    x = np.arange(n_cases)
    case_labels = [f'Case{i+1}' for i in range(n_cases)]

    for ax, (metric_name, y_label) in zip(axes, metrics):
        pod = df.loc['POD_Attn', metric_name].values[start:start+n_cases]
        hfuse = df.loc['HFuse', metric_name].values[start:start+n_cases]

        # 画分组柱状图
        bar1 = ax.bar(x - width/2, pod, width, label='POD_Attn', color=colors['POD_Attn'], alpha=0.7, edgecolor='black')
        bar2 = ax.bar(x + width/2, hfuse, width, label='HFuse', color=colors['HFuse'], alpha=0.7, edgecolor='black')

        # 数值标签
        for xs, ys in zip([x-width/2, x+width/2], [pod, hfuse]):
            for xx, yy in zip(xs, ys):
                ax.annotate(f'{yy:.2f}', xy=(xx, yy), xytext=(0,3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=14)

        # 样式
        ax.set_xticks(x)
        ax.set_xticklabels(case_labels, fontsize=20)
        ax.set_ylabel(y_label, fontsize=20)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # ===================== 图例放在 最 上 方 居 中 =====================
    fig.legend(handles=[bar1, bar2], loc='upper center', bbox_to_anchor=(0.5, 1.04), 
               ncol=2, fontsize=26, frameon=False)

    plt.subplots_adjust(top=0.9)  # 给顶部图例留出空间
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'three_metrics_combined.pdf'), bbox_inches='tight', dpi=300)
    plt.show()

# 执行绘图
plot_combined_figure()

def plot_simple_bar(metric_name, y_label, y_func, fig_size=(9,6)):
    # 提取 POD 和 HFuse 的数据
    pod_values = df.loc['POD_Attn', metric_name].values
    hfuse_values = df.loc['HFuse', metric_name].values

    # 自动匹配长度
    start = 0
    # n_cases = 4
    n_cases = len(df.loc['HFuse', metric_name])

    pod_values = pod_values[start:start+n_cases]
    hfuse_values = hfuse_values[start:start+n_cases]

    x = np.arange(n_cases)  # case 编号
    width = 0.35            # 柱子宽度

    # 保持你原来的图大小：9,6 不变
    fig, ax = plt.subplots(figsize=(9, 6))

    # 画分组柱状图：每个 case 两根柱子
    bars_pod = ax.bar(x - width/2, pod_values, width, 
                      label='POD_Attn', color=colors['POD_Attn'], alpha=0.7, edgecolor='black')
    bars_hfuse = ax.bar(x + width/2, hfuse_values, width, 
                        label='HFuse', color=colors['HFuse'], alpha=0.7, edgecolor='black')

    # 柱子顶部显示数值
    for bars in [bars_pod, bars_hfuse]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=16)

    # 样式
    ax.set_xticks(x)
    ax.set_xticklabels([f'Case{i+1}' for i in range(n_cases)], fontsize=18)
    # ax.set_xlabel('Case', fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(top=max(max(pod_values), max(hfuse_values)) * 1.15)  # 多留出15%空间

    # 统一图例：每张图顶部只显示一次
    if y_label == 'SM Balance':
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=22, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric_name}_bar.pdf'), bbox_inches='tight', dpi=300)
    plt.show()

# ===================== 调用绘图（分开画3张，保持原图大小） =====================
fs = (12, 6)
plot_simple_bar(
    metric_name='sm_balance',
    y_label='SM Balance',
    y_func=lambda vals: ['{:.1f}'.format(x) for x in vals],
    fig_size=fs
)

plot_simple_bar(
    metric_name='stall_long',
    y_label='Stall Longscoreboard',
    y_func=lambda vals: ['{:.1f}'.format(x) for x in vals],
)

plot_simple_bar(
    metric_name='warp_balance',
    y_label='Warp Balance',
    y_func=lambda vals: ['{:.1f}'.format(x) for x in vals],
)