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
            'Llama-13B' if 'llama-13b' in model else \
            'Llama-3-8B' if 'llama-3-8b' in model else model

def prettify_attn_name(attn):
    return 'POD_Attn' if 'pod_attn' in attn else \
            'HFuse' if 'hfuse' in attn else \
            'Libra' if 'load_attn' in attn else \
            'FI_Batch' if 'fi_bpf' in attn else attn

def read_record(path):
    model = prettify_model_name(get_substring(path, 'op_util/op_util_model_', '_method_'))
    attn = prettify_attn_name(get_substring(path, '_method_', '.csv'))
    if model == 'Llama-3-8B':
        return
    if model == 'Yi-6B':
        return
    # if model != 'Llama-13B':
    #     return
    # if model != 'Llama-2-7B':
    #     return

    df = pd.read_csv(path, skiprows=3, thousands=',')
    # print(df)
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
    warp_avg = pd.to_numeric(df['smsp__inst_executed.avg'].astype(str).str.replace(',', ''), errors='coerce')
    warp_min = pd.to_numeric(df['smsp__inst_executed.min'].astype(str).str.replace(',', ''), errors='coerce')
    warp_balance = (warp_max - warp_min) / warp_avg

    stall_long = pd.to_numeric(df['smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce')
    
    cycle_exec = pd.to_numeric(df['smsp__average_warps_active_per_inst_executed.ratio'].astype(str).str.replace(',', ''), errors='coerce')
    cycle_issue = pd.to_numeric(df['smsp__average_warp_latency_per_inst_issued.ratio'].astype(str).str.replace(',', ''), errors='coerce')
    cycle_sum = cycle_exec + cycle_issue
    mem_stall = stall_long / cycle_sum

    if attn not in record:
        record[attn] = {
            'compute_throughput': [],
            'dram_throughput': [],
            'sm_balance' : [],
            'warp_balance' : [],
            'stall_long': [],
            'cycle_exec': [],
            'cycle_issue': [],
            'cycle_sum': [],
            'mem_stall' : []
        }
    record[attn]['compute_throughput'].append(compute)
    record[attn]['dram_throughput'].append(dram)
    record[attn]['sm_balance'].append(sm_balance)
    record[attn]['warp_balance'].append(warp_balance)
    record[attn]['stall_long'].append(stall_long)
    record[attn]['cycle_exec'].append(cycle_exec)
    record[attn]['cycle_issue'].append(cycle_issue)
    record[attn]['cycle_sum'].append(cycle_sum)
    record[attn]['mem_stall'].append(mem_stall)

        
def read_logs():
    for root, dirs, files in os.walk(logs):
        for file in files:
            if not file.endswith('.csv'):
                continue
            path = os.path.join(root, file)
            read_record(path)

read_logs()
for attn, metrics in record.items():
    for metric, values in metrics.items():
        if values:
            record[attn][metric] = pd.concat(values, ignore_index=True)

# print(record)

methods = ["FI_Batch", 'POD_Attn', "HFuse", 'Libra']

plt.rcParams.update({'font.size': 24})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

colors = {
    'FI_Batch': '#D674FF',
    'POD_Attn': '#5CAFFF',
    'HFuse': '#47BC4C',
    'Libra': '#F98127',
}

labels = {
    'FI_Batch': 'FlashInfer',
    # 'fa_streams': 'Stream',
    # 'fi_batchprefill': 'FI_2',
    'POD_Attn': 'POD',
    'HFuse': 'HFuse',
    'Libra': " Libra",
}


schemes = [
    'fi_serial',
    # 'fa_streams',
    # 'fi_batchprefill',
    'pod_attn',
    'fa_hfuse',
    'libra'
]

lines = ['cmedians', 'cmins']

font_size = 30

def plt_violin(ax, balance_data, y_label):

    # plot violin plot
    vp = ax.violinplot(balance_data,
                        widths=0.7,
                        showmeans=False,
                        showmedians=True)

    lines = ['cbars', 'cmaxes', 'cmedians', 'cmins']

    # adding horizontal grid lines
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(balance_data))],
                labels=[labels[i] for i in methods],
                fontsize=font_size,
                )
    xticks = ax.get_xticklabels()
    # Set the first and third tick labels to bold
    for i, tick in enumerate(xticks):
            tick.set_fontweight('bold')
    for line in lines:
        vp[line].set_color("black")
        vp[line].set_alpha(0.5)
        vp[line].set_linewidth(2.0)

    # x_vals = np.array(ax.get_xlim())
    # y_vals = 1 + 0 * x_vals
    # ax.plot(x_vals, y_vals, '-', color='black', linewidth=2.5)

    ax.set_yticks(ax.get_yticks())
    ax.tick_params(axis='y', labelsize=26)
    ax.set_ylim(bottom=0)

    # vals = ax.get_yticks()
    # if y_label == 'Mem Stall Ratio':
    #     ax.set_yticklabels(['{:}%'.format(int(x * 100)) for x in vals],
    #                     fontsize=26)
            
    #ax.set_xlabel('')
    # ax.set_ylabel(y_label, fontsize=font_size, fontweight='bold')
    for it, title in enumerate(methods):
        vp['bodies'][it].set_facecolor(colors[title])
        vp['bodies'][it].set_alpha(0.75)
        #vp['bodies'][it].set_label(labels[title])

sm_balance_data = [
    record['FI_Batch']['sm_balance'].dropna().tolist(),
    record['POD_Attn']['sm_balance'].dropna().tolist(),
    record['HFuse']['sm_balance'].dropna().tolist(),
    record['Libra']['sm_balance'].dropna().tolist()
    ]

cta_balance_data = [
    record['FI_Batch']['mem_stall'].dropna().tolist(),
    record['POD_Attn']['mem_stall'].dropna().tolist(),
    record['HFuse']['mem_stall'].dropna().tolist(),
    record['Libra']['mem_stall'].dropna().tolist()
    ]

warp_balance_data = [
    record['FI_Batch']['warp_balance'].dropna().tolist(),
    record['POD_Attn']['warp_balance'].dropna().tolist(),
    record['HFuse']['warp_balance'].dropna().tolist(),
    record['Libra']['warp_balance'].dropna().tolist()
    ]

for i in ["sm_balance", "mem_stall", "warp_balance"]:
    print(f"====={i}=====")
    print("min_fi", min(record["FI_Batch"][i]))
    print("avg_fi", sum(record["FI_Batch"][i]) / len(record["FI_Batch"][i]))
    print("med_fi", np.median(record["FI_Batch"][i]))
    print("max_fi", max(record["FI_Batch"][i]))

    print("min_pod", min(record["POD_Attn"][i]))
    print("avg_pod", sum(record["POD_Attn"][i]) / len(record["POD_Attn"][i]))
    print("med_pod", np.median(record["POD_Attn"][i]))
    print("max_pod", max(record["POD_Attn"][i]))

    print("min_hfuse", min(record["HFuse"][i]))
    print("avg_hfuse", sum(record["HFuse"][i]) / len(record["HFuse"][i]))
    print("med_hfuse", np.median(record["HFuse"][i]))
    print("max_hfuse", max(record["HFuse"][i]))

    print("min_libra", min(record["Libra"][i]))
    print("avg_libra", sum(record["Libra"][i]) / len(record["Libra"][i]))
    print("med_libra", np.median(record["Libra"][i]))
    print("max_libra", max(record["Libra"][i]))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6), sharex=False)

plt_violin(ax1, sm_balance_data, "SM Balance Metric")
plt_violin(ax2, cta_balance_data, "CTA Balance Metric")
plt_violin(ax3, warp_balance_data, "Warp Balance Metric")

fig.supylabel(
    'Imbalance Metric',  # 你的标签文字
    fontsize=font_size+4,    # 字体大小
    fontweight='bold',     # 加粗
    x=0.02,                # 水平位置（靠左）
    va='center'            # 垂直居中
)
plt.tight_layout()

plt.savefig(output_dir + "/op_util_overall.pdf", bbox_inches='tight', pad_inches=1)
plt.savefig(
    os.path.join(output_dir + "/op_util_overall.svg"),
    format="svg",
    bbox_inches="tight",
    dpi=300,
    transparent=True    
)
