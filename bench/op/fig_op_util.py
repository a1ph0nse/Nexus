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
    if model != 'Llama-3-8B':
        return
    # if model != 'Yi-6B':
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
    warp_avg = pd.to_numeric(df['smsp__inst_executed.avg'].astype(str).str.replace(',', ''), errors='coerce')
    warp_min = pd.to_numeric(df['smsp__inst_executed.min'].astype(str).str.replace(',', ''), errors='coerce')
    warp_balance = (warp_max - warp_min) / warp_avg

    if attn not in record:
        record[attn] = {
            'compute_throughput': [],
            'dram_throughput': [],
            'sm_balance' : [],
            'warp_balance' : [],
        }
    record[attn]['compute_throughput'].append(compute)
    record[attn]['dram_throughput'].append(dram)
    record[attn]['sm_balance'].append(sm_balance)
    record[attn]['warp_balance'].append(warp_balance)
        
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
plt.rcParams.update({'font.family': 'Sans Serif'})

colors = {
    'FI_Batch': '#549CD6',
    'POD_Attn': '#F896A3',
    'HFuse': '#F8545C',
    'Libra': '#61EE54',
}

lines = ['cmedians', 'cmins']

def plot_violin(metric_name, y_label, y_func):
    fib_data = record['FI_Batch'][metric_name].dropna().tolist()
    pod_data = record['POD_Attn'][metric_name].dropna().tolist()
    hfuse_data = record['HFuse'][metric_name].dropna().tolist()
    load_data = record['Libra'][metric_name].dropna().tolist()
    # print("len pod", len(pod_data))
    # print("len load", len(load_data))
    print(y_label)
    print("min_fib: ", min(fib_data))
    print("avg_fib: ", sum(fib_data) / len (fib_data))
    print("median_fib: ", np.median(fib_data))
    print("max_fib: ", max(fib_data))

    print("min_pod: ", min(pod_data))
    print("avg_pod: ", sum(pod_data) / len (pod_data))
    print("median_pod: ", np.median(pod_data))
    print("max_pod: ", max(pod_data))
    
    print("min_hfuse: ", min(hfuse_data))
    print("avg_hfuse: ", sum(hfuse_data) / len (hfuse_data))
    print("median_hfuse: ", np.median(hfuse_data))
    print("max_hfuse: ", max(hfuse_data))

    print("min_la: ", min(load_data))
    print("avg_la: ", sum(load_data) / len (load_data))
    print("median_la: ", np.median(load_data))
    print("max_la: ", max(load_data))
    if "throughput" in metric_name:
        # relative_improvement = [i / j for i, j in zip(load_data, pod_data)]
        # print("max_improvement: ", max(relative_improvement))
        # print("avg_improvement: ", sum(relative_improvement) / len (relative_improvement))
        print("pod max_improvement: ", max(pod_data) / max(fib_data))
        print("pod avg_improvement: ", sum(pod_data) / sum(fib_data))
        print("hfuse max_improvement: ", max(hfuse_data) / max(fib_data))
        print("hfuse avg_improvement: ", sum(hfuse_data) / sum(fib_data))
        print("la max_improvement: ", max(load_data) / max(fib_data))
        print("la avg_improvement: ", sum(load_data) / sum(fib_data))
    else:
        # relative_improvement = [j / i for i, j in zip(load_data, pod_data)]
        # print("max_improvement: ", max(relative_improvement))
        # print("avg_improvement: ", sum(relative_improvement) / sum(relative_improvement))
        print("pod max_improvement: ", max(fib_data) / max(pod_data))
        print("pod avg_improvement: ", sum(fib_data) / sum(pod_data))
        print("hfuse max_improvement: ", max(fib_data) / max(hfuse_data))
        print("hfuse avg_improvement: ", sum(fib_data) / sum(hfuse_data))
        print("la max_improvement: ", max(fib_data) / max(load_data))
        print("la avg_improvement: ", sum(fib_data) / sum(load_data))

    data_all = [fib_data, pod_data, hfuse_data, load_data]

    # Create the violin plot
    plt.figure(figsize=(10, 7))
    #fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    axs = ['a']
    axs[0] = plt.subplot(111)
    # plot violin plot
    vp = axs[0].violinplot(data_all,
                    showmeans=False,
                    showmedians=True)

    lines = ['cbars', 'cmaxes', 'cmedians', 'cmins']

    # adding horizontal grid lines
    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_xticks([y + 1 for y in range(len(data_all))],
                    labels=methods,
                    fontsize=30)
        xticks = ax.get_xticklabels()
        # Set the first and third tick labels to bold
        for i, tick in enumerate(xticks):
            if "Libra" in tick.get_text():
                tick.set_fontweight('bold')
        for line in lines:
            vp[line].set_color("black")
            vp[line].set_alpha(0.5)
            vp[line].set_linewidth(2.0)

        vals = ax.get_yticks()
        # ax.set_yticklabels(vals, fontsize=26)
        ax.set_yticklabels(y_func(vals), fontsize=26)
        # ax.set_yticklabels(['{:.0%}'.format(x / 100) for x in vals],
        #             fontsize=26)
        
        #ax.set_xlabel('')
        ax.set_ylabel(y_label, fontsize=30, fontweight='bold')
        for it, title in enumerate(methods):
            vp['bodies'][it].set_facecolor(colors[title])
            vp['bodies'][it].set_alpha(0.75)
            #vp['bodies'][it].set_label(labels[title])

    output_path = output_dir + '/' + metric_name + "_3work" + ".pdf"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=1) 
    plt.close()

plot_violin(
    metric_name='compute_throughput',
    y_label='Compute Throughput',
    y_func=lambda vals: ['{:.0%}'.format(x / 100) for x in vals], 
)

plot_violin(
    metric_name='dram_throughput',
    y_label='DRAM Throughput',
    y_func=lambda vals: ['{:.0%}'.format(x / 100) for x in vals],
)

plot_violin(
    metric_name='sm_balance',
    y_label='SM Balance',
    y_func=lambda vals: ['{:.1f}'.format(x) for x in vals],
)

plot_violin(
    metric_name='warp_balance',
    y_label='Warp Balance',
    y_func=lambda vals: ['{:.1f}'.format(x) for x in vals],
)