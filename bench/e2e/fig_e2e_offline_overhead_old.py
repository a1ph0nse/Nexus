import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.ticker import PercentFormatter
import utils

_, root, _ = utils.get_paths()
logs = os.path.join(root, 'perf/e2e_offline_val_overhead')
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Sans Serif'})

record = {}
def get_substring(string, start, end):
    return string[string.find(start)+len(start):string.find(end)]

def prettify_model_name(model):
    return 'Yi-6B' if 'yi-6b' in model else \
            'Llama-2-7B' if 'llama-2-7b' in model else \
            'Llama-3-8B' if 'llama-3-8b' in model else \
            'Llama-2-13B' if 'llama-13b' in model else \
            'Yi-34B' if 'yi-34b' in model else model

def prettify_attn_name(attn):
    return 'POD_Attn' if 'fa_pod' in attn else \
            'FA_Serial' if 'fa_paged' in attn else \
            'Load_Attn' if 'la_paged' in attn else \
            'HFuse' if 'hfuse_vattn' in attn else attn

def read_operation_latency(path):
    model = prettify_model_name(get_substring(path, 'e2e_offline_val/model_', '_attn_'))
    attn = prettify_attn_name(get_substring(path, '_attn_', '_pdr_'))
    df = pd.read_csv(path)
    if model not in record:
        record[model] = {}
    if attn not in record[model]:
        record[model][attn] = df

def read_logs():
    print(f"Logs path: {logs}")
    print(f"Path exists: {os.path.exists(logs)}")
    print(f"Is directory: {os.path.isdir(logs)}")
    for dir_path, dirs, files in os.walk(logs):
        for file in files:
            if file == 'cpu_operation_metrics.csv':
                path = os.path.join(dir_path, file)
                read_operation_latency(path)

stack_cols = [
    # 'schedule',
    # 'sample_e2e',
    # 'prepare_inputs_e2e',
    'model_execution_e2e',
    'process_model_outputs',
    'tile_schedule'
]

colors = {
    'schedule': "#F896A3",
    'sample_e2e': "#FD913F",
    'prepare_inputs_e2e': "#DB71F8",
    'model_execution_e2e': "#549CD6",
    'process_model_outputs': "#F8545C",
    'tile_schedule': "#61EE54",
}

bin_sizes = {
    "Llama-3-8B" : 1024,
    "Llama-2-7B" : 4096,
    "Yi-6B" : 1024,
}

opacity=0.65
max_n_bins = 5  # 这里是每1/10分段

def plot_figure_on_ax(df, ax, bin_size):
    
    # bin_size = len(df) // n_bins
    # bin_size = 4096
    n_bins = len(df) // bin_size
    n_bins = min(n_bins, max_n_bins)

    binned_means = []
    bin_ranges = []

    for i in range(n_bins):
        start = i * bin_size + 1
        end = (i + 1) * bin_size + 1 if i < n_bins - 1 else len(df)
        batch_start = int(df.iloc[start]['Batch Id'])
        batch_end = int(df.iloc[end - 1]['Batch Id'])
        bin_ranges.append(f"{batch_start // 1024}k-{batch_end // 1024}k")
        seg = df.iloc[start:end][stack_cols].mean()
        seg_sum = seg.sum()
        if seg_sum > 0:
            seg = seg / seg_sum  # 转为百分比
        binned_means.append(seg)
    binned_df = pd.DataFrame(binned_means)

    x = np.arange(n_bins)
    bottom = np.zeros(n_bins)
    
    # width = 0.15
    fontsize = 16
    for col in stack_cols:
        if col == 'tile_schedule':
            print(f"total len: {len(df)}, max pct: {binned_df[col].max()}, min pct: {binned_df[col].min()}")
        ax.bar(x, binned_df[col], bottom=bottom, label=col, color=colors[col])
        bottom += binned_df[col].values
    
    ax.set_xticks(x)
    ax.set_xticklabels(bin_ranges, fontsize=fontsize-2, rotation=0)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize-2)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

read_logs()
# print(record)
# # models = ["Llama-3-8B", "Llama-2-7B", "Llama-2-13B", "Yi-6B", "Yi-34B"]
# # models = ["Llama-2-7B", "Llama-3-8B"]
# # methods = ['FA_Serial', 'POD_Attn', 'HFuse','Load_Attn']
models = ["Llama-3-8B", "Llama-2-7B", "Yi-6B"]
# models = ["Llama-2-7B"]
methods = ['Load_Attn']
# bin_sizes = 

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
for r, attn in enumerate(methods):
    for c, model in enumerate(models):
        plot_figure_on_ax(record[model][attn], axes[c], bin_sizes[model])

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=20)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.supylabel("Percentage", x=0.01, va='center', fontsize=24, fontweight='bold')
os.makedirs(os.path.join(root, "perf/fig/e2e_offline"), exist_ok=True)
fig.savefig(os.path.join(root, "perf/fig/e2e_offline/offline_ovehead.pdf"), bbox_inches='tight', pad_inches=0.05)
