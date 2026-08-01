import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.ticker import PercentFormatter
import utils

_, root, _ = utils.get_paths()
logs = os.path.join(root, 'perf/e2e_online_val_overhead')
plt.rcParams.update({'font.size': 24})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
    qps_raw = get_substring(path, '_qps_', '_rqs_')
    try:
        qps = float(qps_raw) if "." in qps_raw else int(qps_raw)
    except ValueError:
        qps = qps_raw
    df = pd.read_csv(path)
    if attn not in record:
        record[attn] = {}
    if qps not in record[attn]:
        record[attn][qps] = df

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
    'schedule',
    'prepare_inputs_e2e',
    'sample_e2e',
    'process_model_outputs',
    'model_execution_e2e',
    'tile_schedule'
]

show_cols = [
    'input_processing',
    'model_execution_e2e',
    'output_processing',
    'tile_schedule'
]

label_mapper = {
    'schedule' : 'batching',
    'prepare_inputs_e2e' : 'prepare_input',
    'sample_e2e' : 'sample',
    'process_model_outputs' : 'process output',
    'input_processing' : 'input processing', # schedule + prepare_inputs_e2e
    'model_execution_e2e' : 'model execution', # sample_e2e + process_model_outputs
    'output_processing' : 'output processing',
    'tile_schedule' : 'Libra'
}

colors = {
    'process_model_outputs' : '#D674FF',
    'input_processing' : '#FC797D',
    'model_execution_e2e' : '#5CAFFF',
    'output_processing' : '#47BC4C',
    'tile_schedule' : '#F98127',
}

bin_sizes = {
    "Llama-3-8B" : 1024,
    "Llama-2-7B" : 4096,
    "Yi-6B" : 1024,
}


opacity=0.65

def plot_figure(rec_qps):
    
    binned_means = []
    bin_ranges = []
    n_qps = 0
    for qps in qps_list:
        df = rec_qps[qps]
        # print(qps)
        # print(df)
        seg = df.iloc[1:][stack_cols].mean()

        seg['input_processing'] = seg['schedule'] + seg['prepare_inputs_e2e']
        seg['output_processing'] = seg['sample_e2e'] + seg['process_model_outputs']
        
        drop_cols = [
            'schedule',
            'prepare_inputs_e2e',
            'sample_e2e',
            'process_model_outputs'
        ]
        seg = seg.drop(drop_cols)

        seg_sum = seg.sum()
        if seg_sum > 0:
            seg = seg / seg_sum  # 转为百分比
        binned_means.append(seg)
        n_qps += 1
        bin_ranges.append(qps)
        
        print(f"qps: {qps}, total len: {len(df)}, pct: {seg['tile_schedule']}")
    binned_df = pd.DataFrame(binned_means)

    x = np.arange(n_qps)
    bottom = np.zeros(n_qps)
    fig, ax = plt.subplots(figsize=(18, 7))
    width = 0.30
    fontsize = 36

    for col in show_cols:
        ax.bar(x, binned_df[col], width=width, bottom=bottom, label=label_mapper[col], color=colors[col])
        bottom += binned_df[col].values
    
    ax.set_xticks(x)
    ax.set_xticklabels(bin_ranges, fontsize=fontsize, rotation=0, ha='center', fontweight='bold')
    ax.set_yticks(np.arange(0, 1.1, 0.25))
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize-4)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.ylabel("Latency Percentage", fontweight='bold', fontsize=fontsize)
    plt.legend(loc='upper center', bbox_to_anchor=(0.495, 1.2), ncol=4, frameon=False, columnspacing=0.5, handlelength=0.8, handleheight=0.8, handletextpad=0.2, prop={'weight' : 'bold', 'size': fontsize-2})
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    os.makedirs(os.path.join(root, "perf/fig/e2e_online"), exist_ok=True)
    fig.savefig(os.path.join(root, "perf/fig/e2e_online/online_ovehead.pdf"), bbox_inches='tight', pad_inches=0.05)

read_logs()
# print(record)
# # models = ["Llama-3-8B", "Llama-2-7B", "Llama-2-13B", "Yi-6B", "Yi-34B"]
# # models = ["Llama-2-7B", "Llama-3-8B"]
# # methods = ['FA_Serial', 'POD_Attn', 'HFuse','Load_Attn']
# models = ["Llama-3-8B", "Llama-2-7B", "Yi-6B"]
models = ["Llama-2-7B"]
qps_list = [0.8, 0.9, 1.0, 1.1, 1.2]
methods = ['Load_Attn']
# bin_sizes = 

plot_figure(record[methods[0]])

