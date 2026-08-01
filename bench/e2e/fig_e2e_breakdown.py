import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.ticker import PercentFormatter
import utils

_, root, _ = utils.get_paths()
logs_on = os.path.join(root, 'perf/e2e_online_val_overhead')
logs_off = os.path.join(root, 'perf/e2e_offline_val_overhead')
plt.rcParams.update({'font.size': 24})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

record_qps = {}
record_model = {}
def get_substring(string, start, end):
    return string[string.find(start)+len(start):string.find(end)]

def prettify_model_name(model):
    return 'Yi-6B' if 'yi-6b' in model else \
            'Llama2-7B' if 'llama-2-7b' in model else \
            'Llama3-8B' if 'llama-3-8b' in model else \
            'Llama2-13B' if 'llama-13b' in model else \
            'Yi-34B' if 'yi-34b' in model else model

def prettify_attn_name(attn):
    return 'POD_Attn' if 'fa_pod' in attn else \
            'FA_Serial' if 'fa_paged' in attn else \
            'Load_Attn' if 'la_paged' in attn else \
            'HFuse' if 'hfuse_vattn' in attn else attn

def read_operation_latency_qps(path, record):
    model = prettify_model_name(get_substring(path, 'e2e_offline_val/model_', '_attn_'))
    attn = prettify_attn_name(get_substring(path, '_attn_', '_pdr_'))
    qps_raw = get_substring(path, '_qps_', '_rqs_')
    try:
        qps = float(qps_raw) if "." in qps_raw else int(qps_raw)
    except ValueError:
        qps = qps_raw
    df = pd.read_csv(path)
    if qps not in record:
        record[qps] = df
    
def read_operation_latency_model(path, record):
    model = prettify_model_name(get_substring(path, 'e2e_offline_val/model_', '_attn_'))
    attn = prettify_attn_name(get_substring(path, '_attn_', '_pdr_'))
    df = pd.read_csv(path)
    if model not in record:
        record[model] = df

def read_logs(logs, record, target):
    print(f"Logs path: {logs}")
    print(f"Path exists: {os.path.exists(logs)}")
    print(f"Is directory: {os.path.isdir(logs)}")
    for dir_path, dirs, files in os.walk(logs):
        for file in files:
            if file == 'cpu_operation_metrics.csv':
                path = os.path.join(dir_path, file)
                if target == "qps":
                    read_operation_latency_qps(path, record)
                else:
                    read_operation_latency_model(path, record)

stack_cols = [
    'schedule',
    'prepare_inputs_e2e',
    'sample_e2e',
    'process_model_outputs',
    'model_execution_e2e',
    'tile_schedule'
]

show_cols = [
    'output_processing',
    'model_execution_e2e',
    'input_processing',
    'tile_schedule'
]

label_mapper = {
    'schedule' : 'batching',
    'prepare_inputs_e2e' : 'prepare_input',
    'sample_e2e' : 'sample',
    'process_model_outputs' : 'process output',
    'input_processing' : 'Input Pre-processing', # schedule + prepare_inputs_e2e
    'model_execution_e2e' : 'Workload Execution', # sample_e2e + process_model_outputs
    'output_processing' : 'Output Post-processing',
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
fontsize = 40

def plot_figure(ax, record, xlabels):
    
    binned_means = []
    bin_ranges = []
    n_xlabel = 0
    for xlabel in xlabels:
        df = record[xlabel]
        # print(xlabel)
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
        n_xlabel += 1
        bin_ranges.append(xlabel)
        
        print(f"xlabel: {xlabel}, total len: {len(df)}, pct: {seg['tile_schedule']}")
    binned_df = pd.DataFrame(binned_means)

    min_bar_height = 0.1
    plot_df = binned_df.copy()
    
    for idx in range(len(plot_df)):
        row = plot_df.iloc[idx]
        row_clamped = row.clip(lower=min_bar_height)
        plot_df.iloc[idx] = row_clamped / row_clamped.sum()

    # print(binned_df)

    x = np.arange(n_xlabel) * 0.8
    bottom = np.zeros(n_xlabel)
    width = 0.4

    for col in show_cols:
        ax.bar(x, plot_df[col], width=width, bottom=bottom, label=label_mapper[col], color=colors[col])
        bottom += plot_df[col].values

    bottom_real = np.zeros(n_xlabel)
    for col in show_cols:
        true_values = binned_df[col].values
        plot_values = plot_df[col].values
        
        for i in range(len(x)):
            x_pos = x[i]
            val_plot = plot_values[i]
            val_true = true_values[i]
            
            y_pos = bottom_real[i] + val_plot / 2
            
            ax.text(
                x_pos, y_pos,
                f'{val_true*100:.1f}%',  # 显示真实值
                ha='center', va='center',
                fontsize=fontsize-4,
                fontweight='bold',
                color='black'
            )
        
        bottom_real += plot_values
    
    ax.set_xticks(x)
    ax.set_xticklabels(bin_ranges, fontsize=fontsize, rotation=0, ha='center', fontweight='bold')
    ax.set_yticks(np.arange(0, 1.1, 0.25))
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize-4)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(top=ax.get_ylim()[1] * 1.08)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

read_logs(logs_on, record_qps, 'qps')
read_logs(logs_off, record_model, 'model')

models = ["Yi-6B", "Llama2-7B", "Llama3-8B", "Llama2-13B", "Yi-34B"]
qps_list = [0.8, 0.9, 1.0, 1.1, 1.2]
# bin_sizes = 

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), constrained_layout=True, gridspec_kw={'wspace': 0.05})

plot_figure(ax1, record_model, models)
plot_figure(ax2, record_qps, qps_list)

handles = [
        plt.Rectangle((0,0),1,1, color=colors[show_cols[0]], label=label_mapper[show_cols[0]]),
        plt.Rectangle((0,0),1,1, color=colors[show_cols[1]], label=label_mapper[show_cols[1]]),
        plt.Rectangle((0,0),1,1, color=colors[show_cols[2]], label=label_mapper[show_cols[2]]),
        plt.Rectangle((0,0),1,1, color=colors[show_cols[3]], label=label_mapper[show_cols[3]]),
    ]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.54,1.15), ncol=2, frameon=False, columnspacing=0.9, handletextpad=0.15, handlelength=1.2, handleheight=1.2, prop={'weight' : 'bold', 'size' : fontsize})
fig.supylabel(
    'Latency Breakdown',  # 你的标签文字
    fontsize=fontsize,    # 字体大小
    fontweight='bold',     # 加粗
    x=-0.03,                # 水平位置（靠左）
    va='center'            # 垂直居中
)
plt.savefig(os.path.join(root, "perf/fig/e2e_online/breakdown.pdf"), bbox_inches='tight', dpi=300)
plt.savefig(
    os.path.join(root, "perf/fig/e2e_online/breakdown.svg"),
    format="svg",
    bbox_inches="tight",
    dpi=300,            
    transparent=True    
)
plt.show()