import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import utils

_, root, _ = utils.get_paths()
logs = os.path.join(root, 'perf/e2e_offline_val_lengthsensitivity')
plt.rcParams.update({'font.size': 28})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

configs = ["POD_Attn", "HFuse", "Load_Attn"]

colors = {
    'FA_Serial': "#D674FF",
    # 'fa_streams': "#FC797D",
    # 'fi_batchprefill': "#DB71F8",
    'POD_Attn': "#5CAFFF",
    'HFuse': "#47BC4C",
    'Load_Attn': "#F98127",
}

hatches = {
    'FA_Serial': '',
    'POD_Attn': '',
    'HFuse': '',
    'Load_Attn': '',
}

labels = {
    'FA_Serial': 'FlashAttention',
    'POD_Attn': 'POD',
    'HFuse': 'HFuse',
    'Load_Attn': 'Libra',
}

opacity=0.65

def plot_figure(df):
    fig, ax = plt.subplots(figsize=(18, 7))
    length_pos = df.index.tolist()
    width = 0.20
    x_pos = np.arange(len(length_pos))
    fontsize = 36

    # 计算相对FA_Serial的加速比
    speedup_df = df.copy()
    for method in ["POD_Attn", "Load_Attn"]:
        speedup_df[method] = df[method] / df["FA_Serial"]
    speedup_df["FA_Serial"] = 1.0  # FA_Serial自身加速比为1

    ax.bar(x_pos - 1*width, speedup_df["FA_Serial"], width, label=labels["FA_Serial"], color=colors["FA_Serial"], hatch=hatches["FA_Serial"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(speedup_df["FA_Serial"]):
        ax.text(i - 1*width, v + 0.05, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-16, rotation=0)
    
    ax.bar(x_pos - 0*width, speedup_df["POD_Attn"], width, label=labels["POD_Attn"], color=colors["POD_Attn"], hatch=hatches["POD_Attn"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(speedup_df["POD_Attn"]):
        ax.text(i - 0*width, v + 0.05, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-16, rotation=0)
    
    # ax.bar(x_pos + 0.5*width, speedup_df["HFuse"], width, label=labels["HFuse"], color=colors["HFuse"], hatch=hatches["HFuse"], alpha=opacity, edgecolor='black')
    # for i, v in enumerate(speedup_df["HFuse"]):
    #     ax.text(i + 0.5*width, v + 0.05, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-16, rotation=0)
    
    ax.bar(x_pos + 1*width, speedup_df["Load_Attn"], width, label=labels["Load_Attn"], color=colors["Load_Attn"], hatch=hatches["Load_Attn"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(speedup_df["Load_Attn"]):
        ax.text(i + 1*width, v + 0.02, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-16, rotation=0)
    
    plt.xticks(x_pos, length_pos, fontsize=fontsize)
    plt.yticks(np.arange(0, 1.6, 0.5), fontsize=fontsize-4)
    ax.grid(axis='y', linestyle='--')
    plt.ylabel("Normalized Throughput", fontweight='bold', fontsize=fontsize-10)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=fontsize-4)
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    os.makedirs(os.path.join(root, "perf/fig/e2e_offline"), exist_ok=True)
    plt.savefig(os.path.join(root, "perf/fig/e2e_offline/offline_length_sensitivity.pdf"))

record = {}
def get_substring(string, start, end):
    return string[string.find(start)+len(start):string.find(end)]

def prettify_length(cl):
    return '4k' if '4096' in cl else \
            '8k' if '8192' in cl else \
            '16k' if '16384' in cl else \
            '20k' if '20480' in cl else \
            '24k' if '24576' in cl else cl

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

def read_perf_record(path):
    model = prettify_model_name(get_substring(path, 'e2e_offline_val/model_', '_attn_'))
    attn = prettify_attn_name(get_substring(path, '_attn_', '_pdr_'))
    cl = prettify_attn_name(get_substring(path, '_cl_', '_sarathi'))
    df = pd.read_csv(path)

    latency = df['request_e2e_time'].max()
    num_requests = len(df)
    if cl not in record:
        record[cl] = {}
    if attn not in record[cl]:
        record[cl][attn] = (num_requests * 60) / latency

def read_logs():
    print(f"Logs path: {logs}")
    print(f"Path exists: {os.path.exists(logs)}")
    print(f"Is directory: {os.path.isdir(logs)}")
    for dir_path, dirs, files in os.walk(logs):
        for file in files:
            if file == 'sequence_metrics.csv':
                path = os.path.join(dir_path, file)
                read_perf_record(path)

read_logs()
df = pd.DataFrame.from_dict(record).transpose()
print(df)
#models = df.index.tolist()
lengths = ["4096", "8192", "16384", "20480", "24576"]
methods = ['FA_Serial', 'POD_Attn','Load_Attn']
for length in lengths:
    if length not in df.index:
        df.loc[length] = 0
    for attn in methods:
        if attn not in df.loc[length]:
            df.loc[length, attn] = 0
df.fillna(0, inplace=True)
df = df.reindex(lengths)
plot_figure(df)