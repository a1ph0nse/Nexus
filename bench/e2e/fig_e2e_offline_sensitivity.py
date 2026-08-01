import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import utils

_, root, _ = utils.get_paths()
logs_len = os.path.join(root, 'perf/e2e_offline_val_lengthsensitivity')
logs_pd = os.path.join(root, 'perf/e2e_offline_val_pdsensitivity')
logs_rqs = os.path.join(root, 'perf/e2e_offline_val_rqssensitivity')
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

def plot_figure_subplot(df):
    fig, ax = plt.subplots(figsize=(18, 5))
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
    # for i, v in enumerate(speedup_df["FA_Serial"]):
    #     ax.text(i - 1*width, v + 0.05, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-16, rotation=0)
    
    ax.bar(x_pos - 0*width, speedup_df["POD_Attn"], width, label=labels["POD_Attn"], color=colors["POD_Attn"], hatch=hatches["POD_Attn"], alpha=opacity, edgecolor='black')
    # for i, v in enumerate(speedup_df["POD_Attn"]):
    #     ax.text(i - 0*width, v + 0.05, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-16, rotation=0)
    
    # ax.bar(x_pos + 0.5*width, speedup_df["HFuse"], width, label=labels["HFuse"], color=colors["HFuse"], hatch=hatches["HFuse"], alpha=opacity, edgecolor='black')
    # for i, v in enumerate(speedup_df["HFuse"]):
    #     ax.text(i + 0.5*width, v + 0.05, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-16, rotation=0)
    
    ax.bar(x_pos + 1*width, speedup_df["Load_Attn"], width, label=labels["Load_Attn"], color=colors["Load_Attn"], hatch=hatches["Load_Attn"], alpha=opacity, edgecolor='black')
    # for i, v in enumerate(speedup_df["Load_Attn"]):
    #     ax.text(i + 1*width, v + 0.02, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-16, rotation=0)
    
    plt.xticks(x_pos, length_pos, fontsize=fontsize)
    plt.yticks(np.arange(0, 1.6, 0.5), fontsize=fontsize-4)
    ax.grid(axis='y', linestyle='--')
    plt.ylabel("Normalized Throughput", fontweight='bold', fontsize=fontsize-10)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=fontsize-4)
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    # os.makedirs(os.path.join(root, "perf/fig/e2e_offline"), exist_ok=True)
    # plt.savefig(os.path.join(root, "perf/fig/e2e_offline/offline_length_sensitivity.pdf"))

def plot_figure_overall(df_len, df_pd, df_req):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(26, 4), constrained_layout=True, gridspec_kw={'wspace': 0.12})
    axes = [ax1, ax2, ax3]
    df_overall = [df_len, df_pd, df_req]
    fontsize = 26
    width = 0.30

    for ax, df in zip(axes, df_overall):
        x_name = df.index.tolist()
        x = np.arange(len(x_name)) * 1
        
        speedup_df = df.copy()
        for method in ["POD_Attn", "Load_Attn"]:
            speedup_df[method] = df[method] / df["FA_Serial"]
        speedup_df["FA_Serial"] = 1.0  # FA_Serial自身加速比为1

        ax.bar(x - 1*width, speedup_df["FA_Serial"], width, label=labels["FA_Serial"], color=colors["FA_Serial"], hatch=hatches["FA_Serial"], alpha=opacity, edgecolor='black')
        # for i, v in enumerate(speedup_df["FA_Serial"]):
        #     ax.text(i - 1*width, v + 0.05, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-12, rotation=0)
        
        ax.bar(x - 0*width, speedup_df["POD_Attn"], width, label=labels["POD_Attn"], color=colors["POD_Attn"], hatch=hatches["POD_Attn"], alpha=opacity, edgecolor='black')
        # for i, v in enumerate(speedup_df["POD_Attn"]):
        #     ax.text(i - 0*width, v + 0.05, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-12, rotation=0)
        
        ax.bar(x + 1*width, speedup_df["Load_Attn"], width, label=labels["Load_Attn"], color=colors["Load_Attn"], hatch=hatches["Load_Attn"], alpha=opacity, edgecolor='black')
        # for i, v in enumerate(speedup_df["Load_Attn"]):
        #     ax.text(i + 1*width, v + 0.02, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-12, rotation=0)
        
        ax.set_xticks(x, x_name, fontsize=fontsize, weight='bold')
        # ax.yticks(np.arange(0, 1.6, 0.5), fontsize=fontsize-4)
        ax.grid(axis='y', linestyle='--')
        ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
        # ax.ylabel("Normalized Throughput", fontweight='bold', fontsize=fontsize-10)

    handles = [
        plt.Rectangle((0,0),1,1, color=colors['FA_Serial'], label='FlashAttention'),
        plt.Rectangle((0,0),1,1, color=colors['POD_Attn'], label='POD'),
        plt.Rectangle((0,0),1,1, color=colors['Load_Attn'], label='Libra'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.51, 1.06), ncol=3, frameon=False, columnspacing=0.9, handletextpad=0.15, handlelength=2, handleheight=0.8, prop={'weight' : 'bold', 'size' : fontsize})
    fig.supylabel("Normalized Throughput", fontweight='bold', fontsize=fontsize, x=0.085)
    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
    plt.subplots_adjust(top=0.87, wspace=0.30)
    

    plt.savefig(os.path.join(os.path.join(root, "perf/fig/e2e_offline"), 'sensitivity.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(
        os.path.join(os.path.join(root, "perf/fig/e2e_offline"), 'sensitivity.svg'),
        format="svg",
        bbox_inches="tight",
        dpi=300,            
        transparent=True    
    )

def get_substring(string, start, end):
    return string[string.find(start)+len(start):string.find(end)]

def prettify_length(cl):
    return  '1K' if '1024' == cl else \
            '2K' if '2048' == cl else \
            '4K' if '4096' == cl else \
            '8K' if '8192' == cl else \
            '16K' if '16384' == cl else \
            '20K' if '20480' == cl else \
            '24K' if '24576' == cl else cl

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

def read_perf_record(path, record, data_type):
    model = prettify_model_name(get_substring(path, 'e2e_offline_val/model_', '_attn_'))
    attn = prettify_attn_name(get_substring(path, '_attn_', '_pdr_'))
    pdr = get_substring(path, '_pdr_', '_rqs_')
    cl = prettify_length(get_substring(path, '_cl_', '_sarathi'))
    if cl == "4K":
        return
    req = prettify_length(get_substring(path, '_rqs_', '_cl_'))
    df = pd.read_csv(path)

    latency = df['request_e2e_time'].max()
    num_requests = len(df)
    if data_type == "cl":
        if cl not in record:
            record[cl] = {}
        if attn not in record[cl]:
            record[cl][attn] = (num_requests * 60) / latency
    if data_type == "req":
        if req not in record:
            record[req] = {}
        if attn not in record[req]:
            record[req][attn] = (num_requests * 60) / latency
    if data_type == "pd":
        if pdr not in record:
            record[pdr] = {}
        if attn not in record[pdr]:
            record[pdr][attn] = (num_requests * 60) / latency

def read_logs(logs, record, data_type):
    print(f"Logs path: {logs}")
    print(f"Path exists: {os.path.exists(logs)}")
    print(f"Is directory: {os.path.isdir(logs)}")
    for dir_path, dirs, files in os.walk(logs):
        for file in files:
            if file == 'sequence_metrics.csv':
                path = os.path.join(dir_path, file)
                read_perf_record(path, record, data_type)

record_len = {}
record_pd = {}
record_req = {}

read_logs(logs_len, record_len, 'cl')
read_logs(logs_pd, record_pd, 'pd')
read_logs(logs_rqs, record_req, 'req')

df_len = pd.DataFrame.from_dict(record_len).transpose()
df_pd = pd.DataFrame.from_dict(record_pd).transpose()
df_req = pd.DataFrame.from_dict(record_req).transpose()
print(df_len)
print(df_pd)
print(df_req)

#models = df.index.tolist()
# lengths = ["4096", "8192", "16384", "20480", "24576"]
lengths = ["8K", "16K", "20K", "24K"]
pds = ["8", "16", "32", "64"]
reqs = ["256", "512", "1K", "2K"]

methods = ['FA_Serial', 'POD_Attn','Load_Attn']
for length in lengths:
    if length not in df_len.index:
        df_len.loc[length] = 0
    for attn in methods:
        if attn not in df_len.loc[length]:
            df_len.loc[length, attn] = 0
df_len.fillna(0, inplace=True)
df_len = df_len.reindex(lengths)

for pd in pds:
    if pd not in df_pd.index:
        df_pd.loc[pd] = 0
    for attn in methods:
        if attn not in df_pd.loc[pd]:
            df_pd.loc[pd, attn] = 0
df_pd.fillna(0, inplace=True)
df_pd = df_pd.reindex(pds)

for req in reqs:
    if req not in df_req.index:
        df_req.loc[req] = 0
    for attn in methods:
        if attn not in df_req.loc[req]:
            df_req.loc[req, attn] = 0
df_req.fillna(0, inplace=True)
df_req = df_req.reindex(reqs)

plot_figure_overall(df_len, df_pd, df_req)
