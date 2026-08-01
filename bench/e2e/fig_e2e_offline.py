import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import utils

_, root, _ = utils.get_paths()
logs_a100 = os.path.join(root, 'perf/e2e_offline_val_a100')
logs_a800 = os.path.join(root, 'perf/e2e_offline_val_a800')
plt.rcParams.update({'font.size': 28})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

configs = ["FA_Serial", "POD_Attn", "HFuse", "Load_Attn"]

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

opacity=0.65
fontsize = 30

def plot_figure(ax, df):
    # fig, ax = plt.subplots(figsize=(14, 5))
    models = df.index.tolist()
    # print(models)
    width = 0.20
    x_pos = np.arange(len(models)) * 1.1

    # 计算相对FA_Serial的加速比
    speedup_df = df.copy()
    for method in ["POD_Attn", "HFuse", "Load_Attn"]:
        speedup_df[method] = df[method] / df["FA_Serial"]
    speedup_df["FA_Serial"] = 1.0  # FA_Serial自身加速比为1

    ax.bar(x_pos - 1.5*width, speedup_df["FA_Serial"], width, label="FlashAttention", color=colors["FA_Serial"], hatch=hatches["FA_Serial"], alpha=opacity, edgecolor='black')
    # for i, v in enumerate(speedup_df["FA_Serial"]):
    #     ax.text(i - 1.5*width, v + 0.05, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)
    
    ax.bar(x_pos - 0.5*width, speedup_df["POD_Attn"], width, label="POD", color=colors["POD_Attn"], hatch=hatches["POD_Attn"], alpha=opacity, edgecolor='black')
    # for i, v in enumerate(speedup_df["POD_Attn"]):
    #     ax.text(i - 0.5*width, v + 0.05, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)
    
    ax.bar(x_pos + 0.5*width, speedup_df["HFuse"], width, label="HFuse", color=colors["HFuse"], hatch=hatches["HFuse"], alpha=opacity, edgecolor='black')
    # for i, v in enumerate(speedup_df["HFuse"]):
    #     ax.text(i + 0.5*width, v + 0.05, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)
    
    ax.bar(x_pos + 1.5*width, speedup_df["Load_Attn"], width, label="Libra", color=colors["Load_Attn"], hatch=hatches["Load_Attn"], alpha=opacity, edgecolor='black')
    # for i, v in enumerate(speedup_df["Load_Attn"]):
    #     ax.text(i + 1.5*width, v + 0.05, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)
    
    ax.set_xticks(x_pos, models, fontsize=fontsize, weight='bold')
    ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
    # ax.set_yticks(np.arange(0, 1.6, 0.5), fontsize=fontsize-4)
    ax.grid(axis='y', linestyle='--')
    # for label in ax.get_xticklabels():
    #     label.set_fontweight('bold')
    # plt.ylabel("Normalized Throughput", fontweight='bold', fontsize=fontsize)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.21), ncol=4, frameon=False, columnspacing=1, handletextpad=0.15, fontsize=fontsize, prop={'weight' : 'bold'})
    # # plt.tight_layout()
    # plt.subplots_adjust(top=0.8)
    # os.makedirs(os.path.join(root, "perf/fig/e2e_offline"), exist_ok=True)
    # plt.savefig(os.path.join(root, "perf/fig/e2e_offline/offline_perf_a100.pdf"))
    # plt.savefig(
    #     os.path.join(os.path.join(root, "perf/fig/e2e_offline"), 'e2e_offline.svg'),
    #     format="svg",
    #     bbox_inches="tight",
    #     dpi=300,            
    #     transparent=True    
    # )

record_a100 = {}
record_a800 = {}
def get_substring(string, start, end):
    return string[string.find(start)+len(start):string.find(end)]

def prettify_model_name(model):
    return 'Yi-6B' if 'yi-6b' in model else \
            'Llama2-7B' if 'llama-2-7b' in model else \
            'Llama3-8B' if 'llama-3-8b' in model else \
            'Llama2-13B' if 'llama-13b' in model else \
            'Progen2-6B' if 'progen' in model else \
            'Yi-34B' if 'yi-34b' in model else \
            'Mistral-7B' if 'mistral-7b' in model else \
            'Mixtral-8x7B' if 'mixtral-8x7b' in model else model

def prettify_attn_name(attn):
    return 'POD_Attn' if 'fa_pod' in attn else \
            'FA_Serial' if 'fa_paged' in attn or 'fa_sparse' in attn else \
            'Load_Attn' if 'la_paged' in attn or 'la_sparse' in attn else \
            'HFuse' if 'hfuse_vattn' in attn or 'hfuse_sparse' in attn else attn

def read_perf_record(path, record):
    model = prettify_model_name(get_substring(path, 'e2e_offline_val/model_', '_attn_'))
    attn = prettify_attn_name(get_substring(path, '_attn_', '_pdr_'))
    df = pd.read_csv(path)

    latency = df['request_e2e_time'].max()
    num_requests = len(df)
    if model not in record:
        record[model] = {}
    if attn not in record[model]:
        record[model][attn] = (num_requests * 60) / latency

def read_logs(logs, record):
    print(f"Logs path: {logs}")
    print(f"Path exists: {os.path.exists(logs)}")
    print(f"Is directory: {os.path.isdir(logs)}")
    for dir_path, dirs, files in os.walk(logs):
        for file in files:
            if file == 'sequence_metrics.csv':
                path = os.path.join(dir_path, file)
                read_perf_record(path, record)

read_logs(logs_a100, record_a100)
read_logs(logs_a800, record_a800)
df_a100 = pd.DataFrame.from_dict(record_a100).transpose()
df_a800 = pd.DataFrame.from_dict(record_a800).transpose()
# print(df)
#models = df.index.tolist()
# models = ["Llama-3-8B", "Llama-2-7B", "Llama-2-13B", "Yi-6B", "Yi-34B"]
# models = ["Llama-2-7B", "Llama-2-13B", "Yi-6B", "Yi-34B", "Progen2"]
# models = ["Llama-2-7B", "Llama-2-13B", "Yi-6B", "Yi-34B", "Progen-2-6B"]
# models = ["Progen-2-6B", "Yi-6B", "Llama-2-7B", "Llama-2-13B", "Yi-34B"]
# models = ["Progen-2-6B", "Llama-2-7B", "Yi-6B",  "Llama-2-13B", "Yi-34B"]
# models = ["Progen2-6B", "Llama2-7B", "Yi-6B",  "Llama2-13B", "Yi-34B"]
models_a100 = ["Progen2-6B", "Llama2-7B", "Llama3-8B", "Yi-6B",  "Llama2-13B", "Mistral-7B", "Mixtral-8x7B"]
models_a800 = ["Progen2-6B", "Llama2-7B", "Yi-6B",  "Llama2-13B", "Yi-34B", "Mistral-7B", "Mixtral-8x7B"]
# models = ["Llama-2-7B", "Llama-3-8B"]
methods = ['FA_Serial', 'POD_Attn', 'HFuse','Load_Attn']

for model in models_a100:
    if model not in df_a100.index:
        df_a100.loc[model] = 0
    for attn in methods:
        if attn not in df_a100.loc[model]:
            df_a100.loc[model, attn] = 0
df_a100.fillna(0, inplace=True)
df_a100 = df_a100.reindex(models_a100)

print("a100:")
print(df_a100)
print("a800:")
print(df_a800)

for model in models_a800:
    if model not in df_a800.index:
        df_a800.loc[model] = 0
    for attn in methods:
        if attn not in df_a800.loc[model]:
            df_a800.loc[model, attn] = 0
df_a800.fillna(0, inplace=True)
df_a800 = df_a800.reindex(models_a800)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 5), sharex=False)

plot_figure(ax1, df_a100)
plot_figure(ax2, df_a800)

fig.supylabel(
    'Normalized\nThroughput',  # 你的标签文字
    fontsize=fontsize+4,    # 字体大小
    fontweight='bold',     # 加粗
    x=0.03,                # 水平位置（靠左）
    va='center'            # 垂直居中
)
plt.tight_layout()


handles = [
    plt.Rectangle((0,0),1,1, color=colors['FA_Serial'], label='FlashAttention'),
    plt.Rectangle((0,0),1,1, color=colors['POD_Attn'], label='POD'),
    plt.Rectangle((0,0),1,1, color=colors['HFuse'], label='HFuse'),
    plt.Rectangle((0,0),1,1, color=colors['Load_Attn'], label='Libra'),
]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, 1.08), ncol=4, frameon=False, columnspacing=0.9, handletextpad=0.15, handlelength=2, handleheight=0.8, prop={'weight' : 'bold', 'size' : fontsize})

os.makedirs(os.path.join(root, "perf/fig/e2e_offline"), exist_ok=True)
plt.savefig(os.path.join(root, "perf/fig/e2e_offline/offline_perf_overall.pdf"))
plt.savefig(
    os.path.join(os.path.join(root, "perf/fig/e2e_offline"), 'e2e_offline_overall.svg'),
    format="svg",
    bbox_inches="tight",
    dpi=300,            
    transparent=True    
)