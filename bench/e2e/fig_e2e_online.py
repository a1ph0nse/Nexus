import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from collections import defaultdict

src = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(src))

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

# 三维结构: model -> attn -> qps -> metrics(dict)
records = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

def parse_meta(path: str):
    """从路径中提取 model / attn / qps"""
    model = prettify_model_name(get_substring(path, 'model_', '_attn_'))
    attn = prettify_attn_name(get_substring(path, '_attn_', '_qps_'))
    qps_raw = get_substring(path, '_qps_', '_rqs_')
    try:
        qps = float(qps_raw) if "." in qps_raw else int(qps_raw)
    except ValueError:
        qps = qps_raw
    return model, attn, qps

def upsert_metrics(path: str, metric_dict: dict):
    model, attn, qps = parse_meta(path)
    records[model][attn][qps].update(metric_dict)

def read_perf_record(path):
    df = pd.read_csv(path)
    num_requests = len(df)
    metric_dict = {
        'latency_p50': df['request_e2e_time'].quantile(0.5),
        'latency_p99': df['request_e2e_time'].quantile(0.99),
        'ttft': df['prefill_e2e_time'].max() / num_requests,
    }
    upsert_metrics(path, metric_dict)

def read_tpot_record(path):
    df = pd.read_csv(path)
    num_requests = len(df)
    metric_dict = {
        'tpot': df['decode_token_execution_plus_preemption_time'].max() / num_requests,
    }
    upsert_metrics(path, metric_dict)

def build_main_df():
    rows = []
    for model, attn_map in records.items():
        for attn, qps_map in attn_map.items():
            for qps, metrics in qps_map.items():
                row = {'model': model, 'attn': attn, 'qps': qps}
                row.update(metrics)
                rows.append(row)
    return pd.DataFrame(rows)

def build_sub_df(long_df, metric):
    return long_df.pivot_table(
        index=['model', 'qps'],
        columns='attn',
        values=metric,
        aggfunc='first'
    ).reset_index()

def read_logs(logs):
    for root, dirs, files in os.walk(logs):
        for file in files:
            if file == 'sequence_metrics.csv':
                path = os.path.join(root, file)
                read_perf_record(path)
            if file == 'decode_token_execution_plus_preemption_time.csv':
                path = os.path.join(root, file)
                read_tpot_record(path)

def plot_figure_on_ax(ax, df, metric_label):
    df = df.sort_values("qps").reset_index(drop=True)
    qps_labels = df["qps"].astype(str).tolist()

    width = 0.15
    group_step = 0.75
    x_pos = np.arange(len(df)) * group_step
    fontsize = 16

    for col in ["FA_Serial", "POD_Attn", "HFuse", "Load_Attn"]:
        if col not in df.columns:
            df[col] = np.nan

    norm_df = df.copy()
    for method in ["POD_Attn", "HFuse", "Load_Attn"]:
        norm_df[method] = df[method] / df["FA_Serial"]
    norm_df["FA_Serial"] = 1.0

    b1 = ax.bar(x_pos - 1.5 * width, norm_df["FA_Serial"], width, label="FlashAttention",
                color=colors["FA_Serial"], hatch=hatches["FA_Serial"], alpha=opacity, edgecolor='black')
    b2 = ax.bar(x_pos - 0.5 * width, norm_df["POD_Attn"], width, label="POD",
                color=colors["POD_Attn"], hatch=hatches["POD_Attn"], alpha=opacity, edgecolor='black')
    b3 = ax.bar(x_pos + 0.5 * width, norm_df["HFuse"], width, label="HFuse",
                color=colors["HFuse"], hatch=hatches["HFuse"], alpha=opacity, edgecolor='black')
    b4 = ax.bar(x_pos + 1.5 * width, norm_df["Load_Attn"], width, label="Libra",
                color=colors["Load_Attn"], hatch=hatches["Load_Attn"], alpha=opacity, edgecolor='black')

    # ax.bar_label(b1, fmt="%.2f", padding=1, fontsize=fontsize-8, color="black")
    # ax.bar_label(b2, fmt="%.2f", padding=1, fontsize=fontsize-8, color="black")
    # ax.bar_label(b3, fmt="%.2f", padding=1, fontsize=fontsize-8, color="black")
    # ax.bar_label(b4, fmt="%.2f", padding=1, fontsize=fontsize-8, color="black")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(qps_labels, fontsize=fontsize+4)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    ax.set_yticks(np.arange(0, 1.4, 0.4))
    ax.set_ylim(0, 1.4)
    ax.tick_params(axis='y', labelsize=fontsize-2)
    # ax.set_ylabel(metric_label, fontweight='bold',fontsize=fontsize)

    ax.grid(axis='y', linestyle='--', alpha=0.5)


if __name__ == "__main__":
    dir_names = ['A100_yi', 'A800_llama2']
    metric_name = ['ttft', 'tpot', 'latency_p50', 'latency_p99']
    label_map = {
        'ttft': "Normalized TTFT",
        'tpot': "Normalized TPOT",
        'latency_p50': "Normalized Latency P50",
        'latency_p99': "Normalized Latency P99"
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 5), sharey=False)

    for r, dir_name in enumerate(dir_names):
        records = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        logs = os.path.join(root, f'perf/e2e_online_val_{dir_name}')
        read_logs(logs)
        df_main = build_main_df().sort_values(['model', 'qps', 'attn'])
        print(df_main)

        for c, m in enumerate(metric_name):
            sub_df = build_sub_df(df_main, m)
            plot_figure_on_ax(axes[r, c], sub_df, label_map[m])

    # 全局图例（只取一次）
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=30)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False, prop={'weight' : 'bold', 'size': 20})
    fig.supylabel(
        'Normalized Latency',  # 你的标签文字
        fontsize=20,    # 字体大小
        fontweight='bold',     # 加粗
        x=0.01,                # 水平位置（靠左）
        va='center'            # 垂直居中
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.join(root, "perf/fig/e2e_online"), exist_ok=True)
    fig.savefig(os.path.join(root, "perf/fig/e2e_online/online_perf_overall.pdf"), bbox_inches='tight', pad_inches=0.05)
    plt.savefig(
        os.path.join(root, "perf/fig/e2e_online/online_perf_overall.svg"),
        format="svg",
        bbox_inches="tight",
        dpi=300,            
        transparent=True    
    )