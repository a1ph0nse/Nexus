import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

src = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(src))
logs = os.path.join(root, 'perf/op_util')
output_dir = os.path.join(root, 'perf', 'fig', 'op')
os.makedirs(output_dir, exist_ok=True)


def read_all_records(filepath):
    """
    读取单个 NCU profile CSV 文件，返回所有行的指标数据。
    """
    df = pd.read_csv(filepath, skiprows=3, thousands=',')
    df = df.dropna(subset=['ID'])
    # 保留所有行（不跳过 iloc[1:]）

    def to_numeric(series):
        return pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce')

    # SM-level: cycles_active (max, avg, min)
    sm_max = to_numeric(df['sm__cycles_active.max'])
    sm_avg = to_numeric(df['sm__cycles_active.avg'])
    sm_min = to_numeric(df['sm__cycles_active.min'])

    # Warp-level: inst_executed (max, avg, min)
    warp_max = to_numeric(df['smsp__inst_executed.max'])
    warp_avg = to_numeric(df['smsp__inst_executed.avg'])
    warp_min = to_numeric(df['smsp__inst_executed.min'])

    # CTA-level: stall_long and stall_sum
    stall_long = to_numeric(df['smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'])

    # stall_sum = 所有 stall 原因之和
    stall_cols = [
        'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_drain_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_membar_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_misc_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_selected_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio',
        'smsp__average_warps_issue_stalled_wait_per_issue_active.ratio',
    ]
    stall_sum = pd.Series(0.0, index=df.index)
    for col in stall_cols:
        stall_sum += to_numeric(df[col])

    # cycle_exec + cycle_issue (用于 cta 级别，保留兼容)
    cycle_exec = to_numeric(df['smsp__average_warps_active_per_inst_executed.ratio'])
    cycle_issue = to_numeric(df['smsp__average_warp_latency_per_inst_issued.ratio'])
    cycle_sum = cycle_exec + cycle_issue

    return {
        'sm_max': sm_max.values,
        'sm_avg': sm_avg.values,
        'sm_min': sm_min.values,
        'warp_max': warp_max.values,
        'warp_avg': warp_avg.values,
        'warp_min': warp_min.values,
        'stall_long': stall_long.values,
        'stall_sum': stall_sum.values,
        'cycle_sum': cycle_sum.values,
    }


# ===================== 绘图风格 =====================
plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 颜色方案
colors = {
    'max':  '#D674FF',   # 紫色 - 最大值
    'avg':  '#5CAFFF',   # 蓝色 - 平均值
    'min':  '#FC797D',   # 红色 - 最小值
    'stall_sum': '#F98127',  # 橙色 - stall_sum
    'stall_long': '#47BC4C',  # 绿色 - stall_long
}


def plot_line_charts(record, selected_indices, x_labels):
    """
    绘制折线图，横坐标为 context length，仅绘制选定的几个 index。
    record: read_all_records 返回的完整数据 dict（每个字段是 numpy array）
    selected_indices: 要绘制的 index 列表
    x_labels: 横坐标标签列表（context length）
    """
    font_size = 18

    # 从完整数据中提取选定 index 的指标
    def pick(arr):
        return [arr[i] for i in selected_indices]

    sm_max_vals = pick(record['sm_max'])
    sm_avg_vals = pick(record['sm_avg'])
    sm_min_vals = pick(record['sm_min'])

    stall_sum_vals = pick(record['stall_sum'])
    stall_long_vals = pick(record['stall_long'])

    warp_max_vals = pick(record['warp_max'])
    warp_avg_vals = pick(record['warp_avg'])
    warp_min_vals = pick(record['warp_min'])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    # ===================== 子图1: SM Active Cycles =====================
    ax1.plot(x_labels, sm_max_vals, color=colors['max'], linewidth=2, label='Max',
             marker='o', markersize=10, markerfacecolor='white', markeredgewidth=2)
    ax1.plot(x_labels, sm_avg_vals, color=colors['avg'], linewidth=2, label='Avg',
             marker='s', markersize=10, markerfacecolor='white', markeredgewidth=2)
    ax1.plot(x_labels, sm_min_vals, color=colors['min'], linewidth=2, label='Min',
             marker='^', markersize=10, markerfacecolor='white', markeredgewidth=2)
    ax1.set_ylabel('SM Active Cycles', fontsize=font_size + 2, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=font_size - 2, frameon=False, ncol=3)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax1.tick_params(axis='y', labelsize=font_size - 4)
    ax1.yaxis.get_offset_text().set_fontsize(font_size - 6)

    # ===================== 子图2: CTA Relative Stall Cycles =====================
    stall_long_rel = [l / s for l, s in zip(stall_long_vals, stall_sum_vals)]
    ax2.plot(x_labels, [1.0] * len(x_labels), color=colors['stall_sum'], linewidth=2, label='Stall Sum',
             marker='o', markersize=10, markerfacecolor='white', markeredgewidth=2)
    ax2.plot(x_labels, stall_long_rel, color=colors['stall_long'], linewidth=2, label='Stall Long',
             marker='s', markersize=10, markerfacecolor='white', markeredgewidth=2)
    ax2.set_ylabel('Relative Stall Cycles', fontsize=font_size + 2, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=font_size - 2, frameon=False, ncol=2)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.tick_params(axis='y', labelsize=font_size - 4)

    # ===================== 子图3: Warp Instructions =====================
    ax3.plot(x_labels, warp_max_vals, color=colors['max'], linewidth=2, label='Max',
             marker='o', markersize=10, markerfacecolor='white', markeredgewidth=2)
    ax3.plot(x_labels, warp_avg_vals, color=colors['avg'], linewidth=2, label='Avg',
             marker='s', markersize=10, markerfacecolor='white', markeredgewidth=2)
    ax3.plot(x_labels, warp_min_vals, color=colors['min'], linewidth=2, label='Min',
             marker='^', markersize=10, markerfacecolor='white', markeredgewidth=2)
    ax3.set_xlabel('Prefill Context Length', fontsize=font_size + 2, fontweight='bold')
    ax3.set_ylabel('Number of Instructions', fontsize=font_size + 2, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=font_size - 2, frameon=False, ncol=3)
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax3.tick_params(axis='y', labelsize=font_size - 4)
    ax3.yaxis.get_offset_text().set_fontsize(font_size - 6)

    # x 轴设置（三个子图都显示相同的 x labels）
    for ax in (ax1, ax2, ax3):
        ax.set_xticks(x_labels)
        ax.set_xticklabels([str(x) for x in x_labels], fontsize=font_size - 4)
        ax.tick_params(axis='x', labelsize=font_size - 4)

    # 隐藏 ax1, ax2 的 x 轴刻度标签（保留刻度线）
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])

    plt.savefig(os.path.join(output_dir, 'mov_example_v3.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(
        os.path.join(output_dir, 'mov_example_v3.svg'),
        format="svg",
        bbox_inches="tight",
        dpi=300,
        transparent=True
    )
    plt.show()


if __name__ == '__main__':
    # 只读取 Llama-2-7B (llama-7b-tp4) + POD_Attn 的数据
    target_file = os.path.join(logs, 'op_util_model_llama-7b-tp4_method_pod_attn.csv')

    if not os.path.exists(target_file):
        print(f"File not found: {target_file}")
        exit(1)

    print(f"Reading: {target_file}")
    record = read_all_records(target_file)
    print(f"Loaded {len(record['sm_max'])} data points total")

    # ===================== 选定数据点 =====================
    # 从 run_op.py 的 seqlen_configs 生成顺序可知：
    #   chunk 2048, 第一个 var len 组 (max prefill=24576 那组):
    #     idx 60: [(24576,1)]*1 + [(1024,1)]*53 + [(2048,2048)]*1   → varying prefill CL = 2048
    #     idx 61: [(24576,1)]*1 + [(1024,1)]*53 + [(4096,2048)]*1   → varying prefill CL = 4096
    #     idx 62: [(24576,1)]*1 + [(1024,1)]*53 + [(6144,2048)]*1   → varying prefill CL = 6144
    #     idx 63: [(24576,1)]*1 + [(1024,1)]*53 + [(8192,2048)]*1   → varying prefill CL = 8192
    # 这 4 组数据中，除一个 prefill query 的 context length 随循环改变外，
    # 其余 54 个 query 的 shape 保持不变，可用来观察 prefill context length 对负载不均的影响。
    selected_indices = [60, 61, 62, 63]
    x_labels = [2048, 4096, 6144, 8192]

    plot_line_charts(record, selected_indices, x_labels)
