import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

src = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(src))
logs = os.path.join(root, 'perf/op_util')
output_dir = os.path.join(root, 'perf', 'fig', 'op')
os.makedirs(output_dir, exist_ok=True)

# src = os.path.dirname(os.path.abspath(__file__))
# logs = os.path.join(src, 'op_util')
# output_dir = src
# os.makedirs(output_dir, exist_ok=True)

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

def read_record(path, record, mmaper):
    # ===================== 关键：放开模型限制，读取所有模型 =====================
    model = prettify_model_name(get_substring(path, 'op_util/op_util_model_', '_method_'))
    attn = prettify_attn_name(get_substring(path, '_method_', '.csv'))

    df = pd.read_csv(path, skiprows=3, thousands=',')
    df = df.dropna(subset=['ID'])
    df = df.iloc[1:].reset_index(drop=True)

    if model not in mmaper:
        return  # 没有配置下标就跳过
    target_idx = mmaper[model]

    compute = pd.to_numeric(df['sm__throughput.avg.pct_of_peak_sustained_elapsed'], errors='coerce').iloc[target_idx]
    dram = pd.to_numeric(df['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'], errors='coerce').iloc[target_idx]
    sm_max = pd.to_numeric(df['sm__cycles_active.max'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    sm_avg = pd.to_numeric(df['sm__cycles_active.avg'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    sm_min = pd.to_numeric(df['sm__cycles_active.min'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    sm_balance = (sm_max - sm_min) / sm_avg
    warp_max = pd.to_numeric(df['smsp__inst_executed.max'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    warp_avg = pd.to_numeric(df['smsp__inst_executed.avg'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    warp_min = pd.to_numeric(df['smsp__inst_executed.min'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    warp_balance = (warp_max - warp_min) / warp_avg
    stall_long = pd.to_numeric(df['smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    
    cycle_exec = pd.to_numeric(df['smsp__average_warps_active_per_inst_executed.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    cycle_issue = pd.to_numeric(df['smsp__average_warp_latency_per_inst_issued.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    cycle_sum = cycle_exec + cycle_issue

    stall_sum = stall_long
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_drain_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_membar_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_misc_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_selected_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]
    stall_sum += pd.to_numeric(df['smsp__average_warps_issue_stalled_wait_per_issue_active.ratio'].astype(str).str.replace(',', ''), errors='coerce').iloc[target_idx]

    # 按 model + attn 存储
    key = (model, attn)
    if key not in record:
        record[key] = {
            'compute_throughput': [],
            'dram_throughput': [],
            'sm_balance': [],
            'stall_long': [],
            'stall_sum': [],
            'cycle_exec': [],
            'cycle_issue': [],
            'cycle_sum': [],
            'warp_balance': [],
            'sm_max': [],
            'sm_min': [],
            'warp_max': [],
            'warp_min': []

        }
    record[key]['compute_throughput'].append(compute)
    record[key]['dram_throughput'].append(dram)
    record[key]['stall_long'].append(stall_long)
    record[key]['stall_sum'].append(stall_sum)
    record[key]['cycle_exec'].append(cycle_exec)
    record[key]['cycle_issue'].append(cycle_issue)
    record[key]['cycle_sum'].append(cycle_sum)
    record[key]['sm_balance'].append(sm_balance)
    record[key]['warp_balance'].append(warp_balance)
    record[key]['sm_balance'].append(sm_balance)
    record[key]['warp_balance'].append(warp_balance)
    record[key]['sm_max'].append(sm_max)
    record[key]['sm_min'].append(sm_min)
    record[key]['warp_max'].append(warp_max)
    record[key]['warp_min'].append(warp_min)
    return record

def read_logs(mapper):
    record = {}
    for root_dir, dirs, files in os.walk(logs):
        for file in files:
            if not file.endswith('.csv'): continue
            path = os.path.join(root_dir, file)
            read_record(path, record, mapper)
    return record

# ===================== 绘图风格 =====================
plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

colors = {
    1 : '#D674FF',
    2 : '#FC797D',
    3 : '#5CAFFF',
    4 : '#47BC4C',
    5 : '#F98127',
}

def plot_combined_by_model(record):
    # print(record)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True, gridspec_kw={'wspace': 0.05})

    # fig = plt.figure(figsize=(12, 4))
    # gs = fig.add_gridspec(1, 3, wspace=0.30)  # 加大子图间距
    # ax1, ax2, ax3 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2])
    axes = [ax1, ax2, ax3]
    # for ax in axes:
    #     ax.set_position([ax.get_position().x0, ax.get_position().y0, 0.23, ax.get_position().height])
    font_size = 24
    width = 0.04

    metrics = [
        ('sm',      'SM Active Cycle',        ['sm_max', 'sm_min']),
        ('cta',   'Relative Cycle', ['cycle_sum', 'stall_long']),
        ('warp',    'Number of Instruction',    ['warp_max', 'warp_min'])
    ]

    model_list = sorted(list(set(k for k in record.keys())))
    x = np.arange(len(model_list)) * 0.12

    def draw_diff_lines(ax, x_pos, max_vals, min_vals, bar_width, fontsize, diff_func):
        """
        在最大值和最小值之间画线，并标注差值
        """
        for xi, max_val, min_val in zip(x_pos, max_vals, min_vals):
            diff_val = diff_func(max_val, min_val)
            if diff_val <= 0:
                continue  # 无差距不画
            
            # 两个柱子的中心 x 坐标
            x_max = xi
            x_min = xi + bar_width
            x_mid = xi + bar_width*0.5
            
            # 画水平线连接两个柱子顶部
            ax.plot([x_max, x_min], [max_val, max_val], 
                    color='black', linewidth=2, linestyle='-')
            ax.plot([x_max, x_min], [min_val, min_val], 
                    color='black', linewidth=2, linestyle='-')
            
            # 画竖线（可选，更明显）
            ax.plot([x_mid, x_mid], [min_val, max_val],
                    color='black', linewidth=2, linestyle='-')
            
            # 标注差值（放在竖线右侧/居中）
            ax.text(x_mid + 0.01, (min_val + max_val)/2, 
                    # f'{diff_val:.1f}',
                    f'{diff_val * 100:.0f}%',
                    ha='left', va='center', fontsize=fontsize, color='black', weight='bold')
            
    for ax, (plot_type, ylab, max_min_keys) in zip(axes, metrics):
        if plot_type == 'cta':
            key_max, key_min = max_min_keys
            # 改为相对值
            total_cycle = [1 for m in model_list]
            
            stall_cycle = [record[m]['POD_Attn'][key_min] / record[m]['POD_Attn'][key_max] for m in model_list]

            ax.bar(x - 0.5*width, total_cycle, width, color=colors[1], edgecolor='black', linewidth=0.8)
            ax.bar(x + 0.5*width, stall_cycle,  width, color=colors[4], edgecolor='black', linewidth=0.8)
            # draw_diff_lines(ax, x, total_cycle, stall_cycle, 
            #                 bar_width=width,
            #                 fontsize=font_size-4,
            #                 diff_func=lambda x, y : x - y)
        else:
            key_max, key_min = max_min_keys
            pod_max = [record[m]['POD_Attn'][key_max] for m in model_list]
            pod_min = [record[m]['POD_Attn'][key_min] for m in model_list]

            ax.bar(x - 0.5*width, pod_max, width, color=colors[1], edgecolor='black', linewidth=0.8)
            ax.bar(x + 0.5*width, pod_min, width, color=colors[4], edgecolor='black', linewidth=0.8)
            # draw_diff_lines(ax, x, pod_max, pod_min, 
            #                 bar_width=width,
            #                 fontsize=font_size-4,
            #                 diff_func=lambda x, y : (x - y) / x)

        ax.set_xticks(x)
        ax.set_xticklabels(model_list, fontsize=font_size+1, rotation=0, ha='center', fontweight='bold')
        # ax.set_xticklabels(["Llama-2-7B, Yi-6B"], fontsize=font_size+4, rotation=0, ha='center', fontweight='bold')
        ax.set_ylabel(ylab, fontsize=font_size+4, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.02)
        # plt.yticks(fontsize=font_size-4)
        ax.tick_params(axis='y', labelsize=font_size-6)
        # ===================== 科学计数法 =====================
        if plot_type != 'cta':
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
            ax.yaxis.get_offset_text().set_fontsize(font_size-8)
        # if plot_type == 'sm':
        #     ax.yaxis.set_offset_position('left')
        # if plot_type == 'warp':
        #     ax.yaxis.set_offset_position('right')

        # 图例
        if plot_type == 'cta':
            handles = [
                plt.Rectangle((0,0),1,1, color=colors[1], label='Total'),
                plt.Rectangle((0,0),1,1, color=colors[4], label='Mem Access'),
            ]
            ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.50,1.14), ncol=2, fontsize=font_size+4, frameon=False, columnspacing=0.2, handletextpad=0.15, handlelength=0.8, handleheight=0.8, prop={'weight' : 'bold'})
        else:
            handles = [
                plt.Rectangle((0,0),1,1, color=colors[1], label='Max'),
                plt.Rectangle((0,0),1,1, color=colors[4], label='Min'),
            ]
            ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.54,1.14), ncol=2, fontsize=font_size+4, frameon=False, columnspacing=0.5, handletextpad=0.15, handlelength=0.8, handleheight=0.8, prop={'weight' : 'bold'})

        # axes[0].text(0.5, -0.25, '(a)', transform=axes[0].transAxes, 
        #             fontsize=font_size+4, ha='center')
        # axes[1].text(0.5, -0.25, '(b)', transform=axes[1].transAxes, 
        #             fontsize=font_size+4, ha='center')
        # axes[2].text(0.5, -0.25, '(c)', transform=axes[2].transAxes, 
        #             fontsize=font_size+4, ha='center')

    # handles = [
    #     plt.Rectangle((0,0),1,1, color=colors[1], label='Max'),
    #     plt.Rectangle((0,0),1,1, color=colors[4], label='Min'),
    #     plt.Rectangle((0,0),1,1, color=colors[1], label='Total'),
    #     plt.Rectangle((0,0),1,1, color=colors[4], label='Mem Access'),
    #     plt.Rectangle((0,0),1,1, color=colors[1], label='Max'),
    #     plt.Rectangle((0,0),1,1, color=colors[4], label='Min'),
    # ]
    # fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.54,1.13), ncol=6, fontsize=font_size+4, frameon=False, columnspacing=0.9, handletextpad=0.15, handlelength=1.2, handleheight=1.2, prop={'weight' : 'bold'})
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.87)
    # plt.subplots_adjust(top=0.87, wspace=0.30)
    plt.savefig(os.path.join(output_dir, 'mov_example_ss.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(
        os.path.join(output_dir, 'mov_example_ss.svg'),
        format="svg",
        bbox_inches="tight",
        dpi=300,            
        transparent=True    
    )
    plt.show()


# SM 
MODEL_INDEX_MAP = {
    # "Llama-3-8B": 19,
    "Llama-2-7B": 15,
    "Yi-6B": 40
}
record_sm = read_logs(MODEL_INDEX_MAP)
# CTA
MODEL_INDEX_MAP = {
    # "Llama-3-8B": 19,
    "Llama-2-7B": 1,
    "Yi-6B": 21
}
record_cta = read_logs(MODEL_INDEX_MAP)
# Warp
MODEL_INDEX_MAP = {
    # "Llama-3-8B": 19,
    "Llama-2-7B": 15,
    "Yi-6B": 40
}
record_warp = read_logs(MODEL_INDEX_MAP)

model_list = ["Llama-2-7B", "Yi-6B"]
model_mapper = {
    "Llama-2-7B" : "Llama2-7B",
    "Yi-6B" : "Yi-6B"
}
attn_list = ['POD_Attn']

data_dict = {}
for model in model_list:
    data_dict[model_mapper[model]] = {}
    for attn in attn_list:
        data_dict[model_mapper[model]][attn] = {
            'sm_max': record_sm.get((model, attn), {}).get('sm_max', [0])[0],
            'sm_min': record_sm.get((model, attn), {}).get('sm_min', [0])[0],
            'stall_sum': record_cta.get((model, attn), {}).get('stall_sum', [0])[0],
            'stall_long': record_cta.get((model, attn), {}).get('stall_long', [0])[0],
            'cycle_exec': record_cta.get((model, attn), {}).get('cycle_exec', [0])[0],
            'cycle_issue': record_cta.get((model, attn), {}).get('cycle_issue', [0])[0],
            'cycle_sum': record_cta.get((model, attn), {}).get('cycle_sum', [0])[0],
            'warp_max': record_warp.get((model, attn), {}).get('warp_max', [0])[0],
            'warp_min': record_warp.get((model, attn), {}).get('warp_min', [0])[0]
        }
print(data_dict)
plot_combined_by_model(data_dict)