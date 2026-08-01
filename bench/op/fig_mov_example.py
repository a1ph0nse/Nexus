import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


src = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(src))
logs = os.path.join(root, 'perf/op_util')
output_dir = os.path.join(root, 'perf', 'fig', 'op')
os.makedirs(output_dir, exist_ok=True)

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


def read_record(path, mmaper):
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
    record[key]['sm_balance'].append(sm_balance)
    record[key]['warp_balance'].append(warp_balance)
    record[key]['sm_balance'].append(sm_balance)
    record[key]['warp_balance'].append(warp_balance)
    record[key]['sm_max'].append(sm_max)
    record[key]['sm_min'].append(sm_min)
    record[key]['warp_max'].append(warp_max)
    record[key]['warp_min'].append(warp_min)

def read_logs(mapper):
    for root_dir, dirs, files in os.walk(logs):
        for file in files:
            if not file.endswith('.csv'): continue
            path = os.path.join(root_dir, file)
            read_record(path, mapper)

# ===================== 绘图风格 =====================
plt.rcParams.update({'font.size': 22})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
colors = {'POD_Attn': '#549CD6', 'HFuse': '#F896A3'}


# ===================== 1. SM Active Cycles 单独图 =====================
def plot_sm_cycles():
    fig = plt.figure(figsize=(12, 10))  # 单图尺寸，可按需调整
    ax = fig.add_subplot(111)
    font_size = 48
    # 配置参数
    model_list = sorted(list(set(k[0] for k in record.keys())))
    width = 0.15
    x = np.arange(len(model_list)) * 0.5  # 保持你要的横坐标间距
    colors = {
        'POD_min': '#D674FF',
        'POD_max': '#47BC4C',
        'HFuse_min': '#C73E1D',
        'HFuse_max': '#F896A3',
    }

    # 提取数据
    key_max, key_min = 'sm_max', 'sm_min'
    pod_max = [record.get((m, 'POD_Attn'), {}).get(key_max, [0])[0] for m in model_list]
    pod_min = [record.get((m, 'POD_Attn'), {}).get(key_min, [0])[0] for m in model_list]

    # 绘制柱子
    ax.bar(x - 0.5*width, pod_max, width, color=colors['POD_max'], edgecolor='black', linewidth=0.8)
    ax.bar(x + 0.5*width, pod_min, width, color=colors['POD_min'], edgecolor='black', linewidth=0.8)

    # 样式设置
    ax.set_xticks(x)
    ax.set_xticklabels(model_list, fontsize=font_size-2, rotation=0, ha='center', fontweight='bold')
    # ax.set_ylabel('SM Active Cycles', fontsize=font_size+4, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.05)
    plt.yticks(fontsize=font_size-4)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    # 图例（单图专属，放在图内更美观）
    handles = [
        plt.Rectangle((0,0),1,1, color=colors['POD_max'], label='Max'),
        plt.Rectangle((0,0),1,1, color=colors['POD_min'], label='Min'),
    ]
    # ax.legend(handles=handles, loc='upper right', fontsize=font_size-2, frameon=True, prop={'weight' : 'bold'})
    # ax.legend(handles=handles, loc='upper right', fontsize=font_size-2, frameon=True)

    # 保存&显示
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sm_active_cycles.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(
        os.path.join(output_dir, 'sm_active_cycles.svg'),
        format="svg",
        bbox_inches="tight",
        dpi=300,            
        transparent=True    
    )
    plt.show()

# ===================== 2. Memory Access Stall Ratio 单独图 =====================
def plot_mem_stall():
    fig = plt.figure(figsize=(12, 10))  # 单图尺寸，可按需调整
    ax = fig.add_subplot(111)
    font_size = 48
    # 配置参数
    model_list = sorted(list(set(k[0] for k in record.keys())))
    width = 0.15
    x = np.arange(len(model_list)) * 0.5  # 保持你要的横坐标间距
    colors = {
        'POD_min': '#D674FF',
        'POD_max': '#47BC4C',
        'HFuse_min': '#C73E1D',
        'HFuse_max': '#F896A3',
    }

    # 提取数据
    pod_stall_sum  = [record.get((m, 'POD_Attn'), {}).get('stall_sum', [0])[0] for m in model_list]
    pod_stall_mem  = [record.get((m, 'POD_Attn'), {}).get('stall_long', [0])[0] for m in model_list]
    hf_stall_sum  = [record.get((m, 'HFuse'), {}).get('stall_sum', [0])[0] for m in model_list]
    hf_stall_mem = [record.get((m, 'HFuse'), {}).get('stall_long', [0])[0] for m in model_list]

    pod_stall_sum = [1 for m in model_list]
            
    pod_stall_mem = [record.get((m, 'POD_Attn'), {}).get('stall_long', [0])[0] / record.get((m, 'POD_Attn'), {}).get('stall_sum', [0])[0] for m in model_list]


    # pod_vals = [record.get((m, 'POD_Attn'), {}).get('stall_long', [0])[0] / record.get((m, 'POD_Attn'), {}).get('stall_sum', [0])[0] for m in model_list]
    # hf_vals  = [record.get((m, 'HFuse'),    {}).get('stall_long', [0])[0] / record.get((m, 'HFuse'), {}).get('stall_sum', [0])[0] for m in model_list]

    # 绘制柱子
    ax.bar(x - 0.5*width, pod_stall_sum, width, color=colors['POD_max'], edgecolor='black', linewidth=0.8)
    ax.bar(x + 0.5*width, pod_stall_mem, width, color=colors['POD_min'], edgecolor='black', linewidth=0.8)
    # ax.bar(x + 0.5*width, hf_stall_sum,  width, color=colors['HFuse_max'], edgecolor='black', linewidth=0.8)
    # ax.bar(x + 1.5*width, hf_stall_mem,  width, color=colors['HFuse_min'], edgecolor='black', linewidth=0.8)

    # 样式设置
    ax.set_xticks(x)
    ax.set_xticklabels(model_list, fontsize=font_size-2, rotation=0, ha='center', fontweight='bold')
    # ax.set_ylabel('Stall Cycle', fontsize=font_size+4, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.05)
    plt.yticks(fontsize=font_size-4)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    # 图例（单图专属，放在图内更美观）
    handles = [
        plt.Rectangle((0,0),1,1, color=colors['POD_max'], label='total stall'),
        plt.Rectangle((0,0),1,1, color=colors['POD_min'], label='mem stall'),
        # plt.Rectangle((0,0),1,1, color=colors['HFuse_max'], label='HFuse total stall'),
        # plt.Rectangle((0,0),1,1, color=colors['HFuse_min'], label='HFuse mem stall'),
    ]
    # ax.legend(handles=handles, loc='upper right', fontsize=font_size-2, frameon=True, prop={'weight' : 'bold'})
    # ax.legend(handles=handles, loc='upper right', fontsize=font_size-2, frameon=True)

    # 保存&显示
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_access_stall_ratio.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(
        os.path.join(output_dir, 'memory_access_stall_ratio.svg'),
        format="svg",
        bbox_inches="tight",
        dpi=300,            
        transparent=True    
    )
    plt.show()

# ===================== 3. Warp Instruction Executed 单独图 =====================
def plot_warp_instruction():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    font_size = 48

    # 配置参数
    model_list = sorted(list(set(k[0] for k in record.keys())))
    width = 0.15
    x = np.arange(len(model_list)) * 0.5
    colors = {
        'POD_min': '#D674FF',
        'POD_max': '#47BC4C',
        'HFuse_min': '#C73E1D',
        'HFuse_max': '#F896A3',
    }

    # 提取数据
    key_max, key_min = 'warp_max', 'warp_min'
    pod_max = [record.get((m, 'POD_Attn'), {}).get(key_max, [0])[0] for m in model_list]
    pod_min = [record.get((m, 'POD_Attn'), {}).get(key_min, [0])[0] for m in model_list]
    hf_max  = [record.get((m, 'HFuse'),    {}).get(key_max, [0])[0] for m in model_list]
    hf_min  = [record.get((m, 'HFuse'),    {}).get(key_min, [0])[0] for m in model_list]

    # 绘制柱子
    ax.bar(x - 0.5*width, pod_max, width, color=colors['POD_max'], edgecolor='black', linewidth=0.8)
    ax.bar(x + 0.5*width, pod_min, width, color=colors['POD_min'], edgecolor='black', linewidth=0.8)
    # ax.bar(x + 0.5*width, hf_max,  width, color=colors['HFuse_max'], edgecolor='black', linewidth=0.8)
    # ax.bar(x + 1.5*width, hf_min,  width, color=colors['HFuse_min'], edgecolor='black', linewidth=0.8)

    # 样式设置
    ax.set_xticks(x)
    ax.set_xticklabels(model_list, fontsize=font_size-2, rotation=0, ha='center', fontweight='bold')
    # ax.set_ylabel('Warp Instruction Executed', fontsize=font_size+4, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.05)
    plt.yticks(fontsize=font_size-4)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    # 图例
    handles = [
        plt.Rectangle((0,0),1,1, color=colors['POD_max'], label='Max'),
        plt.Rectangle((0,0),1,1, color=colors['POD_min'], label='Min'),
        # plt.Rectangle((0,0),1,1, color=colors['HFuse_max'], label='HFuse (Max)'),
        # plt.Rectangle((0,0),1,1, color=colors['HFuse_min'], label='HFuse (Min)'),
    ]
    # ax.legend(handles=handles, loc='upper right', fontsize=font_size-2, frameon=True, prop={'weight' : 'bold'})
    # ax.legend(handles=handles, loc='upper right', fontsize=font_size-2, frameon=True)

    # 保存&显示
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'warp_instruction_executed.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(
        os.path.join(output_dir, 'warp_instruction_executed.svg'),
        format="svg",
        bbox_inches="tight",
        dpi=300,            
        transparent=True    
    )
    plt.show()

# ===================== 调用方式（按需执行） =====================
MODEL_INDEX_MAP = {
    "Llama-3-8B": 19,    # 模型对应下标
    "Llama-2-7B": 15,
    "Yi-6B": 40
}
record = {}
read_logs(MODEL_INDEX_MAP)

# ===================== 构建按模型整理的 DataFrame =====================
model_list = sorted(list(set(k[0] for k in record.keys())))
attn_list = ['POD_Attn', 'HFuse']

data_dict = {}
for attn in attn_list:
    data_dict[attn] = {}
    for model in model_list:
        val = record.get((model, attn), {}).get('sm_balance', [0])[0]
        data_dict[attn][model] = {
            'sm_balance': record.get((model, attn), {}).get('sm_balance', [0])[0],
            'stall_long': record.get((model, attn), {}).get('stall_long', [0])[0],
            'warp_balance': record.get((model, attn), {}).get('warp_balance', [0])[0]
        }
plot_sm_cycles()          # 生成SM活跃周期图

MODEL_INDEX_MAP = {
    "Llama-3-8B": 19,    # 模型对应下标
    "Llama-2-7B": 1,
    "Yi-6B": 21
}
record = {}
read_logs(MODEL_INDEX_MAP)
# ===================== 构建按模型整理的 DataFrame =====================
model_list = sorted(list(set(k[0] for k in record.keys())))
attn_list = ['POD_Attn', 'HFuse']

data_dict = {}
for attn in attn_list:
    data_dict[attn] = {}
    for model in model_list:
        val = record.get((model, attn), {}).get('sm_balance', [0])[0]
        data_dict[attn][model] = {
            'sm_balance': record.get((model, attn), {}).get('sm_balance', [0])[0],
            'stall_long': record.get((model, attn), {}).get('stall_long', [0])[0],
            'warp_balance': record.get((model, attn), {}).get('warp_balance', [0])[0]
        }
plot_mem_stall()       # 生成访存停顿率图

MODEL_INDEX_MAP = {
    "Llama-3-8B": 19,    # 模型对应下标
    "Llama-2-7B": 15,
    "Yi-6B": 40
}
record = {}
read_logs(MODEL_INDEX_MAP)
# ===================== 构建按模型整理的 DataFrame =====================
model_list = sorted(list(set(k[0] for k in record.keys())))
attn_list = ['POD_Attn', 'HFuse']

data_dict = {}
for attn in attn_list:
    data_dict[attn] = {}
    for model in model_list:
        val = record.get((model, attn), {}).get('sm_balance', [0])[0]
        data_dict[attn][model] = {
            'sm_balance': record.get((model, attn), {}).get('sm_balance', [0])[0],
            'stall_long': record.get((model, attn), {}).get('stall_long', [0])[0],
            'warp_balance': record.get((model, attn), {}).get('warp_balance', [0])[0]
        }
plot_warp_instruction()   # 生成Warp执行指令数图