import os
import re
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
# import utils

# src, root = utils.get_paths()
src = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(src))

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# parser = argparse.ArgumentParser(description='create fig for op perf')
# parser.add_argument('input1', type=str, default='op_perf_v11.csv', help='Input CSV file')
# parser.add_argument('input2', type=str, default='op_perf_v11_a800.csv', help='Input CSV file')

# args = parser.parse_args()

input_path_a100 = os.path.join(root, 'perf', 'op_perf', 'op_perf_v11.csv')
input_path_a800 = os.path.join(root, 'perf', 'op_perf', 'op_perf_v11_a800.csv')
output_path = os.path.join(root, 'perf', 'fig', 'op', 'op_perf_merge')
print(f"input_path: {input_path_a100}")
print(f"input_path: {input_path_a800}")
print(f"output_path: {output_path}")

[
    'model', 'seqlen_config', 'causal', 'block_size',
    'fa_p', 'fa_d', 'fa_serial',
    'fi_p', 'fi_d', 'fi_serial',
    'fa_stream', 'fi_bpf', 'fa_hfuse', 'fa_pod', 'la',
    'speedup(fa_serial/fi_serial)',
    'speedup(fa_serial/fa_stream)',
    'speedup(fa_serial/fi_bpf)',
    'speedup(fa_serial/fa_hfuse)',
    'speedup(fa_serial/fa_pod)',
    'speedup(fa_serial/la)'
]

def read_data(csv_filename):
    input_path = os.path.join(root, 'perf', 'op_perf', csv_filename)
    
    fa_serial = []
    fi_serial = []
    fa_stream = []
    fi_batchprefill = []
    fa_hfuse = []
    pod_attn = []
    load_attn = []

    speedup_fi_serial = []
    speedup_fa_stream = []
    speedup_fi_batchprefill = []
    speedup_fa_hfuse = []
    speedup_pod_attn = []
    speedup_load_attn = []

    with open(input_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            try:
                fa_serial.append(float(row['fa_serial']))
                fi_serial.append(float(row['fi_serial']))
                fa_stream.append(float(row['fa_stream']))
                fi_batchprefill.append(float(row['fi_bpf']))
                pod_attn.append(float(row['fa_pod']))
                fa_hfuse.append(float(row['fa_hfuse']))
                load_attn.append(float(row['la']))
                
                speedup_fi_serial.append(float(row['speedup(fa_serial/fi_serial)']))
                speedup_fa_stream.append(float(row['speedup(fa_serial/fa_stream)']))
                speedup_fi_batchprefill.append(float(row['speedup(fa_serial/fi_bpf)']))
                speedup_fa_hfuse.append(float(row['speedup(fa_serial/fa_hfuse)']))
                speedup_pod_attn.append(float(row['speedup(fa_serial/fa_pod)']))
                speedup_load_attn.append(float(row['speedup(fa_serial/la)']))

            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {e}. Row: {row}")
                continue

    speedup = [
        speedup_fi_serial,
        speedup_fa_stream,
        speedup_pod_attn,
        speedup_fa_hfuse,
        speedup_load_attn
    ]
    return speedup

# print(not_good_case)
# print("min_speedup_fi_batch: ", min(speedup_fi_batchprefill))

a100_speedup = read_data(input_path_a100)
a800_speedup = read_data(input_path_a800)
# print(a100_speedup)

schemes = [
    'fi_serial',
    'fa_streams',
    # 'fi_batchprefill',
    'pod_attn',
    'fa_hfuse',
    'libra'
]

colors = {
    'fi_serial': "#D674FF",
    'fa_streams': "#FC797D",
    # 'fi_batchprefill': "#DB71F8",
    'pod_attn': "#5CAFFF",
    'fa_hfuse': "#47BC4C",
    'libra': "#F98127",
}

labels = {
    'fi_serial': 'FlashInfer',
    'fa_streams': 'Stream',
    # 'fi_batchprefill': 'FI_2',
    'pod_attn': 'POD',
    'fa_hfuse': 'HFuse',
    'libra': " Libra",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 6), sharex=False) 
font_size = 40

def plt_violin(ax, speedup_data):
    # plot violin plot
    vp = ax.violinplot(speedup_data,
                        widths=0.7,
                        showmeans=False,
                        showmedians=True)

    # plot box plot
    #axs[1].boxplot(speedup)
    #axs[1].set_title('Box plot')

    lines = ['cbars', 'cmaxes', 'cmedians', 'cmins']

    # adding horizontal grid lines
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(speedup_data))],
                labels=[labels[i] for i in schemes],
                fontsize=font_size,
                )
    xticks = ax.get_xticklabels()
    # Set the first and third tick labels to bold
    for i, tick in enumerate(xticks):
            tick.set_fontweight('bold')
    for line in lines:
        vp[line].set_color("black")
        vp[line].set_alpha(0.5)
        vp[line].set_linewidth(2.0)

    x_vals = np.array(ax.get_xlim())
    y_vals = 1 + 0 * x_vals
    ax.plot(x_vals, y_vals, '-', color='black', linewidth=2.5)

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:.1f}'.format(x) for x in vals],
                    fontsize=26)
    
    #ax.set_xlabel('')
    # ax.set_ylabel('Normlized Speedup\n over FlashAttention', fontsize=font_size-2, fontweight='bold')
    for it, title in enumerate(schemes):
        vp['bodies'][it].set_facecolor(colors[title])
        vp['bodies'][it].set_alpha(0.75)
        #vp['bodies'][it].set_label(labels[title])

plt_violin(ax1, a100_speedup)
plt_violin(ax2, a800_speedup)

fig.supylabel(
    'Normalized Speedup\nover FlashAttention',  # 你的标签文字
    fontsize=font_size,    # 字体大小
    fontweight='bold',     # 加粗
    x=0.00,                # 水平位置（靠左）
    va='center'            # 垂直居中
)
plt.tight_layout()

plt.savefig(output_path + ".pdf", bbox_inches='tight', pad_inches=1)
plt.savefig(
    os.path.join(output_path + ".svg"),
    format="svg",
    bbox_inches="tight",
    dpi=300,
    transparent=True    
)