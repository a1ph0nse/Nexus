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

parser = argparse.ArgumentParser(description='create fig for op perf')
parser.add_argument('input1', type=str, default='op_perf_v11.csv', help='Input CSV file')
parser.add_argument('input2', type=str, default='op_perf_v11_a800.csv', help='Input CSV file')

args = parser.parse_args()

input_path = os.path.join(root, 'perf', 'op_perf', args.input1)
output_path = os.path.join(root, 'perf', 'fig', 'op', args.input1[:-4])
print(f"input_path: {input_path}")
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

not_good_case = []

min_speedup_pod = 100

with open(input_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        try:

            # Append to results
            fa_serial.append(float(row['fa_serial']))
            fi_serial.append(float(row['fi_serial']))
            fa_stream.append(float(row['fa_stream']))
            fi_batchprefill.append(float(row['fi_bpf']))
            pod_attn.append(float(row['fa_pod']))
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

# print(not_good_case)
# print("min_speedup_fi_batch: ", min(speedup_fi_batchprefill))

print("min_speedup_fi: ", min(speedup_fi_serial))
print("max_speedup_fi: ", max(speedup_fi_serial))
print("avg_speedup_fi: ", sum(speedup_fi_serial) / len (speedup_fi_serial))

print("min_speedup_stream: ", min(speedup_fa_stream))
print("max_speedup_stream: ", max(speedup_fa_stream))
print("avg_speedup_stream: ", sum(speedup_fa_stream) / len (speedup_fa_stream))

print("min_speedup_fi_bpf: ", min(speedup_fi_batchprefill))
print("max_speedup_fi_bpf: ", max(speedup_fi_batchprefill))
print("avg_speedup_fi_bpf: ", sum(speedup_fi_batchprefill) / len (speedup_fi_batchprefill))

print("min_speedup_pod: ", min(speedup_pod_attn))
print("max_speedup_pod: ", max(speedup_pod_attn))
print("avg_speedup_pod: ", sum(speedup_pod_attn) / len (speedup_pod_attn))

print("min_speedup_hfuse: ", min(speedup_fa_hfuse))
print("max_speedup_hfuse: ", max(speedup_fa_hfuse))
print("avg_speedup_hfuse: ", sum(speedup_fa_hfuse) / len (speedup_fa_hfuse))

print("min_speedup_la: ", min(speedup_load_attn))
print("max_speedup_la: ", max(speedup_load_attn))
print("avg_speedup_la: ", sum(speedup_load_attn) / len (speedup_load_attn))

print("fi_bpr slow than FA:", sum(1 for x in speedup_fi_batchprefill if x < 1) / len(speedup_fi_batchprefill))
print("pod slow than FA:", sum(1 for x in speedup_pod_attn if x < 1) / len(speedup_pod_attn))
print("hfuse slow than FA:", sum(1 for x in speedup_fa_hfuse if x < 1) / len(speedup_fa_hfuse))
print("la slow than FA:", sum(1 for x in speedup_load_attn if x < 1) / len(speedup_load_attn))

speedup_fi_load = [i / j for i, j in zip(speedup_load_attn, speedup_fi_serial)]
speedup_stream_load = [i / j for i, j in zip(speedup_load_attn, speedup_fa_stream)]
speedup_pod_load = [i / j for i, j in zip(speedup_load_attn, speedup_pod_attn)]
speedup_hfuse_load = [i / j for i, j in zip(speedup_load_attn, speedup_fa_hfuse)]

print("min speedup over fi: ", min(speedup_fi_load))
print("max speedup over fi: ", max(speedup_fi_load))
print("avg speedup over fi: ", sum(speedup_fi_load) / len(speedup_fi_load))

print("min speedup over stream: ", min(speedup_stream_load))
print("max speedup over stream: ", max(speedup_stream_load))
print("avg speedup over stream: ", sum(speedup_stream_load) / len(speedup_stream_load))

print("min speedup over pod: ", min(speedup_pod_load))
print("max speedup over pod: ", max(speedup_pod_load))
print("avg speedup over pod: ", sum(speedup_pod_load) / len(speedup_pod_load))

print("min speedup over hfuse: ", min(speedup_hfuse_load))
print("max speedup over hfuse: ", max(speedup_hfuse_load))
print("avg speedup over hfuse: ", sum(speedup_hfuse_load) / len(speedup_hfuse_load))

speedup = [
    speedup_fi_serial,
    speedup_fa_stream,
    # speedup_fi_batchprefill,
    speedup_pod_attn,
    speedup_fa_hfuse,
    speedup_load_attn
]

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

# Create the violin plot
plt.figure(figsize=(18, 6))
#fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
axs = ['a']
axs[0] = plt.subplot(111)
# plot violin plot
vp = axs[0].violinplot(speedup,
                       widths=0.7,
                       showmeans=False,
                       showmedians=True)
#axs[0].set_title('Violin plot')

# plot box plot
#axs[1].boxplot(speedup)
#axs[1].set_title('Box plot')

lines = ['cbars', 'cmaxes', 'cmedians', 'cmins']

font_size = 40

# adding horizontal grid lines
for ax in axs:
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(speedup))],
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
    plt.plot(x_vals, y_vals, '-', color='black', linewidth=2.5)

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:.1f}'.format(x) for x in vals],
                       fontsize=26)
    
    #ax.set_xlabel('')
    ax.set_ylabel('Normlized Speedup\n over FlashAttention', fontsize=font_size-2, fontweight='bold')
    for it, title in enumerate(schemes):
        vp['bodies'][it].set_facecolor(colors[title])
        vp['bodies'][it].set_alpha(0.75)
        #vp['bodies'][it].set_label(labels[title])

# Show the plot
#plt.show()
# plt.savefig(output_path + ".pdf", bbox_inches='tight', pad_inches=1)
# plt.savefig(
#     os.path.join(output_path + ".svg"),
#     format="svg",
#     bbox_inches="tight",
#     dpi=300,            
#     transparent=True    
# )