import subprocess
import os
import sys
import argparse
import utils

if __name__ == "__main__":
    methods = {
        "load_attn": "LoadAttnKernelTemplate",
        # "fi_bpf": "BatchPrefillWithPagedKVCacheKernel",
        # "pod_attn": "true_fused_tb_fwd_kernel",
        # "hfuse": "regex:flash_fwd_.*fused_kernel_hfuse_idx_0",
        }

    models = [
        # MHA
        # 'llama-7b-tp1',
        # 'llama-7b-tp2',
        # 'llama-7b-tp4',
        # 'llama-13b-tp1',
        # 'llama-13b-tp2',
        # GQA
        # 'llama-3-8b-tp1',
        # 'llama-3-8b-tp2',
        # 'yi-6b-tp1',
        # 'yi-6b-tp2',
        # 'yi-34b-tp1',
        # 'yi-34b-tp2',
        # MQA
        # 'PaLM-7b-tp1',
        # 'chatglm2-6b-tp1',
        # 'chatglm2-6b-tp2',
        # 'yi-6b-tp4',
        # 'llama-3-8b-tp8',
        'progen2-tp1'
    ]

    src, root = utils.get_paths()
    run_op_path = os.path.join(src, 'run_op.py')
    util_path = os.path.join(root, 'perf', 'op_util')

    for model_name in models:
        for method_name in methods:
            kernel_name = methods[method_name]
            cmd = [
                "ncu",
                "--set", "full",
                "-f",
                "--kernel-name", kernel_name,
                "--page=raw",
                "--csv",
                "python", run_op_path, method_name, model_name,
            ]
            print("Running command:", " ".join(cmd))

            try:
                out_path = os.path.join(util_path, f"op_util_model_{model_name}_method_{method_name}.csv")
                with open(out_path, "w") as f:
                    subprocess.run(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            except:
                continue