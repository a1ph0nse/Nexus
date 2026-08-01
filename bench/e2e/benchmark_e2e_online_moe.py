import subprocess
import sys
import os
import utils


# configurable
num_requests = 256
gpu_mem_util = 0.9
max_batch_size = 999
max_token_in_batch = 65536
max_tokens = 32768
min_tokens = 2048

models = utils.models
models = ['mixtral-8x7b-1']
attention_backends = [
    'fa_sparse',
    'fa_pod_sparse',
    'la_sparse',
    'hfuse_sparse',
]
# qps_list = [0.85, 0.95]
# qps_list = [0.75, 0.85]
# qps_list = [0.1, 0.85]
# qps_list = [0.65, 0.7, 0.75, 0.8]
qps_list = [0.5, 0.55, 0.6, 0.85, 0.9, 1.0, 1.1, 1.2]

# fixed
src, root, main = utils.get_paths()

experiment_dir = utils.online_experiment_dir
trace_file = os.path.join(src, "traces", "arxiv_sample.csv")

# for quick testing
if utils.args.test == True:
    models = {'mixtral-8x7b-1'}
    num_requests = 8

for model in models:
    model_logentry = utils.models[model]['logentry']
    tp_dim = utils.models[model]['tp']
    # pd_ratio = utils.models[model]['pd_ratio']
    pd_ratio = 16
    chunk_size = utils.models[model]['chunk_size']
    for qps in qps_list:
        for backend in attention_backends:
            kv_block_size = 256 if backend != "hfuse_sparse" else 2 * 1024 * 1024
            attn_backend_arg = utils.get_backend(backend)

            command = [
                'python', main,
                    '--model_name', utils.models[model]['hfrecord'],
                    '--model_tensor_parallel_degree', f'{tp_dim}',
                    '--uniform_request_length_generator_prefill_to_decode_ratio', str(pd_ratio),
                    '--request_generator_provider', 'synthetic',
                    '--synthetic_request_generator_interval_provider', 'poisson', #'static',
                    '--poisson_request_interval_generator_qps', str(qps),
                    '--replica_scheduler_provider', 'sarathi',
                    '--sarathi_scheduler_chunk_size', str(chunk_size),
                    '--vllm_scheduler_max_tokens_in_batch', str(max_token_in_batch),
                    '--model_max_model_len', str(max_token_in_batch),
                    '--metrics_store_enable_op_level_metrics', 'false',
                    '--metrics_store_keep_individual_batch_metrics', 'true',
                    '--output_dir', f'{experiment_dir}_moe/model_{model}_attn_{backend}_qps_{qps}_rqs_{num_requests}',
                    '--synthetic_request_generator_num_requests', str(num_requests),
                    '--trace_request_length_generator_max_tokens', str(max_tokens),
                    '--trace_request_length_generator_min_tokens', str(min_tokens),
                    '--trace_request_length_generator_prefill_scale_factor', '1',
                    '--trace_request_length_generator_trace_file', str(trace_file),
                    '--model_block_size', str(kv_block_size),
                    '--model_attention_backend', str(attn_backend_arg),
                    '--gpu_memory_utilization', str(gpu_mem_util),
                ]
            # assert dataset_name in dataset_path
            print("Running command:", " ".join(command))

            try:
                subprocess.run(command, check=True)
            except:
                continue
