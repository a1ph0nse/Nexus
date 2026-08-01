import subprocess
import sys
import os
import utils


# configurable
# num_requests = 1024
gpu_mem_util = 0.95
max_batch_size = 999
max_tokens = 16384

models = utils.models
# models = ['mistral-7b-1']
models = ['mixtral-8x7b-1']
attention_backends = [
    'fa_sparse',
    'fa_pod_sparse',
    'la_sparse',
    'hfuse_sparse',
]
# num_req_list = [256, 512]
# pd_ratio_list = [24, 16, 32]

num_req_list = [256, 512]
pd_ratio_list = [32, 64]

# fixed
src, root, main = utils.get_paths()

experiment_dir = utils.offline_experiment_dir

# for quick testing
if utils.args.test == True:
    models = ['mixtral-8x7b-1']
    num_req_list = [8]
    pd_ratio_list = [32]

for num_requests in num_req_list:
    for pd_ratio in pd_ratio_list:
        for model in models:
            model_logentry = utils.models[model]['logentry']
            # if model_logentry != 'llama-2-7b':
            #     num_requests = 256
            # num_requests = utils.models[model]['num_req']
            tp_dim = utils.models[model]['tp']
            # pd_ratio = utils.models[model]['pd_ratio']
            chunk_size = utils.models[model]['chunk_size']
            for backend in attention_backends:
                # kv_block_size = utils.get_block_or_page_size(backend)
                kv_block_size = 256 if backend != "hfuse_sparse" else 2 * 1024 * 1024
                attn_backend_arg = utils.get_backend(backend)
                command = [
                    'python', main,
                        '--model_name', utils.models[model]['hfrecord'],
                        '--model_tensor_parallel_degree', f'{tp_dim}',
                        '--request_generator_provider', 'synthetic',
                        '--synthetic_request_generator_length_provider', 'uniform',
                        '--synthetic_request_generator_interval_provider', 'static', #'static',
                        '--uniform_request_length_generator_max_tokens', str(max_tokens),
                        '--uniform_request_length_generator_min_tokens', str(max_tokens),
                        '--uniform_request_length_generator_prefill_to_decode_ratio', str(pd_ratio),
                        '--trace_request_length_generator_prefill_scale_factor', '1',
                        '--trace_request_length_generator_decode_scale_factor', '1',
                        # '--replica_scheduler_provider', 'vllm',
                        '--replica_scheduler_provider', 'sarathi',
                        '--sarathi_scheduler_chunk_size', str(chunk_size),
                        '--replica_scheduler_max_batch_size', str(max_batch_size),
                        '--vllm_scheduler_max_tokens_in_batch', str(max_tokens),
                        '--model_max_model_len', str(max_tokens),
                        '--metrics_store_enable_op_level_metrics', 'false',
                        '--metrics_store_keep_individual_batch_metrics', 'true',
                        '--output_dir', f'{experiment_dir}_moe/model_{model}_attn_{backend}_pdr_{pd_ratio}_rqs_{num_requests}_cl_{max_tokens}',
                        '--synthetic_request_generator_num_requests', str(num_requests),
                        '--trace_request_generator_max_tokens', str(max_tokens),
                        '--model_block_size', f'{kv_block_size}',
                        '--model_attention_backend', f'{attn_backend_arg}',
                        '--gpu_memory_utilization', f'{gpu_mem_util}',
                    ]
                # assert dataset_name in dataset_path
                print("Running command:", " ".join(command))

                try:
                    subprocess.run(command, check=True)
                except:
                    continue
