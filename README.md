# Nexus
Nexus: Load-Balanced Kernel Fusion for GPUs with Hierarchical Profiling

## Setup

1. Download the Nexus source code
2. Create virtual environment through anaconda
```
conda create --name NexusEnv python=3.12
conda activate NexusEnv
```
3. Download [flashinfer v0.5.2](https://github.com/flashinfer-ai/flashinfer) and apply Nexus patch before compilation
4. for comparison against baselines, you need to install [flash-attention](https://github.com/Dao-AILab/flash-attention) and [pod-attention](https://github.com/microsoft/vattention/tree/main/pod_attn)
5. for end-to-end evaluation, you need to install [sarathi-serve](https://github.com/microsoft/sarathi-serve) and [vattention](https://github.com/microsoft/vattention) in this repo

## Reproducing Results
```
cd flashinfer
git am ../Nexus/Nexus.patch
cd ../Nexus
# operator evaluation
python bench/e2e/benchmark_op_perf.py
python bench/e2e/benchmark_op_utilization.py
# end-to-end evaluation
python bench/e2e/benchmark_e2e_offline.py
python bench/e2e/benchmark_e2e_online.py
...
```