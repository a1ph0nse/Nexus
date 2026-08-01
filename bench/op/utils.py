import torch
import os
import math
import numpy as np

import flash_attn
import flashinfer as fi
import pod_attn

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

src = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(src))

def get_paths():
    return src, root

def launch_big_kernel():
    m, n, k = 48000, 48000, 48000
    a = torch.randn(m, k, device='cuda', dtype=torch.float16)
    b = torch.randn(k, n, device='cuda', dtype=torch.float16)
    c = torch.matmul(a, b)
    return

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
warmup = 50
repeat = 200
device = 'cuda:0'
dtype = torch.float16
calc_latency = lambda start, end, steps : round(start.elapsed_time(end) / steps, 3)

@torch.inference_mode
def generate_random_shape(cs, bs=256, stride=16, sparsity=0.05, len_min=1000, len_max=11000):
    full_kv_len = np.random.randint(len_min, len_max, size=bs)
    seq_len = []
    for i in range(bs):
        if i % stride == 0:
            kv_len, qo_len = int(full_kv_len[i]), cs
        else:
            kv_len, qo_len = int(full_kv_len[i] * sparsity), 1
        seq_len.append((kv_len, qo_len))    
    return seq_len

@torch.inference_mode
def generate_page_input(seqlen_configs, num_head, num_kv_heads, head_size, block_size):
    # seqlen_config = [(kv_len, qo_len), ...]

    kv_lens_p = []
    qo_lens_p = []
    
    kv_lens_d = []
    qo_lens_d = []

    chunk_equal = True
    chunk_size = None

    for seqlen_config in seqlen_configs:
        if seqlen_config[1] == 1:
            kv_lens_d.append(seqlen_config[0])
            qo_lens_d.append(seqlen_config[1])
        else:
            kv_lens_p.append(seqlen_config[0])
            qo_lens_p.append(seqlen_config[1])
            if chunk_size is None:
                chunk_size = seqlen_config[1]
            elif chunk_size != seqlen_config[1]:
                chunk_equal = False
    
    kv_lens_p = torch.tensor(kv_lens_p, dtype=torch.int32)
    qo_lens_p = torch.tensor(qo_lens_p, dtype=torch.int32)
    kv_lens_d = torch.tensor(kv_lens_d, dtype=torch.int32)
    qo_lens_d = torch.tensor(qo_lens_d, dtype=torch.int32)

    # prefill
    bs_q = qo_lens_p.shape[0]
    
    kv_lens_blocks_p = torch.ceil(kv_lens_p / block_size).int()
    kv_indptr_p = torch.cat(
        [torch.tensor([0]), torch.cumsum(kv_lens_blocks_p, 0)], dim=0
    ).int().cuda()

    qo_indptr_p = torch.cat(
        [torch.tensor([0]), torch.cumsum(qo_lens_p, 0)], dim=0
    ).int().cuda()

    num_blocks_p = kv_indptr_p[-1].item()
    block_table_p = torch.arange(num_blocks_p).int().cuda()
    last_page_len_p = (kv_lens_p - 1) % block_size + 1

    q_p = None
    if chunk_equal:
        q_p = torch.randn(bs_q, chunk_size, num_head, head_size, device='cuda', dtype=torch.float16)
    else:
        q_p = torch.randn(qo_indptr_p[-1].item(), num_head, head_size, device='cuda', dtype=torch.float16)

    k_cache_p = torch.randn(num_blocks_p, block_size, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
    v_cache_p = torch.randn(num_blocks_p, block_size, num_kv_heads, head_size, device='cuda', dtype=torch.float16)

    # decode
    bs_d = qo_lens_d.shape[0]
    kv_lens_blocks_d = torch.ceil(kv_lens_d / block_size).int()
    kv_indptr_d = torch.cat(
        [torch.tensor([0]), torch.cumsum(kv_lens_blocks_d, 0)], dim=0
    ).int().cuda()

    qo_indptr_d = torch.cat(
        [torch.tensor([0]), torch.cumsum(qo_lens_d, 0)], dim=0
    ).int().cuda()

    num_blocks_d = kv_indptr_d[-1].item()
    block_table_d = torch.arange(num_blocks_d).int().cuda()
    last_page_len_d = (kv_lens_d - 1) % block_size + 1

    q_d = torch.randn(bs_d, 1, num_head, head_size, device='cuda', dtype=torch.float16)
    k_cache_d = torch.randn(num_blocks_d, block_size, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
    v_cache_d = torch.randn(num_blocks_d, block_size, num_kv_heads, head_size, device='cuda', dtype=torch.float16)

    return (q_p, k_cache_p, v_cache_p, qo_indptr_p, kv_indptr_p, block_table_p, qo_lens_p.cuda(), kv_lens_p.cuda(), last_page_len_p,
            q_d, k_cache_d, v_cache_d, qo_indptr_d, kv_indptr_d, block_table_d, qo_lens_d.cuda(), kv_lens_d.cuda(), last_page_len_d)

@torch.inference_mode
def modify_block_table(kv_indptr, block_table):
    bs = kv_indptr.shape[0] - 1
    num_blocks_list = kv_indptr[1:] - kv_indptr[:-1]
    max_num_blocks_per_bs = num_blocks_list.max().item()
    # print(num_blocks_list)
    # print(max_num_blocks_per_bs)

    block_table_sq = torch.empty((bs, max_num_blocks_per_bs), device=block_table.device, dtype=torch.int32)
    for i in range(bs):
        block_table_sq[i, :num_blocks_list[i]] = torch.arange(kv_indptr[i], kv_indptr[i] + num_blocks_list[i])
        block_table_sq[i, num_blocks_list[i]:] = block_table_sq[i, num_blocks_list[i] - 1]
    # print(block_table_sq)

    return block_table_sq

@torch.inference_mode
def merge_kv_cache(k_cache, v_cache):
    return torch.stack([k_cache, v_cache], dim=1).contiguous()

@torch.inference_mode
def merge_pd(
    q_p, kv_p,
    q_d, kv_d,
    qo_indptr_p, qo_indptr_d,
    kv_indptr_p, kv_indptr_d,
    kv_lens_p, kv_lens_d,
    last_page_len_p, last_page_len_d
):
    qo_indptr_pd = torch.cat([qo_indptr_d, qo_indptr_p[1:] + qo_indptr_d[-1]], dim=0)

    kv_indptr_pd = torch.cat([kv_indptr_d, kv_indptr_p[1:] + kv_indptr_d[-1]], dim=0)

    kv_lens_pd = torch.cat([kv_lens_d, kv_lens_p], dim=0)
    num_blocks_pd = kv_indptr_pd[-1].item()
    block_table_pd = torch.arange(num_blocks_pd, dtype=torch.int32, device='cuda')

    num_heads, head_size = q_d.shape[-2], q_d.shape[-1]
    q_pd = torch.cat([q_d.reshape(-1, num_heads, head_size), q_p.reshape(-1, num_heads, head_size)], dim=0).contiguous()

    kv_cache_pd = torch.cat([kv_d, kv_p], dim=0).contiguous()

    last_page_len_pd = torch.cat([last_page_len_d, last_page_len_p], dim=0).contiguous()

    return q_pd, kv_cache_pd, qo_indptr_pd, kv_indptr_pd, block_table_pd, kv_lens_pd, last_page_len_pd

# Pytorch Kernel
# prefill-only or train
# def torch_attn_with_paged_kvcache(
#     Q, kcache, vcache, block_table, cache_seqlens, causal=True
# ):

#     num_batch = Q.shape[0]
#     num_head_q = Q.shape[-2]
#     block_size = kcache.shape[1]
#     num_head_kv = kcache.shape[-2]
#     head_dim = kcache.shape[-1]
#     head_per_group = num_head_q // num_head_kv
#     Q = Q.reshape(num_batch, -1, num_head_q, head_dim) # [b, s, h_q, d]
#     output = torch.empty_like(Q)
#     for bi in range(num_batch):
#         n_blocks = (cache_seqlens[bi] + block_size - 1) // block_size
#         BQ = Q[bi:bi+1].transpose(1, 2).float()  # [h_q, s_q, d]
#         blk_ids = block_table[bi, : n_blocks]
#         seqlen = cache_seqlens[bi]
#         BK = (
#             kcache[blk_ids, :, :, :]
#             .reshape(-1, num_head_kv, head_dim)
#             .transpose(0, 1)[:, :seqlen, :]
#             .repeat_interleave(head_per_group, dim=0)
#         ).float().unsqueeze(0) # [1, h_kv, s_kv, d]
#         BV = (
#             vcache[blk_ids, :, :, :]
#             .reshape(-1, num_head_kv, head_dim)
#             .transpose(0, 1)[:, :seqlen, :]
#             .repeat_interleave(head_per_group, dim=0)
#         ).float().unsqueeze(0) # [1, h_kv, s_kv, d]
        
#         query_len = BQ.shape[2]  # query length
        
#         # P = BQ @ BK.transpose(-1, -2)
#         # P = P / math.sqrt(head_dim)
#         causal_mask = torch.ones(
#            query_len, seqlen - query_len, device=Q.device, dtype=torch.bool
#         )
#         tail_causal_mask = torch.tril(
#             torch.ones(query_len, query_len, device=Q.device, dtype=torch.bool)
#         )
#         causal_mask = torch.cat([causal_mask, tail_causal_mask], dim=-1).unsqueeze(0)
#         causal_mask = torch.where(causal_mask, 0.0, float("-inf")).to(BQ.dtype)

#         Y = torch.nn.functional.scaled_dot_product_attention(
#             BQ, BK, BV,
#             attn_mask=causal_mask,
#             scale=1.0 / math.sqrt(head_dim),
#             dropout_p=0.0,
#         )

#         # P = P.masked_fill(~causal_mask, float("-inf"))
#         # attn_weights = torch.nn.functional.softmax(P, dim=-1)
#         # Y = torch.matmul(attn_weights, BV)

#         output[bi] = Y.squeeze(0).transpose(0, 1)
#         # print(output)

#     return output.reshape(num_batch, -1, num_head_q, head_dim)

# prefill-only or train
def torch_attn_with_paged_kvcache(
    Q, kcache, vcache, block_table, cache_seqlens, causal=True
):

    num_batch = Q.shape[0]
    num_head_q = Q.shape[-2]
    block_size = kcache.shape[1]
    num_head_kv = kcache.shape[-2]
    head_dim = kcache.shape[-1]
    head_per_group = num_head_q // num_head_kv
    
    Q_t = Q.transpose(1, 2) # [b, h_q, s, d]
    output = torch.empty_like(Q)
    # print(output.shape)

    for bi in range(num_batch):
        n_blocks = (cache_seqlens[bi] + block_size - 1) // block_size
        BQ = Q_t[bi:bi+1]  # [1, h_q, s_q, d]
        blk_ids = block_table[bi, : n_blocks]
        seqlen = cache_seqlens[bi]
        BK = (
            kcache[blk_ids, :, :, :]
            .view(-1, num_head_kv, head_dim)
            .transpose(0, 1)[:, :seqlen, :]
            .repeat_interleave(head_per_group, dim=0)
        ).unsqueeze(0) # [1, h_kv, s_kv, d]
        BV = (
            vcache[blk_ids, :, :, :]
            .view(-1, num_head_kv, head_dim)
            .transpose(0, 1)[:, :seqlen, :]
            .repeat_interleave(head_per_group, dim=0)
        ).unsqueeze(0) # [1, h_kv, s_kv, d]
        
        query_len = BQ.shape[2]  # query length
        
        causal_mask = torch.ones(
           query_len, seqlen - query_len, device=Q.device, dtype=torch.bool
        )
        tail_causal_mask = torch.tril(
            torch.ones(query_len, query_len, device=Q.device, dtype=torch.bool)
        )
        causal_mask = torch.cat([causal_mask, tail_causal_mask], dim=-1).unsqueeze(0)
        causal_mask = torch.where(causal_mask, 0.0, float("-inf")).to(BQ.dtype)

        Y = torch.nn.functional.scaled_dot_product_attention(
            BQ, BK, BV,
            attn_mask=causal_mask,
            dropout_p=0.0,
        )

        output[bi] = Y.squeeze(0).transpose(0, 1)

    return output.reshape(num_batch, -1, num_head_q, head_dim)


@torch.inference_mode
def bench_torch_prefill(q, k, v, cache_seqlens=None, block_table=None, causal=True, profile=False):
    try:
        output = torch_attn_with_paged_kvcache(q, k, v, cache_seqlens=cache_seqlens, block_table=block_table, causal=causal)
        if profile:
            return output, 0
        else:
            for _ in range(warmup):
                output = torch_attn_with_paged_kvcache(q, k, v, cache_seqlens=cache_seqlens, block_table=block_table, causal=causal)
            torch.cuda.synchronize()
            start.record()
            for _ in range(repeat):
                output = torch_attn_with_paged_kvcache(q, k, v, cache_seqlens=cache_seqlens, block_table=block_table, causal=causal)
            end.record()
            torch.cuda.synchronize()
            if repeat == 0:
                return output, 0
            return output, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return None, -1


@torch.inference_mode
def bench_pt_prefill(q, k, v, seq_lens_k=None, causal=True):
    try:
        output = None
        attn_mask = torch.ones(q.shape[1], k.shape[1], dtype=torch.bool, device=device).tril(diagonal=0) if causal else None
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        for _ in range(warmup):
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeat):
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        end.record()
        torch.cuda.synchronize()
        if repeat == 0:
            return output, 0
        return output, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return None, -1
    
@torch.inference_mode
def bench_pt_decode(q, k, v, k_new=None, v_new=None, seq_lens_k=None, cache_batch_idx=None, splits=0):
    try:
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        for _ in range(warmup):
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeat):
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
        end.record()
        torch.cuda.synchronize()
        if repeat == 0:
            return output, 0
        return output, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return None, -1
    
# FA Based Kernel
@torch.inference_mode
def bench_fa_prefill(q, k, v, seq_lens_k=None, causal=True, window_size=(-1, -1), profile=False):
    try:
        output = flash_attn.flash_attn_with_kvcache(q, k, v, causal=causal, cache_seqlens=seq_lens_k, window_size=window_size)
        if profile:
            return output, 0
        else:
            for _ in range(warmup):
                output = flash_attn.flash_attn_with_kvcache(q, k, v, causal=causal, cache_seqlens=seq_lens_k, window_size=window_size)
            torch.cuda.synchronize()
            start.record()
            for _ in range(repeat):
                output = flash_attn.flash_attn_with_kvcache(q, k, v, causal=causal, cache_seqlens=seq_lens_k, window_size=window_size)
            end.record()
            torch.cuda.synchronize()
            if repeat == 0:
                return output, 0
            return output, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return None, -1
    
@torch.inference_mode
def bench_fa_prefill_paged(q, k, v, cache_seqlens=None, block_table=None, causal=True, window_size=(-1, -1), profile=False):
    try:
        output = flash_attn.flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens, block_table=block_table, causal=causal, window_size=window_size)
        if profile:
            return output, 0
        else:
            for _ in range(warmup):
                output = flash_attn.flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens, block_table=block_table, causal=causal, window_size=window_size)
            torch.cuda.synchronize()
            start.record()
            for _ in range(repeat):
                output = flash_attn.flash_attn_with_kvcache(q, k, v, cache_seqlens=cache_seqlens, block_table=block_table, causal=causal, window_size=window_size)
            end.record()
            torch.cuda.synchronize()
            if repeat == 0:
                return output, 0
            return output, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return None, -1


@torch.inference_mode
def bench_fa_decode(q, k, v, k_new=None, v_new=None, seq_lens_k=None, cache_batch_idx=None, splits=0, window_size=(-1, -1), profile=False):
    try:
        output = flash_attn.flash_attn_with_kvcache(q, k, v, causal=False, k=k_new, v=v_new, cache_seqlens=seq_lens_k, cache_batch_idx=cache_batch_idx, num_splits=splits, window_size=window_size)
        if profile:
            return output, 0
        else:
            for _ in range(warmup):
                output = flash_attn.flash_attn_with_kvcache(q, k, v, causal=False, k=k_new, v=v_new, cache_seqlens=seq_lens_k, cache_batch_idx=cache_batch_idx, num_splits=splits, window_size=window_size)
            torch.cuda.synchronize()
            start.record()
            for _ in range(repeat):
                output = flash_attn.flash_attn_with_kvcache(q, k, v, causal=False, k=k_new, v=v_new, cache_seqlens=seq_lens_k, cache_batch_idx=cache_batch_idx, num_splits=splits, window_size=window_size)
            end.record()
            torch.cuda.synchronize()
            if repeat == 0:
                return output, 0
            return output, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return None, -1

@torch.inference_mode
def bench_fa_decode_paged(q, k, v, k_new=None, v_new=None, cache_seqlens=None, block_table=None, splits=0, window_size=(-1, -1), profile=False):
    try:
        output = flash_attn.flash_attn_with_kvcache(q, k, v, causal=False, k=k_new, v=v_new, cache_seqlens=cache_seqlens, block_table=block_table, num_splits=splits, window_size=window_size)
        if profile:
            return output, 0
        else:
            for _ in range(warmup):
                output = flash_attn.flash_attn_with_kvcache(q, k, v, causal=False, k=k_new, v=v_new, cache_seqlens=cache_seqlens, block_table=block_table, num_splits=splits, window_size=window_size)
            torch.cuda.synchronize()
            start.record()
            for _ in range(repeat):
                output = flash_attn.flash_attn_with_kvcache(q, k, v, causal=False, k=k_new, v=v_new, cache_seqlens=cache_seqlens, block_table=block_table, num_splits=splits, window_size=window_size)
            end.record()
            torch.cuda.synchronize()
            if repeat == 0:
                return output, 0
            return output, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return None, -1

@torch.inference_mode
def bench_fa_pod(q_p, k_p, v_p, q_d, k_d, v_d, fused_params=0, k_new=None, v_new=None, seq_lens_k_p=None, seq_lens_k_d=None, cache_batch_idx=None, num_splits_p=0, num_splits_d=0, causal=True, window_size=(-1, -1), profile=False):
    try:
        output_pref, output_dec = pod_attn.true_fused_attn_with_kvcache(q_p, k_p, v_p, q_d, k_d, v_d, k=k_new, v=v_new, causal=causal, \
                cache_seqlens_p=seq_lens_k_p, cache_seqlens_d=seq_lens_k_d, cache_batch_idx=cache_batch_idx, fused_params=fused_params, num_splits_p=num_splits_p, num_splits_d=num_splits_d, window_size=window_size)
        if profile:
            return output_pref, output_dec, 0
        else:
            for _ in range(warmup):
                output_pref, output_dec = pod_attn.true_fused_attn_with_kvcache(q_p, k_p, v_p, q_d, k_d, v_d, k=k_new, v=v_new, causal=causal, \
                        cache_seqlens_p=seq_lens_k_p, cache_seqlens_d=seq_lens_k_d, cache_batch_idx=cache_batch_idx, fused_params=fused_params, num_splits_p=num_splits_p, num_splits_d=num_splits_d, window_size=window_size)
            torch.cuda.synchronize()
            start.record()
            for _ in range(repeat):
                output_pref, output_dec = pod_attn.true_fused_attn_with_kvcache(q_p, k_p, v_p, q_d, k_d, v_d, k=k_new, v=v_new, causal=causal, \
                    cache_seqlens_p=seq_lens_k_p, cache_seqlens_d=seq_lens_k_d, cache_batch_idx=cache_batch_idx, fused_params=fused_params, num_splits_p=num_splits_p, num_splits_d=num_splits_d, window_size=window_size)
            end.record()
            torch.cuda.synchronize()
            if repeat == 0:
                return output_pref, output_dec, 0
            return output_pref, output_dec, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return None, None, -1

@torch.inference_mode
def bench_fa_pod_paged(q_p, k_p, v_p, q_d, k_d, v_d, fused_params=0, k_new=None, v_new=None, cache_seqlens_p=None, cache_seqlens_d=None, block_table_p=None, block_table_d=None, num_splits_p=0, num_splits_d=0, causal=True, window_size=(-1, -1), profile=False):
    try:
        output_pref, output_dec = pod_attn.true_fused_attn_with_kvcache(q_p, k_p, v_p, q_d, k_d, v_d, k=k_new, v=v_new, causal=causal, \
                cache_seqlens_p=cache_seqlens_p, cache_seqlens_d=cache_seqlens_d,
                block_table_p=block_table_p, block_table_d=block_table_d, fused_params=fused_params, num_splits_p=num_splits_p, num_splits_d=num_splits_d, window_size=window_size)
        if profile:
            return output_pref, output_dec, 0
        else:
            for _ in range(warmup):
                output_pref, output_dec = pod_attn.true_fused_attn_with_kvcache(q_p, k_p, v_p, q_d, k_d, v_d, k=k_new, v=v_new, causal=causal, \
                        cache_seqlens_p=cache_seqlens_p, cache_seqlens_d=cache_seqlens_d,
                        block_table_p=block_table_p, block_table_d=block_table_d, fused_params=fused_params, num_splits_p=num_splits_p, num_splits_d=num_splits_d, window_size=window_size)
            torch.cuda.synchronize()
            start.record()
            for _ in range(repeat):
                output_pref, output_dec = pod_attn.true_fused_attn_with_kvcache(q_p, k_p, v_p, q_d, k_d, v_d, k=k_new, v=v_new, causal=causal, \
                    cache_seqlens_p=cache_seqlens_p, cache_seqlens_d=cache_seqlens_d,
                    block_table_p=block_table_p, block_table_d=block_table_d, fused_params=fused_params, num_splits_p=num_splits_p, num_splits_d=num_splits_d, window_size=window_size)
            end.record()
            torch.cuda.synchronize()
            if repeat == 0:
                return output_pref, output_dec, 0
            return output_pref, output_dec, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return None, None, -1
    
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()
@torch.inference_mode
def bench_fa_stream_paged(q_p, k_p, v_p, q_d, k_d, v_d, k_new=None, v_new=None, cache_seqlens_p=None, cache_seqlens_d=None, block_table_p=None, block_table_d=None, num_splits_p=0, num_splits_d=0, causal=True, window_size=(-1, -1), profile=False):
    try:
        with torch.cuda.stream(stream1):
            output_pref = flash_attn.flash_attn_with_kvcache(q_p, k_p, v_p, cache_seqlens=cache_seqlens_p, block_table=block_table_p, num_splits=num_splits_p, causal=causal, window_size=window_size)
        with torch.cuda.stream(stream2):
            output_dec = flash_attn.flash_attn_with_kvcache(q_d, k_d, v_d, causal=False, k=k_new, v=v_new, cache_seqlens=cache_seqlens_d, block_table=block_table_d, num_splits=num_splits_d, window_size=window_size)
        stream1.synchronize()
        stream2.synchronize()

        if profile:
            return output_pref, output_dec, 0
        else:
            for _ in range(warmup):
                with torch.cuda.stream(stream1):
                    output_pref = flash_attn.flash_attn_with_kvcache(q_p, k_p, v_p, cache_seqlens=cache_seqlens_p, block_table=block_table_p, num_splits=num_splits_p, causal=causal, window_size=window_size)
                with torch.cuda.stream(stream2):
                    output_dec = flash_attn.flash_attn_with_kvcache(q_d, k_d, v_d, causal=False, k=k_new, v=v_new, cache_seqlens=cache_seqlens_d, block_table=block_table_d, num_splits=num_splits_d, window_size=window_size)
                stream1.synchronize()
                stream2.synchronize()

            torch.cuda.synchronize()
            start.record()
            for _ in range(repeat):
                with torch.cuda.stream(stream1):
                    output_pref = flash_attn.flash_attn_with_kvcache(q_p, k_p, v_p, cache_seqlens=cache_seqlens_p, block_table=block_table_p, num_splits=num_splits_p, causal=causal, window_size=window_size)
                with torch.cuda.stream(stream2):
                    output_dec = flash_attn.flash_attn_with_kvcache(q_d, k_d, v_d, causal=False, k=k_new, v=v_new, cache_seqlens=cache_seqlens_d, block_table=block_table_d, num_splits=num_splits_d, window_size=window_size)
                stream1.synchronize()
                stream2.synchronize()
            end.record()

            torch.cuda.synchronize()
            if repeat == 0:
                return output_pref, output_dec, 0
            return output_pref, output_dec, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return None, None, -1

# FI Based Kernel
@torch.inference_mode
def bench_flashinfer_prefill(q, k, v, causal=True, window_left=-1):
    try:
        # assert bs == 1, "batch size must be 1 for flashinfer prefill"
        # q = torch.randn(cl, num_heads, head_dim, device=device, dtype=dtype)
        # k = torch.randn(cl, num_kv_heads, head_dim, device=device, dtype=dtype)
        # v = torch.randn(cl, num_kv_heads, head_dim, device=device, dtype=dtype)
        o = fi.single_prefill_with_kv_cache(q, k, v, causal=causal, window_left=window_left)
        for _ in range(warmup):
            fi.single_prefill_with_kv_cache(q, k, v, causal=causal, window_left=window_left)
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeat):
            fi.single_prefill_with_kv_cache(q, k, v, causal=causal, window_left=window_left)
        end.record()
        torch.cuda.synchronize()
        latency = calc_latency(start, end, repeat)
        return o, latency
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def bench_flashinfer_single_decode(q, k, v, window_left=-1):
    try:
        o = fi.single_decode_with_kv_cache(q, k, v, window_left=window_left)
        for _ in range(warmup):
            fi.single_decode_with_kv_cache(q, k, v, window_left=window_left)
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeat):
            fi.single_decode_with_kv_cache(q, k, v, window_left=window_left)
        end.record()
        torch.cuda.synchronize()
        return o, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_batch_prefill(
    q,
    kv,
    bs,
    qo_lens,
    kv_lens,
    num_qo_heads=32,
    num_kv_heads=32,
    head_dim=128,
    device=0,
    causal=True,
    page_block_size = 1,
    window_left=-1
):
    seq_lens = torch.tensor([kv_lens] * bs, dtype=torch.int32)
    q_lens = torch.tensor([qo_lens] * bs, dtype=torch.int32)

    seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()

    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()

    num_blocks = kv_indptr[-1].item()

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    kv_layout = "NHD"

    wrapper_old = fi.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout=kv_layout,
        backend="fa2",
    )
    last_page_len = (seq_lens - 1) % page_block_size + 1
    wrapper_old.plan(
        q_indptr.to(device),
        kv_indptr.to(device),
        torch.arange(num_blocks).int().to(device),
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        causal=causal,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        window_left=window_left,
    )
    o = wrapper_old.run(q, kv)
    return o, wrapper_old
    
@torch.inference_mode
def bench_flashinfer_batch_prefill(
    q,
    kv,
    bs,
    qo_lens,
    kv_lens,
    num_qo_heads=32,
    num_kv_heads=32,
    head_dim=128,
    device=0,
    causal=True,
    block_size = 1,
    window_left=-1
):
    try:
        o, wrapper_old = do_flashinfer_batch_prefill(
            q,
            kv,
            bs,
            qo_lens,
            kv_lens,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            device,
            causal,
            block_size,
            window_left
        )
        for _ in range(warmup):
            wrapper_old.run(q, kv)
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeat):
            wrapper_old.run(q, kv)
        end.record()
        torch.cuda.synchronize()
        return o, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def bench_flashinfer_batch_prefill_paged(
    q,
    kv,
    qo_indptr,
    kv_indptr,
    block_table,
    last_page_len,
    num_head=32,
    num_kv_head=32,
    head_dim=128,
    causal=True,
    block_size=1,
    window_left=-1,
    profile=False
):
    try:
        workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device='cuda')
        wrapper_old = fi.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout="NHD",
            backend="fa2",
        )
        wrapper_old.plan(
            qo_indptr,
            kv_indptr,
            block_table,
            last_page_len,
            num_head,
            num_kv_head,
            head_dim,
            block_size,
            causal=causal,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            window_left=window_left,
        )
        o = wrapper_old.run(q, kv)
        if profile:
            return o, 0
        else:
            for _ in range(warmup):
                wrapper_old.run(q, kv)
            torch.cuda.synchronize()
            start.record()
            for _ in range(repeat):
                wrapper_old.run(q, kv)
            end.record()
            torch.cuda.synchronize()
            return o, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def bench_flashinfer_decode_paged(q, kv, bs, cl, num_heads, num_kv_heads, head_dim, block_size = 16, window_left=-1):
    try:
        # q = torch.randn(bs, num_heads, head_dim, dtype=dtype, device=device)
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
        # 实际是Batch Prefill Kernel
        decode_wrapper = fi.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD", use_tensor_cores=True)
        num_pages_per_req = math.ceil(cl / block_size)
        max_num_pages = num_pages_per_req * bs
        kv_page_indices = torch.arange(max_num_pages).int().to(device)
        kv_page_indptr = torch.arange(0, bs + 1).int().to(device) * num_pages_per_req
        kv_last_page_len = torch.full((bs,), (cl  - 1) % block_size + 1, dtype=torch.int32).to(device)
        # kv = kv.reshape(max_num_pages, 2, block_size, num_kv_heads, head_dim)
        # kv = torch.randn(max_num_pages, 2, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
        decode_wrapper.plan(
            kv_page_indptr,
            kv_page_indices,
            kv_last_page_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            window_left=window_left,
        )
        o = decode_wrapper.forward(q, kv)
        for _ in range(warmup):
            decode_wrapper.forward(q, kv)
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeat):
            decode_wrapper.forward(q, kv)
        end.record()
        torch.cuda.synchronize()
        return o, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_pod(q_p, k_p, v_p, q_d, kv_d, bs, cl, num_heads, num_kv_heads, head_dim, use_tensor, block_size, causal = True, window_left=-1):
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)
    pod_wrapper = fi.PODWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    num_pages_per_req = math.ceil(cl / block_size)
    max_num_pages = num_pages_per_req * bs
    kv_page_indices = torch.arange(max_num_pages).int().to(device)
    kv_page_indptr = torch.arange(0, bs + 1).int().to(device) * num_pages_per_req
    kv_last_page_len = torch.full((bs,), (cl  - 1) % block_size + 1, dtype=torch.int32).to(device)
    # kv = kv_d.reshape(max_num_pages, 2, block_size, num_kv_heads, head_dim)
    # kv = torch.randn(max_num_pages, 2, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    pod_wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        data_type=torch.float16,
        q_data_type=torch.float16,
        window_left=window_left,
    )
    o_p, o_d = pod_wrapper.run(q_p, k_p, v_p, q_d, kv_d, 
            causal_p = causal)
    
    return o_p, o_d, pod_wrapper

@torch.inference_mode
def bench_flashinfer_pod(q_p, k_p, v_p, q_d, kv_d, bs, cl, num_heads, num_kv_heads, head_dim, use_tensor = True, block_size = 16, causal = True, window_left=-1):
    try:
        o_p, o_d, pod_wrapper = do_flashinfer_pod(q_p, k_p, v_p, q_d, kv_d, bs, cl, num_heads, num_kv_heads, head_dim, use_tensor, block_size, causal, window_left)
        for _ in range(warmup):
            pod_wrapper.run(q_p, k_p, v_p, q_d, kv_d, 
                causal_p = causal)
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeat):
            pod_wrapper.run(q_p, k_p, v_p, q_d, kv_d, 
                causal_p = causal)
        end.record()
        torch.cuda.synchronize()
        latency = calc_latency(start, end, repeat)
        return o_p, o_d, latency
    except Exception as e:
        print(e)
        return -1, -1, -1

@torch.inference_mode
def do_flashinfer_batch_pod(q_p, kv_p, q_d, kv_d, p_bs, chunk_size, p_cache_seqlen, d_bs, d_cache_seqlen, num_heads, num_kv_heads, head_dim, block_size, causal=True, window_left=-1):
    p_qo_lens = [chunk_size] * p_bs
    p_kv_lens = [p_cache_seqlen] * p_bs
    p_seq_lens_blocks = torch.ceil(
        torch.tensor(p_kv_lens, dtype=torch.int32) / block_size
    ).int()
    last_page_len_p = (p_seq_lens_blocks - 1) % block_size + 1    

    d_qo_lens = [1] * d_bs
    d_kv_lens = [d_cache_seqlen] * d_bs
    d_seq_lens_blocks = torch.ceil(
        torch.tensor(d_kv_lens, dtype=torch.int32) / block_size
    ).int()
    last_page_len_d = (d_seq_lens_blocks - 1) % block_size + 1

    q_lens = torch.tensor(d_qo_lens + p_qo_lens, dtype=torch.int32)
    seq_lens = torch.tensor(d_kv_lens + p_kv_lens, dtype=torch.int32)
    seq_lens_blocks = torch.ceil(seq_lens / block_size).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()

    p_q_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(torch.tensor(p_qo_lens), 0)], dim=0
    ).int()
    p_kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(p_seq_lens_blocks, 0)], dim=0
    ).int()

    d_q_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(torch.tensor(d_qo_lens), 0)], dim=0
    ).int()
    d_kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(d_seq_lens_blocks, 0)], dim=0
    ).int()
    num_blocks = kv_indptr[-1].item()

    kv_indices_d = torch.arange(0, d_kv_indptr[-1], device=device, dtype=torch.int32)
    kv_indices_p = torch.arange(0, p_kv_indptr[-1], device=device, dtype=torch.int32)

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper_batchpod = fi.BatchPODWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD"
        )
    wrapper_batchpod.plan(
        # Prefill params
        p_q_indptr.to(device),
        p_kv_indptr.to(device),
        kv_indices_p.to(device),
        last_page_len_p,
        # Decode params
        d_q_indptr.to(device),
        d_kv_indptr.to(device),
        kv_indices_d.to(device),
        last_page_len_d,
        # Common params
        num_qo_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=block_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        window_left=window_left,
    )

    o_p, o_d = wrapper_batchpod.run(
        q_p,
        kv_p,
        q_d,
        kv_d,
        causal_p=causal,
    )
    return o_p, o_d, wrapper_batchpod

@torch.inference_mode
def bench_flashinfer_batch_pod(q_p, kv_p, q_d, kv_d, p_bs, chunk_size, p_cache_seqlen, d_bs, d_cache_seqlen, num_heads, num_kv_heads, head_dim, block_size, causal=True, window_left=-1):
    try:
        o_p, o_d, wrapper_batchpod = do_flashinfer_batch_pod(q_p, kv_p, q_d, kv_d, p_bs, chunk_size, p_cache_seqlen, d_bs, d_cache_seqlen, num_heads, num_kv_heads, head_dim, block_size, causal, window_left)
        for _ in range(warmup):
            wrapper_batchpod.run(
                q_p,
                kv_p,
                q_d,
                kv_d,
                causal_p=causal,
            )
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeat):
            wrapper_batchpod.run(
                q_p,
                kv_p,
                q_d,
                kv_d,
                causal_p=causal,
            )
        end.record()
        torch.cuda.synchronize()
        return o_p, o_d, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_batch_attn(q, kv, q_bs, chunk_size, p_cache_seqlen, d_bs, d_cache_seqlen, num_heads, num_kv_heads, head_dim, block_size, causal=True):
    seq_lens = torch.tensor([d_cache_seqlen] * d_bs + [p_cache_seqlen] * q_bs, dtype=torch.int32)
    q_lens = torch.tensor([1] * d_bs + [chunk_size] * q_bs, dtype=torch.int32)

    seq_lens_blocks = torch.ceil(seq_lens / block_size).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()
    num_blocks = kv_indptr[-1].item()

    wrapper_persistent = fi.BatchAttention(kv_layout="NHD")
    wrapper_persistent.plan(
        q_indptr.to(device),
        kv_indptr.to(device),
        torch.arange(num_blocks, dtype=torch.int32, device=device),
        seq_lens.to(device),
        num_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        block_size,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    kv = kv.reshape(num_blocks, 2, block_size, num_kv_heads, head_dim)
    o, _ = wrapper_persistent.run(q, kv)
    return o, wrapper_persistent

@torch.inference_mode
def bench_flashinfer_batch_attn(q, kv, q_bs, chunk_size, p_cache_seqlen, d_bs, d_cache_seqlen, num_heads, num_kv_heads, head_dim, block_size, causal):
    try:
        o, wrapper_persistent = do_flashinfer_batch_attn(q, kv, q_bs, chunk_size, p_cache_seqlen, d_bs, d_cache_seqlen, num_heads, num_kv_heads, head_dim, block_size, causal)
        for _ in range(warmup):
            wrapper_persistent.run(q, kv)
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeat):
            wrapper_persistent.run(q, kv)
        end.record()
        torch.cuda.synchronize()
        return o, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def bench_flashinfer_batch_attn_paged(
    q, kv,
    qo_indptr,
    kv_indptr,
    block_table,
    kv_lens,
    num_head=32,
    num_kv_head=32,
    head_dim=128,
    causal=True,
    block_size=1,
    profile=False
    ):
    try:
        wrapper_ba = fi.BatchAttention(kv_layout="NHD")
        wrapper_ba.plan(
            qo_indptr,
            kv_indptr,
            block_table,
            kv_lens,
            num_head,
            num_kv_head,
            head_dim,
            head_dim,
            block_size,
            causal=causal,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )
        o, _ = wrapper_ba.run(q, kv)
        if profile:
            return o, 0
        else:
            for _ in range(warmup):
                wrapper_ba.run(q, kv)
            torch.cuda.synchronize()
            start.record()
            for _ in range(repeat):
                wrapper_ba.run(q, kv)
            end.record()
            torch.cuda.synchronize()
            return o, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def do_flashinfer_load_attn(q, kv, q_bs, chunk_size, p_cache_seqlen, d_bs, d_cache_seqlen, num_heads, num_kv_heads, head_dim, block_size, causal=True, disable_split=False, window_left=-1):
    seq_lens = torch.tensor([d_cache_seqlen] * d_bs + [p_cache_seqlen] * q_bs, dtype=torch.int32)
    q_lens = torch.tensor([1] * d_bs + [chunk_size] * q_bs, dtype=torch.int32)

    seq_lens_blocks = torch.ceil(seq_lens / block_size).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()
    num_blocks = kv_indptr[-1].item()

    wrapper_load = fi.LoadAttention(kv_layout="NHD")
    wrapper_load.plan(
        q_indptr.to(device),
        kv_indptr.to(device),
        torch.arange(num_blocks, dtype=torch.int32, device=device),
        seq_lens.to(device),
        num_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        block_size,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
        disable_split=disable_split,
        window_left=window_left,
    )
    o, _ = wrapper_load.run(q, kv)
    return o, wrapper_load

@torch.inference_mode
def bench_flashinfer_load_attn(q, kv, q_bs, chunk_size, p_cache_seqlen, d_bs, d_cache_seqlen, num_heads, num_kv_heads, head_dim, block_size, causal=True, disable_split=False, window_left=-1):
    try:
        o, wrapper_load = do_flashinfer_load_attn(q, kv, q_bs, chunk_size, p_cache_seqlen, d_bs, d_cache_seqlen, num_heads, num_kv_heads, head_dim, block_size, causal, disable_split, window_left)
        for _ in range(warmup):
            wrapper_load.run(q, kv)
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeat):
            wrapper_load.run(q, kv)
        end.record()
        torch.cuda.synchronize()
        return o, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return -1
    
@torch.inference_mode
def bench_flashinfer_load_attn_paged(
    q, kv,
    qo_indptr,
    kv_indptr,
    block_table,
    kv_lens,
    num_head=32,
    num_kv_head=32,
    head_dim=128,
    causal=True,
    block_size=1,
    disable_split=False,
    window_left=-1,
    profile=False
    ):
    try:
        workspace_buffer = torch.empty(2 * 1024 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper_load = fi.LoadAttention(
            workspace_buffer,
            kv_layout="NHD"
        )
        wrapper_load.plan(
            qo_indptr,
            kv_indptr,
            block_table,
            kv_lens,
            num_head,
            num_kv_head,
            head_dim,
            head_dim,
            block_size,
            causal=causal,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            disable_split=disable_split,
            window_left=window_left,
        )
        o, _ = wrapper_load.run(q, kv)
        if profile:
            return o, 0
        else:
            for _ in range(warmup):
                wrapper_load.run(q, kv)
            torch.cuda.synchronize()
            start.record()
            for _ in range(repeat):
                wrapper_load.run(q, kv)
            end.record()
            torch.cuda.synchronize()
            return o, calc_latency(start, end, repeat)
    except Exception as e:
        print(e)
        return -1

@torch.inference_mode
def bench_load_attention_cost(
    q_pd, kv_cache_pd,
    qo_indptr_pd, kv_indptr_pd, block_table_pd,
    kv_lens_pd,
    num_head=32, num_kv_head=32, head_dim=128,
    causal=True, block_size=256, disable_split=False,
    window_left=-1,
):
    """仅调用 LoadAttention.plan() 获取调度 cost 统计，不执行 run()。
    
    Returns:
        (sm_cost_min, sm_cost_max, sm_cost_avg, cta_cost_min, cta_cost_max, cta_cost_avg)
        失败时返回 (-1, -1, -1, -1, -1, -1)
    """
    try:
        workspace_buffer = torch.empty(2 * 1024 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = fi.LoadAttention(workspace_buffer, kv_layout="NHD")
        wrapper.plan(
            qo_indptr_pd,
            kv_indptr_pd,
            block_table_pd,
            kv_lens_pd,
            num_head,
            num_kv_head,
            head_dim,
            head_dim,
            block_size,
            causal=causal,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            disable_split=disable_split,
            window_left=window_left,
        )
        plan_info = wrapper._plan_info
        cost_scale = 10000.0
        return (
            plan_info[16] / cost_scale,  # sm_cost_min
            plan_info[17] / cost_scale,  # sm_cost_max
            plan_info[18] / cost_scale,  # sm_cost_avg
            plan_info[19] / cost_scale,  # cta_cost_min
            plan_info[20] / cost_scale,  # cta_cost_max
            plan_info[21] / cost_scale,  # cta_cost_avg
        )
    except Exception as e:
        print(f"[bench_load_attention_cost] Error: {e}")
        return (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
