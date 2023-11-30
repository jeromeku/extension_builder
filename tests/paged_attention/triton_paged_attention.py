import random
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.compiler import CompiledKernel


@triton.jit
def paged_attention(
    output,
    query,
    key_cache,
    value_cache,
    head_mapping,
    block_tables,
    context_lens,
    medusa_attn_mask,
    ################### DEBUG ONLY #########################
    # Q, K,
    # QK,
    # P,
    # V,
    # ATTN_MASK,
    # stride_q1, stride_q2, stride_k1, stride_k2,
    # stride_qk1, stride_qk2,
    # stride_p1, stride_p2,
    # stride_v1, stride_v2,
    # stride_am_1, stride_am_2,
    ################### DEBUG ONLY #########################
    scale,
    num_candidates,
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kxd,
    stride_kbs,
    stride_kx,
    stride_vb,
    stride_vh,
    stride_vd,
    stride_vbs,
    stride_bts,
    stride_btb,
    stride_mm_row,
    stride_mm_col,
    key_block_index,
    key_head_index,
    value_block_index,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGES_PER_BLOCK_N: tl.constexpr,
    BLOCK_DKEYCACHE: tl.constexpr,
    X: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_batch_seq_len = tl.load(context_lens + cur_seq)
    prompt_len = cur_batch_seq_len - num_candidates
    block_start_loc = BLOCK_M * start_m

    max_block_pages = (cur_batch_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    # initialize offsets
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_x = tl.arange(0, X)
    offs_page = tl.arange(0, BLOCK_N)

    off_q = (
        cur_seq * stride_qb
        + offs_m[:, None] * stride_qs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(query + off_q, mask=offs_m[:, None] < num_candidates, other=0.0)

    ################### DEBUG ONLY #########################
    # tl.debug_barrier()
    # tl.store(Q + (offs_m[:, None] * stride_q1 + offs_d[None, :] * stride_q2), q)
    ################### DEBUG ONLY #########################

    # Supports MQA/GQA
    cur_kv_head = tl.load(head_mapping + cur_head)

    # offs for one KV page
    # later in the loop, move the offsets to different pages according to block_table
    #
    # note:
    # 1. the shape of K and V differs, K is in [num_blocks, num_heads, head_size // x, block_size, x]
    #                            while V is in [num_blocks, num_heads, head_size, block_size]
    # 2. for each block, the target K tensor should be [head_size, block_size]
    #                    the target V tensor should be [block_size, head_size]
    #    so we need to reformat K and transpose V on the fly
    offs_kv_blocks = tl.arange(0, BLOCK_SIZE)

    # use global memory to build head offs for K cache
    for subhead_idx in range(0, BLOCK_DKEYCACHE):
        current_subhead_idx = (
            tl.full([X], subhead_idx * stride_kxd, tl.int64) + offs_x * stride_kx
        )
        tl.store(key_head_index + subhead_idx * X + offs_x, current_subhead_idx)
    # trigger memory fence before direct load
    tl.debug_barrier()

    key_head_offs = tl.load(key_head_index + offs_d)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < num_candidates, 1, 0)
    # process block by block
    for start_n in range(0, block_mask * cur_batch_seq_len, BLOCK_N):
        # inform compiler for optimization
        start_n = tl.multiple_of(start_n, BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # -- compute qk of a single page ----
        start_page_idx = start_n // BLOCK_SIZE

        # use global memory to build key block offs
        for page_idx in range(0, PAGES_PER_BLOCK_N):
            block_idx = tl.load(
                block_tables
                + cur_seq * stride_bts
                + (start_page_idx + page_idx) * stride_btb,
                mask=start_page_idx + page_idx < max_block_pages,
                other=0,
            )
            current_block_idx = (
                tl.full([BLOCK_SIZE], block_idx, tl.int64) * stride_kb
                + offs_kv_blocks * stride_kbs
            )
            tl.store(
                key_block_index + page_idx * BLOCK_SIZE + offs_kv_blocks,
                current_block_idx,
            )
        # trigger memory fence before direct load
        tl.debug_barrier()
        key_block_offs = tl.load(key_block_index + offs_page)

        k_offs = (
            cur_kv_head * stride_kh + key_head_offs[:, None] + key_block_offs[None, :]
        )
        k = tl.load(
            key_cache + k_offs,
            mask=(start_n + offs_page[None, :]) < cur_batch_seq_len,
            other=0.0,
        )
        tl.debug_barrier()

        ################### DEBUG ONLY #########################
        # tl.store(K + (offs_d[:, None] * stride_k1 + offs_page[None, :] * stride_k2), k)
        # tl.debug_barrier()
        ################### DEBUG ONLY #########################

        qk += tl.dot(q, k, allow_tf32=False)
        qk *= scale
        # FIXME(sunpeng17): assume num_candidates < BLOCK_M

        attn_offs_n = start_n + offs_page - prompt_len
        attn_mask = tl.load(
            medusa_attn_mask
            + (offs_m[:, None]) * stride_mm_row
            + attn_offs_n[None, :] * stride_mm_col,
            mask=(offs_m[:, None] < num_candidates)
            & (attn_offs_n[None, :] < num_candidates)
            & (attn_offs_n[None, :] >= 0),
            other=0,
        )
        tl.debug_barrier()

        ################### DEBUG ONLY #########################
        # tl.store(ATTN_MASK + offs_m[:, None] * stride_am_1 + offs_page[None, :] * stride_am_2, attn_mask)
        ################### DEBUG ONLY #########################

        qk = tl.where(
            (start_n + offs_page[None, :] < prompt_len) | (attn_mask > 0),
            qk,
            float("-inf"),
        )

        ################### DEBUG ONLY #########################
        # tl.debug_barrier()
        # tl.store(QK + (offs_m[:, None] * stride_qk1 + offs_page[None, :] * stride_qk2), qk)
        ################### DEBUG ONLY #########################

        # compute online softmax for a single row
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc_scale = tl.where(tl.math.isnan(acc_scale), 1.0, acc_scale)
        acc = acc * acc_scale[:, None]

        for page_idx in range(0, PAGES_PER_BLOCK_N):
            block_idx = tl.load(
                block_tables
                + cur_seq * stride_bts
                + (start_page_idx + page_idx) * stride_btb,
                mask=start_page_idx + page_idx < max_block_pages,
                other=0,
            )
            current_block_idx = (
                tl.full([BLOCK_SIZE], block_idx, tl.int64) * stride_vb
                + offs_kv_blocks * stride_vbs
            )
            tl.store(
                value_block_index + page_idx * BLOCK_SIZE + offs_kv_blocks,
                current_block_idx,
                mask=(page_idx * BLOCK_SIZE + offs_kv_blocks) < cur_batch_seq_len,
            )

        # trigger memory fence before direct load
        tl.debug_barrier()
        value_block_offs = tl.load(value_block_index + offs_page)

        v_offs = (
            cur_kv_head * stride_vh
            + value_block_offs[:, None]
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(
            value_cache + v_offs,
            mask=(start_n + offs_page[:, None]) < cur_batch_seq_len,
            other=0.0,
        )

        p = tl.where(tl.math.isnan(p), 0.0, p)
        p = p.to(v.dtype)

        ################### DEBUG ONLY #########################
        # tl.debug_barrier()
        # tl.store(P + (offs_m[:, None] * stride_p1 + offs_page[None, :] * stride_p2), p)
        # tl.store(V + (offs_page[:, None] * stride_v1 + offs_d[None, :] * stride_v2), v)
        ################### DEBUG ONLY #########################

        acc += tl.dot(p, v, allow_tf32=False)
        # update m_i and l_i
        l_i = l_i_new
        l_i = tl.where(tl.math.isnan(l_i), 0.0, l_i)
        m_i = m_i_new
        m_i = tl.where(tl.math.isnan(m_i), 0.0, m_i)
    # initialize pointers to output
    off_o = (
        cur_seq * stride_ob
        + offs_m[:, None] * stride_os
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(output + off_o, acc, mask=offs_m[:, None] < num_candidates)
    return


key_block_idx = torch.zeros([128], dtype=torch.int64, device="cuda")
key_head_idx = torch.zeros([128], dtype=torch.int64, device="cuda")
value_block_idx = torch.zeros([128], dtype=torch.int64, device="cuda")

cached_bin = None


@torch.no_grad()
def paged_flash_attention_fwd(
    output: torch.Tensor,  # [num_seqs, num_candidates, num_heads, head_dim]
    query: torch.Tensor,  # [num_seqs, num_candidates, num_heads, head_dim]
    key_cache: torch.Tensor,  # [num_blocks, num_heads, head_size // x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_heads, head_size, block_size]
    head_mapping: torch.Tensor,  # [num_heads]
    scale: float,
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    context_lens: torch.Tensor,  # [num_seqs]
    block_size: int,
    max_context_len: int,
    alibi_slops: Optional[torch.Tensor],
    medusa_attn_mask: Optional[torch.Tensor],  # [num_candidates, num_candidates]
):
    global key_block_idx, key_head_idx, value_block_idx
    # does not support alibi for now
    assert alibi_slops is None

    global cached_bin

    # query = query.expand(1, -1, -1, -1)
    # output = output.expand(1, -1, -1, -1)
    # medusa_attn_mask = medusa_attn_mask[0][0]

    BLOCK_M = 16
    BLOCK_N = 128
    assert BLOCK_N % block_size == 0

    # FIXME(sunpeng17): avoid hardcode
    X = 16 // torch.tensor([], dtype=query.dtype).element_size()

    # batch, num_candidates, num_heads, head_dim = query.shape
    batch, num_candidates, num_heads, head_dim = query.shape
    # batch = 1
    head_dim = query.shape[-1]

    ################### DEBUG ONLY #########################
    # q = torch.zeros([BLOCK_M, head_dim], dtype=query.dtype, device=query.device)
    # k = torch.zeros([head_dim, BLOCK_N], dtype=query.dtype, device=query.device)
    # qk = torch.zeros([BLOCK_M, BLOCK_N], dtype=query.dtype, device=query.device)
    # p = torch.zeros([BLOCK_M, BLOCK_N], dtype=query.dtype, device=query.device)
    # v = torch.zeros([BLOCK_N, head_dim], dtype=query.dtype, device=query.device)
    # am = torch.zeros([BLOCK_M, BLOCK_N], dtype=query.dtype, device=query.device)
    ################### DEBUG ONLY #########################

    assert num_candidates <= BLOCK_M  # some code in the kernel assumes only 1 BLOCK_M
    assert head_dim % X == 0

    # batch, head, q block
    # normally we don't need to partition q because num_candidates are typically small
    # but we keep the code here for future extension
    #
    # a block in the grid processes a BLOCK_M sequences
    #
    # TODO(sunpeng17): possible future parallel alone K and V blocks?
    grid = (batch, num_heads, triton.cdiv(num_candidates, BLOCK_M))

    # if cached_bin is not None:
    #     bin = cached_bin
    #     stream = torch.cuda.current_stream().cuda_stream
    #     args = [
    #         output,
    #         query,
    #         key_cache,
    #         value_cache,
    #         head_mapping,
    #         block_tables,
    #         context_lens,
    #         medusa_attn_mask,
    #         scale,
    #         num_candidates,
    #         output.stride(0),
    #         output.stride(0),
    #         output.stride(1),
    #         output.stride(2),
    #         query.stride(0),
    #         query.stride(0),
    #         query.stride(1),
    #         query.stride(2),
    #         key_cache.stride(0),
    #         key_cache.stride(1),
    #         key_cache.stride(2),
    #         key_cache.stride(3),
    #         key_cache.stride(4),
    #         value_cache.stride(0),
    #         value_cache.stride(1),
    #         value_cache.stride(2),
    #         value_cache.stride(3),
    #         block_tables.stride(0),
    #         block_tables.stride(1),
    #         medusa_attn_mask.stride(0),  # changed from 2
    #         medusa_attn_mask.stride(1),  # changed from 3
    #         key_block_idx,
    #         key_head_idx,
    #         value_block_idx,
    #     ]
    #     bin.c_wrapper(
    #         grid[0],
    #         grid[1],
    #         grid[2],
    #         bin.num_warps,
    #         bin.num_ctas,
    #         bin.clusterDims[0],
    #         bin.clusterDims[1],
    #         bin.clusterDims[2],
    #         bin.shared,
    #         stream,
    #         bin.cu_function,
    #         CompiledKernel.launch_enter_hook,
    #         CompiledKernel.launch_exit_hook,
    #         bin,
    #         *args
    #     )
    #     return

    # print(f'*{output.dtype}, *{query.dtype}, *{key_cache.dtype}, *{value_cache.dtype}, '
    #       f'*{head_mapping.dtype}, *{block_tables.dtype}, *{context_lens.dtype}, *{medusa_attn_mask.dtype}, '
    #       f'{str(type(scale))}, {str(type(num_candidates))}, {",".join([str(type(output.stride(0)))] * 4)}, '
    #       f'{",".join([str(type(query.stride(0)))] * 4)}, '
    #       f'{",".join([str(type(key_cache.stride(0)))] * 5)}, '
    #       f'{",".join([str(type(value_cache.stride(0)))] * 4)}, '
    #       f'{",".join([str(type(block_tables.stride(0)))] * 2)}, '
    #       f'{",".join([str(type(medusa_attn_mask.stride(0)))] * 2)}, '
    #       f'*{key_block_idx.dtype}, *{key_head_idx.dtype}, *{value_block_idx.dtype}, '
    #       f'{BLOCK_M}, {head_dim}, {BLOCK_N}, {block_size}, {BLOCK_N // block_size}, {head_dim // X}, {X}'
    #       )

    num_warps = 4 if head_dim <= 64 else 8
    cached_bin = paged_attention[grid](
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        block_tables,
        context_lens,
        medusa_attn_mask,
        ################### DEBUG ONLY #########################
        # q, k,
        # qk,
        # p,
        # v,
        # am,
        # q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        # qk.stride(0), qk.stride(1),
        # p.stride(0), p.stride(1),
        # v.stride(0), v.stride(1),
        # am.stride(0), am.stride(1),
        ################### DEBUG ONLY #########################
        scale,
        num_candidates,
        output.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        query.stride(0),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        key_cache.stride(4),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        medusa_attn_mask.stride(0),
        medusa_attn_mask.stride(1),
        key_block_idx,
        key_head_idx,
        value_block_idx,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=head_dim,
        BLOCK_N=BLOCK_N,
        BLOCK_SIZE=block_size,
        PAGES_PER_BLOCK_N=BLOCK_N // block_size,  # asserted divisible
        BLOCK_DKEYCACHE=head_dim // X,  # asserted divisible
        X=X,
        num_warps=num_warps,
        num_stages=1,
    )

    ################### DEBUG ONLY #########################
    # torch.set_printoptions(profile="full")
    # torch.set_printoptions(precision=10)
    # print(f'after attn : q: {q}, shape: {q.shape}')
    # print(f'after attn : k: {k}, shape: {k.shape}')
    # print(f'after attn : qk: {qk}, shape: {qk.shape}')
    # print(f'after attn : p: {p}, shape: {p.shape}')
    # print(f'after attn : v: {v}, shape: {v.shape}')
    # print(f'after attn : medusa mask : {medusa_attn_mask}, shape: {medusa_attn_mask.shape}')
    # print(f'after attn : attn mask : {am}, shape: {am.shape}')
    # torch.set_printoptions(profile="default") # reset
    ################### DEBUG ONLY #########################

    # with nvtx.annotate("final output change", color="yellow"):
    #     output = output.squeeze(0)

    return
