import triton
import triton.language as tl


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
