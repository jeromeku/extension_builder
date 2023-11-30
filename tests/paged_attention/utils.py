import random
from typing import Optional

import torch
from vllm.utils import get_max_shared_memory_bytes

from extension_builder.builders.paged_attention import PagedAttentionBuilder


def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str = "cuda",
):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=dtype, device=device)
        key_cache.uniform_(-scale, scale)
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape, dtype=dtype, device=device)
        value_cache.uniform_(-scale, scale)
        value_caches.append(value_cache)
    return key_caches, value_caches


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]
    block_tables = block_tables.cpu().tolist()
    context_lens = context_lens.cpu().tolist()

    outputs = []
    key_results = []
    val_results = []

    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(context_len, device="cuda").int()
            alibi_bias = (position_ids - context_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)
        outputs.append(out)
        key_results.append(keys)
        val_results.append(values)

    return q, key_results, val_results, outputs


from dataclasses import dataclass


@dataclass
class PagedAttentionArgs:
    query: torch.Tensor
    num_queries_per_kv: int
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    scale: float
    alibi_slopes: Optional[torch.Tensor] = None


def prepare_attention_inputs(
    num_seqs,
    num_heads,
    head_size,
    use_alibi,
    block_size,
    dtype,
    seed,
    MAX_SEQ_LEN,
    NUM_BLOCKS,
    device="cuda",
):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(
        num_seqs, num_query_heads, head_size, dtype=dtype, device=device
    )
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"), num_queries_per_kv
    )
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float, device=device)

    context_lens = torch.randint(
        1, MAX_SEQ_LEN, (num_seqs,), dtype=torch.int32, device=device
    )

    # context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    # context_lens[-1] = MAX_SEQ_LEN
    max_context_len = max(context_lens)
    # context_lens = torch.tensor(context_lens, dtype=torch.int, device=device)

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    # block_tables = []
    # for _ in range(num_seqs):
    #     block_table = [
    #         random.randint(0, NUM_BLOCKS - 1) for _ in range(max_num_blocks_per_seq)
    #     ]
    #     block_tables.append(block_table)
    # block_tables = torch.tensor(block_tables, dtype=torch.int, device=device)
    block_tables = torch.randint(
        0,
        NUM_BLOCKS - 1,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int,
        device=device,
    )
    # Create the KV caches.
    key_caches, value_caches = create_kv_caches(
        NUM_BLOCKS, block_size, 1, num_kv_heads, head_size, dtype, seed, device=device
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    return PagedAttentionArgs(
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        scale,
        alibi_slopes,
    )


def run_test(
    num_seqs,
    num_heads,
    head_size,
    use_alibi,
    block_size,
    dtype,
    seed,
    MAX_SEQ_LEN,
    NUM_BLOCKS,
    device="cuda",
):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(
        num_seqs, num_query_heads, head_size, dtype=dtype, device=device
    )
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"), num_queries_per_kv
    )
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float, device=device)

    context_lens = torch.randint(
        1, MAX_SEQ_LEN, (num_seqs,), dtype=torch.int32, device=device
    )

    # context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    # context_lens[-1] = MAX_SEQ_LEN
    max_context_len = max(context_lens)
    # context_lens = torch.tensor(context_lens, dtype=torch.int, device=device)

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    # block_tables = []
    # for _ in range(num_seqs):
    #     block_table = [
    #         random.randint(0, NUM_BLOCKS - 1) for _ in range(max_num_blocks_per_seq)
    #     ]
    #     block_tables.append(block_table)
    # block_tables = torch.tensor(block_tables, dtype=torch.int, device=device)
    block_tables = torch.randint(
        0,
        NUM_BLOCKS - 1,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int,
        device=device,
    )
    # Create the KV caches.
    key_caches, value_caches = create_kv_caches(
        NUM_BLOCKS, block_size, 1, num_kv_heads, head_size, dtype, seed, device=device
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Call the paged attention kernel.
    output = torch.empty_like(query)
    q, keys, values, out = ref_single_query_cached_kv_attention(
        output,
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        scale,
        alibi_slopes,
    )
    return q, keys, values, out
