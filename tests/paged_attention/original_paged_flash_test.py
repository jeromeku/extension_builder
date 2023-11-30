import math
import random
from typing import List, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton_paged_attention import paged_flash_attention_fwd

MAX_SEQ_LEN = 1024  # 2048
TEST_SEED = 0


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    query = query * scale
    attn = torch.einsum("qhd,khd->hqk", query, key)

    if attn_mask is not None:
        attn = attn + attn_mask
    attn = torch.softmax(attn, dim=-1)
    out = torch.einsum("hqk,khd->qhd", attn, value)

    ################### DEBUG ONLY #########################
    # torch.set_printoptions(profile="full")
    # torch.set_printoptions(precision=10)
    # print(f'[ref] q: {query}, shape: {query.shape}')
    # print(f'[ref] k: {key}, shape: {key.shape}')
    # print(f'[ref] q@k: {attn}, shape: {attn.shape}')
    # print(f'[ref] v: {value}, shape: {value.shape}')
    # print(f'[ref] q@k after softmax: {attn}')
    # torch.set_printoptions(profile="default") # reset
    ################### DEBUG ONLY #########################

    return out


# for medusa attn mask
def generate_medusa_attn_mask(medusa_choices, device="cuda"):
    """
    Generate buffers related to the Medusa structure.

    This function generates various buffers used in the Medusa structure, which is a complex data structure consisting of a tree-like hierarchy. The buffers include indices, attention masks, position identifiers, and more.

    Args:
        medusa_choices (torch.Tensor): A tensor containing choices for the Medusa structure.
        context_len: int for context lengths
        dtype: data type of the mask tensor
        device (str, optional): Target device for the generated buffers. Defaults to "cuda".

    Returns:
        The attention mask designed specifically for the Medusa structure, ensuring proper attention computation.
    """
    medusa_choices = torch.tensor(medusa_choices)
    cumulative_product = torch.cumprod(medusa_choices, dim=0)
    medusa_len = cumulative_product.sum().item()
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)

    # 2. Update the Medusa attention mask
    prev_cumprod_sum = -1
    for i in range(medusa_choices.size(0)):
        cumprod_sum = cumulative_product[:i].sum().item()
        if prev_cumprod_sum != -1:
            parent_idx = (
                torch.arange(prev_cumprod_sum, cumprod_sum)
                .repeat(medusa_choices[i], 1)
                .transpose(0, 1)
                .flatten()
            )
            medusa_attn_mask[
                cumprod_sum : cumprod_sum + parent_idx.size(0)
            ] += medusa_attn_mask[parent_idx]
        prev_cumprod_sum = cumulative_product[:i].sum().item()
    return medusa_attn_mask.to(device)


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,  # [num_sequences, candidates, num_heads, head_dim]
    query: torch.Tensor,  # [num_sequeces, candidates, num_heads, head_dim]
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,  # [num_blocks, num_heads, head_size, block_size]
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    medusa_attn_mask: torch.Tensor,  # [num_candidates, num_candidates]
) -> None:
    num_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]

    num_input_sequences = query.shape[0]
    num_candidates = query.shape[1]
    for i in range(num_input_sequences):
        q = query[i]  # [candidates, num_heads, head_dim]
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []

        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        dtype = query.dtype
        k_length = keys.shape[0]
        medusa_attn_mask = 1 - medusa_attn_mask
        history_attn_mask = torch.zeros(
            [num_candidates, k_length - num_candidates], device="cuda"
        )
        attn_mask = torch.cat((history_attn_mask, medusa_attn_mask), dim=-1)
        attn_mask = attn_mask.to(query.dtype)
        attn_mask = attn_mask * torch.finfo(dtype).min

        scale = 1.0 / (head_size**0.5)
        out = ref_masked_attention(q, keys, values, scale, attn_mask=attn_mask)
        out = out.view(num_candidates, num_heads, head_size)
        output[i].copy_(out, non_blocking=True)


@torch.inference_mode()
def run_single_query_cached_kv_attention(
    num_sequences: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    num_kv_heads: int = None,
    medusa_choices: List[int] = None,
) -> None:
    # medusa_candidates equals to cum_prod of medusa_choices
    import itertools
    import operator

    medusa_candidates = sum(itertools.accumulate(medusa_choices, operator.mul))

    medusa_mask = generate_medusa_attn_mask(medusa_choices, device="cuda")
    print(f"medusa mask: {medusa_mask}")

    qkv = torch.empty(
        num_sequences,
        medusa_candidates,
        3,
        num_heads,
        head_size,
        dtype=dtype,
        device="cuda",
    )
    qkv.uniform_(-1e-3, 1e-3)

    # query shape: [num_sequences, medusa_candidates, num_heads, head_size]
    query, _, _ = qkv.unbind(dim=2)

    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_block_shape = (num_heads, head_size // x, block_size, x)
    key_cache = torch.empty(
        size=(num_blocks, *key_block_shape), dtype=dtype, device="cuda"
    )
    key_cache.uniform_(-1e-3, 1e-3)
    value_block_shape = (num_heads, head_size, block_size)
    value_cache = torch.empty(
        size=(num_blocks, *value_block_shape), dtype=dtype, device="cuda"
    )
    value_cache.uniform_(-1e-3, 1e-3)

    context_lens = [
        max(random.randint(1, MAX_SEQ_LEN), medusa_candidates + 10)
        for _ in range(num_sequences)
    ]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

    # taking medusa candidates into account
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_sequences):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")
    head_mapping = torch.arange(num_heads, dtype=torch.int32, device="cuda")

    scale = float(1.0 / (head_size**0.5))

    num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    assert num_heads % num_kv_heads == 0
    num_queries_per_kv = num_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"), num_queries_per_kv
    )

    output = torch.empty(
        num_sequences,
        medusa_candidates,
        num_heads,
        head_size,
        dtype=dtype,
        device="cuda",
    )
    paged_flash_attention_fwd(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        None,  # ALiBi slopes.
        medusa_mask,
    )

    ref_output = torch.empty_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        medusa_mask,
    )
    # NOTE(woosuk): Due to the difference in the data types the two
    # implementations use for attention softmax logits and accumulation,
    # there is a small difference in the final outputs.
    # We should use a relaxed tolerance for the test.
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)


def test_single_query_cached_kv_attention() -> None:
    torch.random.manual_seed(TEST_SEED)
    torch.cuda.manual_seed(TEST_SEED)
    for dtype in [torch.half]:  # , torch.bfloat16, torch.float]:
        for block_size in [16]:
            # for block_size in [8, 16, 32]:
            for head_size in [128]:
                # for head_size in [64, 128]:
                print(
                    f"Testing single_query_cached_kv_attention with "
                    f"dtype={dtype}, block_size={block_size}, "
                    f"head_size={head_size}"
                )
                run_single_query_cached_kv_attention(
                    num_sequences=1,  # ?7,
                    num_heads=4,  # 40,
                    head_size=32,
                    block_size=block_size,
                    num_blocks=400,  # 10240,
                    dtype=dtype,
                    medusa_choices=[
                        1,
                        3,
                        4,
                    ],  # suppose we use topk=2 with 2 medusa heads
                )


test_single_query_cached_kv_attention()
