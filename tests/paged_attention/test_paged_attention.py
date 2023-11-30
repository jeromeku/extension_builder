import random
from dataclasses import dataclass
from typing import List

import torch
from original_paged_flash_test import (
    generate_medusa_attn_mask,
    ref_single_query_cached_kv_attention,
)
from paged_flash_attn_utils import SINGLE_QUERY_CONFIGS
from triton_paged_attention import paged_flash_attention_fwd


@dataclass
class DataClassDict(dict):
    def __post_init__(self):
        self.update(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__.values())


@dataclass(kw_only=True)
class SingleQueryArgs(DataClassDict):
    num_sequences: int
    num_heads: int
    head_size: int
    block_size: int
    num_blocks: int
    dtype: torch.dtype
    BLOCK_M: int
    BLOCK_N: int
    MAX_SEQ_LEN: int
    num_kv_heads: int = None
    medusa_choices: List[int] = None


@dataclass(kw_only=True)
class LaunchArgs(DataClassDict):
    num_warps: int
    num_stages: int


SINGLE_QUERY_CONFIGS = {
    "test": SingleQueryArgs(
        num_sequences=1,
        num_heads=4,
        head_size=32,
        block_size=16,
        num_blocks=400,
        dtype=torch.float16,
        num_kv_heads=4,
        medusa_choices=[1, 3, 4],
        BLOCK_M=16,
        BLOCK_N=16,
        MAX_SEQ_LEN=1024,
    ),
    "default": SingleQueryArgs(
        num_sequences=7,
        num_heads=40,
        head_size=128,
        block_size=16,
        num_blocks=10240,
        dtype=torch.float16,
        num_kv_heads=None,
        medusa_choices=[1, 3, 4],
        BLOCK_M=16,
        BLOCK_N=128,
        MAX_SEQ_LEN=2048,
    ),
}


def construct_args(
    num_sequences,
    num_heads,
    head_size,
    block_size,
    num_blocks,
    dtype,
    num_kv_heads,
    medusa_choices,
    MAX_SEQ_LEN,
    BLOCK_M,
    BLOCK_N,
):
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
    return (
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        medusa_mask,
        head_mapping,
        scale,
        max_context_len,
    )


@torch.inference_mode()
def test_paged_attention(query_config):
    # kernel_path = Path(__file__).parent.absolute() / "paged_flash_attention.py"
    # trace_dir = Path(__file__).parent / kernel_path.split(".")[0]

    # if trace_dir.exists():
    #     shutil.rmtree(trace_dir)
    # trace_dir.mkdir(parents=True, exist_ok=True)

    TEST_SEED = 0
    torch.manual_seed(TEST_SEED)
    random.seed(TEST_SEED)

    print(query_config)

    (
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        medusa_mask,
        head_mapping,
        scale,
        max_context_len,
    ) = construct_args(**query_config)

    ref_output = torch.empty_like(query)

    print(f"Query Shape: {query.shape}")
    print(f"Key Cache Shape: {key_cache.shape}")
    print(f"Value Cache Shape: {value_cache.shape}")
    print(f"Block Tables Shape: {block_tables.shape}")
    print(f"Context Lens Shape: {context_lens.shape}")
    print(f"Medusa Mask Shape: {medusa_mask.shape}")

    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        medusa_mask,
    )

    paged_output = torch.empty_like(query)
    block_size = query_config.block_size

    paged_flash_attention_fwd(
        paged_output,
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
    print("Paged Output: ", paged_output)
    print("Ref Output: ", ref_output)
    assert torch.allclose(paged_output, ref_output, atol=1e-3, rtol=1e-1)


test_paged_attention(SINGLE_QUERY_CONFIGS["test"])


# grid = (num_sequences, num_heads, triton.cdiv(num_candidates, args.BLOCK_M))
# num_warps = 4 if head_dim <= 64 else 8
# num_stages = 1
# jit_fn = JITFunction(kernel_path)
# Cross-check kernel args with args
# trace_config = TraceConfig(
#     trace_dir=trace_dir,
# )
# kernel_name = jit_fn.__name__
# results = jit_fn
# compiler = AOTCompiler(
#     kernel_name=kernel_name,
#     jit_args=args,
#     jit_fn=jit_fn,
#     save_dir=codegen_dir,
# )
# compiled_result: AOTCompilationResult = compiler.generate()
# compiled_result: AOTCompilationResult = compiler.generate()
# compiled_result: AOTCompilationResult = compiler.generate()
