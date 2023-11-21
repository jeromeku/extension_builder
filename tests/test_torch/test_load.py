import torch

from extension_builder.builders.torchtest import TorchTestBuilder

test_lib = TorchTestBuilder().load()
M, N, K = 128, 128, 128
dtype = torch.float16
device = torch.device("cuda")
A = torch.randn(M, K, dtype=dtype, device=device)
B = torch.randn(K, N, dtype=dtype, device=device)
# out = torch.empty(M, N, dtype=dtype, device=device)
C = test_lib.triton_mm(A, B)
check = torch.allclose(torch.mm(A, B).to(torch.float32), C, atol=1e-2, rtol=1e-1)
print(f"check: {check}")
