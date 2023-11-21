#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include "kernel.h"
#include <cuda.h>
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
// void simtNaive(const at::Half *A, const at::Half *B, at::Half *C, size_t M, size_t N, size_t K);
// void add_vectors(const at::Half *a, const at::Half *b, at::Half *c, size_t N);

at::Tensor triton_mm(const at::Tensor A, const at::Tensor B)
{
    int32_t M = A.size(0);
    int32_t K = A.size(1);
    int32_t N = B.size(1);

    // Get strides
    TORCH_CHECK(A.is_contiguous());
    TORCH_CHECK(A.dim() == 2);
    auto stride_am = A.stride(0);
    auto stride_ak = A.stride(1);
    TORCH_CHECK(stride_am == K);
    TORCH_CHECK(stride_ak == 1);

    TORCH_CHECK(B.is_contiguous());
    TORCH_CHECK(B.dim() == 2);
    auto stride_bk = B.stride(0);
    auto stride_bn = B.stride(1);
    TORCH_CHECK(stride_bk == N);
    TORCH_CHECK(stride_bn == 1);

    auto C = at::zeros({M, N}, A.options()).to(at::kFloat);
    auto stride_cm = C.stride(0);
    auto stride_cn = C.stride(1);
    TORCH_CHECK(stride_cm == N);
    TORCH_CHECK(stride_cn == 1);

    TORCH_CHECK(A.dtype() == torch::kFloat16);
    TORCH_CHECK(B.dtype() == torch::kFloat16);

    // Set current CUDA context to torch context
    TORCH_CHECK(cudaSetDevice(A.get_device()) == cudaSuccess);
    CUstream stream = at::cuda::getCurrentCUDAStream();
    cuStreamSynchronize(stream);

    load_matmul_fp16();

    CUresult ret = matmul_fp16_default(stream,
                                       reinterpret_cast<CUdeviceptr>(C.data_ptr()),
                                       reinterpret_cast<CUdeviceptr>(A.data_ptr()),
                                       reinterpret_cast<CUdeviceptr>(B.data_ptr()),
                                       M, N, K, stride_cm, stride_cn, stride_am, stride_ak, stride_bk, stride_bn);
    cuStreamSynchronize(stream);
    TORCH_CHECK(ret == 0);

    unload_matmul_fp16();
    return C;
}
// at::Tensor add_vec(const at::Tensor a, const at::Tensor b)
// {
//     size_t N = a.size(0);
//     TORCH_CHECK(a.is_contiguous());
//     TORCH_CHECK(b.is_contiguous());
//     TORCH_CHECK(a.dtype() == torch::kFloat16);
//     TORCH_CHECK(b.dtype() == torch::kFloat16);
//     TORCH_CHECK(a.is_cuda());
//     TORCH_CHECK(b.is_cuda());
//     auto C = at::empty({N}, a.options());
//     add_vectors(a.data_ptr<at::Half>(), b.data_ptr<at::Half>(), C.data_ptr<at::Half>(), N);
//     return C;
// }
// at::Tensor matmul(const at::Tensor A, const at::Tensor B)
// {
//     size_t M = A.size(0);
//     size_t N = B.size(1);
//     size_t K = A.size(1);
//     TORCH_CHECK(A.is_contiguous());
//     TORCH_CHECK(B.is_contiguous());
//     TORCH_CHECK(A.dtype() == torch::kFloat16);
//     TORCH_CHECK(B.dtype() == torch::kFloat16);
//     TORCH_CHECK(A.is_cuda());
//     TORCH_CHECK(B.is_cuda());

//     auto C = at::empty({M, N}, A.options());
//     simtNaive(A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>(), M, N, K);
//     return C;
// }
// at::Tensor vector_add(at::Tensor input, at::Tensor weight)
// {

//     int64_t batch_size = input.size(0);
//     int64_t in_features = input.size(1);
//     int64_t out_features = weight.size(0);

//     TORCH_CHECK(input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16);
//     TORCH_CHECK(input.dtype() == weight.dtype());
//     TORCH_CHECK(input.is_cuda());
//     TORCH_CHECK(weight.is_cuda());
//     TORCH_CHECK(input.is_contiguous());
//     TORCH_CHECK(weight.is_contiguous());
//     CHECK_SHAPE(input, batch_size, in_features);
//     CHECK_SHAPE(weight, out_features, in_features);
//     // Otherwise the kernel will be launched from cuda:0 device
//     // Cast to char to avoid compiler warning about narrowing
//     // at::cuda::CUDAGuard device_guard{(char)input.get_device()};

//     // create output/workspace tensor
//     auto opts = input.options();
//     auto out = torch::nn::functional::linear(input, weight);
//     // auto output = at::ones({batch_size, out_features}, opts);
//     // at::Tensor pre_act;
//     // // If ReLU, cuBlasLT stores a bit-mask (1 bit per element)
//     // if (save_pre_act)
//     // {
//     //     pre_act = at::empty({batch_size, is_gelu ? out_features : out_features / 8},
//     //                         is_gelu ? opts : opts.dtype(torch::kUInt8));
//     // }
//     // See https://github.com/pytorch/pytorch/issues/73328 for reasoning behind setting this to 1M.
//     // However, Apex sets it to 4M and TransformerEngine sets to 32M for Hopper and 4M for other GPUs
//     // https://github.com/NVIDIA/TransformerEngine/blob/a0f0065498bbcfc1da78cf9e8b166f5381613fbc/transformer_engine/pytorch/module.py#L91
//     // size_t workspaceSize = 1024 * 1024 * (at::cuda::getCurrentDeviceProperties()->major >= 9 ? 32 : 4);
//     // auto lt_workspace = at::empty({static_cast<int64_t>(workspaceSize)}, opts.dtype(torch::kUInt8));

//     // DISPATCH_HALF_AND_BF16(input.scalar_type(), "linear_act_forward", [&]
//     // {
//     // auto result = linear_act_forward_cuda<scalar_t>(
//     //     input.data_ptr<scalar_t>(),
//     //     weight.data_ptr<scalar_t>(),
//     //     bias_.has_value()? bias_.value().data_ptr<scalar_t>() : nullptr,
//     //     in_features,
//     //     batch_size,
//     //     out_features,
//     //     is_gelu,
//     //     heuristic,
//     //     output.data_ptr<scalar_t>(),
//     //     save_pre_act ? pre_act.data_ptr() : nullptr,
//     //     (void*) (lt_workspace.data_ptr()),
//     //     workspaceSize);
//     // TORCH_CHECK(result == 0, "linear_act_forward failed."); });

//     // std::vector<at::Tensor> result = {output};
//     // if (save_pre_act)
//     // {
//     //     result.push_back(pre_act);
//     // };
//     return out;
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("triton_mm", &triton_mm, "triton matmul");
    // m.def("vector_add", &vector_add, "vector add");
    // m.def("matmul", &matmul, "matrix multiplication reference");
    // m.def("add_vec", &add_vec, "add vec kernel");
}
