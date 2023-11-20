#include <cuda_fp16.h>
#include <torch/torch.h>

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

inline __device__ __host__ size_t div_ceil(size_t a, size_t b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
__global__ void add_vector_kernel(const at::Half *a, const at::Half *b, at::Half *c, size_t N)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < N)
        c[id] = a[id] + b[id];
}

void add_vectors(const at::Half *a, const at::Half *b, at::Half *c, size_t N)
{
    add_vector_kernel<<<div_ceil(N, 256), 256>>>(a, b, c, N);
}

__global__ void simtNaiveKernel(const at::Half *A, const at::Half *B, at::Half *C, size_t M,
                                size_t N, size_t K)
{
    size_t row = threadIdx.y + blockDim.y * blockIdx.y;
    size_t col = threadIdx.x + blockDim.x * blockIdx.x;

    if (row >= M && col >= N)
    {
        return;
    }

    at::Half tmp = 0.0f;
#pragma unroll
    for (size_t i = 0; i < K; ++i)
    {
        tmp += A[row * K + i] * B[i + col * K];
    }

    C[row * N + col] = tmp;
}

void simtNaive(const at::Half *A, const at::Half *B, at::Half *C, size_t M, size_t N, size_t K)
{
    dim3 block(16, 16);
    dim3 grid(div_ceil(N, block.x), div_ceil(M, block.y));

    simtNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}
