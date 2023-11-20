#include <cuda.h>
#include <stdint.h>
#include <assert.h>

// launcher for: matmul_fp16_16x16x16_warps1xstages3
CUresult matmul_fp16_3db33494_01234567891011(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn);

CUresult matmul_fp16_16x16x16_warps1xstages3(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn){
if (1)
    return matmul_fp16_3db33494_01234567891011(stream, C, A, B, M, N, K, stride_cm, stride_cn, stride_am, stride_ak, stride_bk, stride_bn);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: matmul_fp16_16x16x16_warps1xstages3
void load_matmul_fp16_3db33494_01234567891011();
void load_matmul_fp16_16x16x16_warps1xstages3() {
  load_matmul_fp16_3db33494_01234567891011();
}

// unload for: matmul_fp16_16x16x16_warps1xstages3
void unload_matmul_fp16_3db33494_01234567891011();
void unload_matmul_fp16_16x16x16_warps1xstages3() {
  unload_matmul_fp16_3db33494_01234567891011();
}

typedef CUresult (*kernel_func_t)(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn);
kernel_func_t matmul_fp16_kernels[] = {
  matmul_fp16_16x16x16_warps1xstages3,
};

int matmul_fp16_get_num_algos(void){
  return (int)sizeof(matmul_fp16_kernels);
}

CUresult matmul_fp16(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int algo_id){
  assert (algo_id < (int)sizeof(matmul_fp16_kernels));
  return matmul_fp16_kernels[algo_id](stream, C, A, B, M, N, K, stride_cm, stride_cn, stride_am, stride_ak, stride_bk, stride_bn);
}

void load_matmul_fp16(void){
  load_matmul_fp16_16x16x16_warps1xstages3();
}

void unload_matmul_fp16(void){
  unload_matmul_fp16_16x16x16_warps1xstages3();
}


CUresult matmul_fp16_default(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn){
  return matmul_fp16(stream, C, A, B, M, N, K, stride_cm, stride_cn, stride_am, stride_ak, stride_bk, stride_bn, 0);
}
