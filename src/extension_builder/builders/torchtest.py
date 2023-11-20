import os

import torch

from .builder import Builder
from .utils import append_nvcc_threads


class TorchTestBuilder(Builder):
    NAME = "torch_test"
    SOURCE_DIR = "triton"

    def __init__(self):
        super().__init__(
            name=self.NAME,
        )

    def include_dirs(self):
        source_includes = [
            # self.csrc_abs_path(self.SOURCE_DIR),
            self.get_cuda_home_include(),
        ]

        return source_includes

    def sources_files(self):
        ret = [
            self.csrc_abs_path(fname)
            for fname in [
                f"{self.SOURCE_DIR}/torch_test.cpp",
                f"{self.SOURCE_DIR}/matmul_test.cu",
            ]
        ]
        return ret

    def cxx_flags(self):
        return ["-O3"]  # + self.version_dependent_macros

    def nvcc_flags(self):
        compute_capability = torch.cuda.get_device_capability()
        cuda_arch = compute_capability[0] * 100 + compute_capability[1] * 10

        extra_cuda_flags = [
            "-v",
            f"-DCUDA_ARCH={cuda_arch}",
        ]

        ret = ["-O3"] + extra_cuda_flags
        return append_nvcc_threads(ret)

    def builder(self):
        try:
            super().builder()
        except:
            import warnings

            warnings.warn("build smoothquant lib not successful")
