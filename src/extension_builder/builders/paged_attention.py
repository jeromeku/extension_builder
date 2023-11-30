import torch

from .builder import Builder
from .utils import append_nvcc_threads


class PagedAttentionBuilder(Builder):
    NAME = "paged_attention"
    SOURCE_DIR = "paged_attention"

    def __init__(self, debug=False):
        super().__init__(
            name=self.NAME,
        )
        self.debug = debug

    def include_dirs(self):
        source_includes = [
            self.csrc_abs_path(self.SOURCE_DIR),
            self.get_cuda_home_include(),
        ]

        return source_includes

    def sources_files(self):
        ret = [
            self.csrc_abs_path(fname)
            for fname in [
                f"{self.SOURCE_DIR}/attention.cpp",
                f"{self.SOURCE_DIR}/attention_extension.cpp",
            ]
        ]
        return ret

    def cxx_flags(self):
        return ["-O3"] if not self.debug else ["-g3"]  # + self.version_dependent_macros

    def nvcc_flags(self):
        compute_capability = torch.cuda.get_device_capability()
        cuda_arch = compute_capability[0] * 100 + compute_capability[1] * 10

        extra_cuda_flags = [
            "-v",
            f"-DCUDA_ARCH={cuda_arch}",
        ]

        opt_flags = ["-O3"] if not self.debug else ["-G"]

        return append_nvcc_threads(extra_cuda_flags + opt_flags)

    def builder(self):
        try:
            super().builder()
        except Exception as e:
            print("Failed to build paged attention: ", e)
