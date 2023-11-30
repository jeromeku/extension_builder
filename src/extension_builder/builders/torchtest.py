import os
import subprocess

import torch

from .builder import Builder
from .utils import append_nvcc_threads


def libcuda_dirs():
    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    msg = "libcuda.so cannot found!\n"
    if locs:
        msg += "Possible files are located at %s." % str(locs)
        msg += "Please create a symlink of libcuda.so to any of the file."
    else:
        msg += 'Please make sure GPU is setup and then run "/sbin/ldconfig"'
        msg += " (requires sudo) to refresh the linker cache."
    assert any(os.path.exists(os.path.join(path, "libcuda.so")) for path in dirs), msg
    return dirs


class TorchTestBuilder(Builder):
    NAME = "torch_test"
    SOURCE_DIR = "triton"

    def __init__(self):
        super().__init__(
            name=self.NAME,
        )

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
                f"{self.SOURCE_DIR}/torch_test.cpp",
                # f"{self.SOURCE_DIR}/matmul_test.cu",
                f"{self.SOURCE_DIR}/matmul_fp16.3db33494_01234567891011.cu",
                f"{self.SOURCE_DIR}/kernel.cu",
            ]
        ]
        return ret

    def cxx_flags(self):
        return [
            "-O3",
        ]

    def ld_flags(self):
        return ["".join(["-L"] + libcuda_dirs()), "-lcuda"]

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
