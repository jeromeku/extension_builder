# This code has been adapted from the DeepSpeed library.
# Copyright (c) Microsoft Corporation.

# Licensed under the MIT License.
import importlib
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from extension_builder import MODULE_ROOT

from .utils import check_cuda_availability, check_system_pytorch_cuda_match

CSRC_DIR = MODULE_ROOT.parent / "csrc"
KERNEL_DIR = CSRC_DIR / "ops"
THIRDPARTY_DIR = CSRC_DIR / "third_party"


class Builder(ABC):
    """
    Builder is the base class to build extensions for PyTorch.

    Args:
        name (str): the name of the kernel to be built
        prebuilt_import_path (str): the path where the extension is installed during pip install
    """

    def __init__(self, name: str):
        self.name = name

        # we store the op as an attribute to avoid repeated building and loading
        self.cached_op_module = None

    def get_cuda_home_include(self):
        """
        return include path inside the cuda home.
        """
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            raise RuntimeError(
                "CUDA_HOME is None, please set CUDA_HOME to compile C++/CUDA kernels in ColossalAI."
            )
        cuda_include = os.path.join(CUDA_HOME, "include")
        return cuda_include

    def csrc_abs_path(self, path):
        return os.path.join(KERNEL_DIR, path)

    # functions must be overrided begin
    @abstractmethod
    def sources_files(self) -> List[str]:
        """
        This function should return a list of source files for extensions.
        """
        raise NotImplementedError

    @abstractmethod
    def include_dirs(self) -> List[str]:
        """
        This function should return a list of include files for extensions.
        """

    @abstractmethod
    def cxx_flags(self) -> List[str]:
        """
        This function should return a list of cxx compilation flags for extensions.
        """

    def ld_flags(self) -> List[str]:
        """
        This function should return a list of ld flags for extensions.
        """
        return []

    @abstractmethod
    def nvcc_flags(self) -> List[str]:
        """
        This function should return a list of nvcc compilation flags for extensions.
        """

    # functions must be overrided over
    def strip_empty_entries(self, args):
        """
        Drop any empty strings from the list of compile and link flags
        """
        return [x for x in args if len(x) > 0]

    def import_op(self):
        """
        This function will import the op module by its string name.
        """
        return importlib.import_module(self.prebuilt_import_path)

    def check_runtime_build_environment(self):
        """
        Check whether the system environment is ready for extension compilation.
        """
        try:
            from torch.utils.cpp_extension import CUDA_HOME

            TORCH_AVAILABLE = True
        except ImportError:
            TORCH_AVAILABLE = False
            CUDA_HOME = None

        if not TORCH_AVAILABLE:
            raise ModuleNotFoundError(
                "PyTorch is not found. You need to install PyTorch first in order to build CUDA extensions"
            )

        if CUDA_HOME is None:
            raise RuntimeError(
                "CUDA_HOME is not found. You need to export CUDA_HOME environment variable or install CUDA Toolkit first in order to build CUDA extensions"
            )

        # make sure CUDA is available for compilation during
        cuda_available = check_cuda_availability()
        if not cuda_available:
            raise RuntimeError(
                "CUDA is not available on your system as torch.cuda.is_available() returns False."
            )

        # make sure system CUDA and pytorch CUDA match, an error will raised inside the function if not
        check_system_pytorch_cuda_match(CUDA_HOME)

    def load(self, verbose: Optional[bool] = True):
        """
        load the kernel during runtime. If the kernel is not built during pip install, it will build the kernel.
        If the kernel is built during runtime, it will be stored in `~/.cache/colossalai/torch_extensions/`. If the
        kernel is built during pip install, it can be accessed through `colossalai._C`.

        Warning: do not load this kernel repeatedly during model execution as it could slow down the training process.

        Args:
            verbose (bool, optional): show detailed info. Defaults to True.
        """
        # if the kernel has be compiled and cached, we directly use it
        if self.cached_op_module is not None:
            return self.cached_op_module

        self.check_runtime_build_environment()

        # time the kernel compilation
        start_build = time.time()

        # construct the build directory
        import torch
        from torch.utils.cpp_extension import load

        torch_version_major = torch.__version__.split(".")[0]
        torch_version_minor = torch.__version__.split(".")[1]
        torch_cuda_version = torch.version.cuda
        home_directory = os.path.expanduser("~")
        extension_directory = f".cache/extension_builder/torch_extensions/torch{torch_version_major}.{torch_version_minor}_cu{torch_cuda_version}"
        build_directory = os.path.join(home_directory, extension_directory)
        Path(build_directory).mkdir(parents=True, exist_ok=True)

        if verbose:
            f"[extension] Compiling or loading the JIT-built {self.name} kernel during runtime now"

        # load the kernel
        op_module = load(
            name=self.name,
            sources=self.strip_empty_entries(self.sources_files()),
            extra_include_paths=self.strip_empty_entries(self.include_dirs()),
            extra_cflags=self.cxx_flags(),
            extra_cuda_cflags=self.nvcc_flags(),
            extra_ldflags=self.ld_flags(),
            build_directory=build_directory,
            verbose=verbose,
        )

        build_duration = time.time() - start_build

        # log jit compilation time
        if verbose:
            f"[extension] Time to compile or load {self.name} op: {build_duration} seconds"

        # cache the built/loaded kernel
        self.cached_op_module = op_module

        return op_module

    # def builder(self) -> CUDAExtension:
    #     """
    #     get a CUDAExtension instance used for setup.py
    #     """
    #     from torch.utils.cpp_extension import CUDAExtension

    #     return CUDAExtension(
    #         name=self.prebuilt_import_path,
    #         sources=self.strip_empty_entries(self.sources_files()),
    #         include_dirs=self.strip_empty_entries(self.include_dirs()),
    #         extra_compile_args={
    #             "cxx": self.strip_empty_entries(self.cxx_flags()),
    #             "nvcc": self.strip_empty_entries(self.nvcc_flags()),
    #         },
    #     )
