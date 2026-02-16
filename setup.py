"""
sparsemma: INT8 Sparse Tensor Core GEMM kernels for PyTorch.

Install:
    pip install .

    # Or editable (for development):
    pip install -e .
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

cuda_flags = [
    "-O3",
    "--use_fast_math",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_80,code=sm_80",
    "--allow-unsupported-compiler",
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sparsemma",
    version="0.1.0",
    description="INT8 Sparse Tensor Core GEMM kernels for PyTorch — built for Windows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="WizardsForgeGames",
    url="https://github.com/WizardsForgeGames/sparsemma",
    license="MIT",
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="int8_gemm_tc",
            sources=[os.path.join(csrc_dir, "int8_gemm_tc.cu")],
            extra_compile_args={"nvcc": cuda_flags},
        ),
        CUDAExtension(
            name="int8_sparse_tc",
            sources=[os.path.join(csrc_dir, "int8_sparse_tc.cu")],
            extra_compile_args={"nvcc": cuda_flags},
        ),
        CUDAExtension(
            name="int8_kernels",
            sources=[os.path.join(csrc_dir, "int8_kernels.cu")],
            extra_compile_args={"nvcc": cuda_flags},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "int8", "quantization", "sparse", "tensor-core", "cuda",
        "pytorch", "gemm", "inference", "windows", "gpu",
        "mma", "ptx", "nvidia", "ampere", "ada-lovelace",
        "structured-sparsity", "vram", "optimization",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/WizardsForgeGames/sparsemma/issues",
        "Source Code": "https://github.com/WizardsForgeGames/sparsemma",
    },
)
