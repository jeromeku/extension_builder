from setuptools import find_packages, setup

REQUIREMENTS = [
    "torch@https://download.pytorch.org/whl/cu118/torch-2.1.1%2Bcu118-cp311-cp311-linux_x86_64.whl",
    "ninja",
]
setup(
    name="extension_builder",
    version="0.0.1",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    author="JK",
    author_email="jk@jk.com",
    description="PyTorch Dynamic Extension Loader",
    long_description="Load PyTorch extensions dynamically",
    license="Apache 2.0 License",
    package_dir={"": "extension_builder"},
    packages=find_packages("extension_builder"),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
    # entry_points={"console_scripts": ["transformers-cli=transformers.commands.transformers_cli:main"]},
    # python_requires=">=3.8.0",
    install_requires=REQUIREMENTS,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # cmdclass={"deps_table_update": DepsTableUpdateCommand},
)
