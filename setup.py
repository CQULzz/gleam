from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


pathfinding_2D_cuda = CUDAExtension(
    name="bfs_cuda_2D",
    sources=[
        "gleam/utils/bfs_cuda_2D.cpp",
        "gleam/utils/bfs_cuda_kernel_2D.cu",
    ],
)


setup(
    name="gleam_lab",
    version="0.1.0",
    description="GLEAM with an Isaac Lab migration path for modern GPU stacks",
    author="CQULzz",
    license="BSD-3-Clause",
    python_requires=">=3.11",
    packages=find_packages(
        include=[
            "gleam",
            "gleam.*",
            "gleam_lab",
            "gleam_lab.*",
            "stable_baselines3",
            "stable_baselines3.*",
            "wandb_utils",
            "wandb_utils.*",
        ]
    ),
    install_requires=[
        "gym==0.26.2",
        "numpy==1.26.0",
        "matplotlib",
        "tensorboard",
        "cloudpickle",
        "pandas==2.2.3",
        "scipy",
        "pyyaml",
        "yapf==0.30.0",
        "wandb==0.25.0",
        "opencv-python==4.11.0.86",
    ],
    ext_modules=[pathfinding_2D_cuda],
    cmdclass={"build_ext": BuildExtension},
)
