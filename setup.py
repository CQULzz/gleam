from setuptools import find_packages, setup


setup(
    name="lynx_gleam",
    version="0.1.0",
    description="Lynx_gleam Isaac-Lab-only package",
    author="Lynx_gleam maintainers",
    license="BSD-3-Clause",
    python_requires=">=3.11",
    packages=find_packages(
        include=[
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
)
