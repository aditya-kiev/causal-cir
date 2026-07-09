from setuptools import setup, find_packages

setup(
    name="causal-cir",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "pandas>=1.3",
        "tqdm>=4.62",
    ],
)
