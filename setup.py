from setuptools import find_packages, setup

install_requires = [
    "decorator==4.4.2",
    "torch",
    "cython",
    "torch_sparse",
    "torch_scatter",
    "torch_geometric",
    "numpy",
    "networkx",
    "dask",
    "dask[dataframe]",
    "pynvml", 
    "tables",
    "dask_pytorch_ddp",
    "distributed",
]
tests_require = ["pytest", "pytest-cov", "mock", "networkx", "tqdm"]

keywords = [
    "machine-learning",
    "distributed data parallel",
    "dask",
    "deep-learning",
    "deeplearning",
    "deep learning",
    "machine learning",
    "signal processing",
    "temporal signal",
    "graph",
    "dynamic graph",
    "embedding",
    "dynamic embedding",
    "graph convolution",
    "gcn",
    "graph neural network",
    "graph attention",
    "lstm",
    "temporal network",
    "representation learning",
    "learning",
]

setup(
    name="torch_geometric_temporal",
    packages=find_packages(),
    version="1.0.0",
    license="MIT",
    description="A distributed and scalability extension for PyTorch Geometric Temporal.",
    author="",
    author_email="",
    url="",
    download_url="",
    keywords=keywords,
    install_requires=install_requires,
    extras_require={
        "test": tests_require,
    },
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
