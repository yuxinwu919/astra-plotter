from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="astra_plotter",
    version="1.0.0",
    author="ASTRA Plotter Team",
    description="ASTRA simulation data post-processing and visualization toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/astra_plotter",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "matplotlib>=3.1.0",
    ],
    extras_require={
        "gui": ["ipywidgets>=7.5.0", "jupyter>=1.0.0"],
        "dev": ["pytest>=6.0", "pytest-cov"],
    },
)
