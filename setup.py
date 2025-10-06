"""
Setup script for AxonNet library.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="axonet",
    version="0.1.0",
    author="AxonNet Team",
    author_email="",
    description="A comprehensive library for neuron morphology analysis from SWC files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/axonet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
        "trimesh>=3.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "plotly>=5.0.0",
            "pyvista>=0.30.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "axonet-analyze=axonet.cli:analyze",
            "axonet-visualize=axonet.cli:visualize",
        ],
    },
)
