#!/usr/bin/env python3
"""
Setup script for NWB Data Viewer
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nwb-data-viewer",
    version="1.0.0",
    author="Parviz Ghaderi",
    author_email="parviz.ghaderi@epfl.ch",
    description="Interactive Neural Data Visualization Tool for Contextual Gating Studies",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nwb-data-viewer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nwb-viewer=nwb_data_viewer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.npy", "*.mat"],
    },
    keywords="neuroscience, nwb, data-visualization, brain, psth, allen-atlas",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/nwb-data-viewer/issues",
        "Source": "https://github.com/yourusername/nwb-data-viewer",
        "Documentation": "https://github.com/yourusername/nwb-data-viewer#readme",
    },
) 