from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="enhanced-opro",
    version="0.2.0",
    author="Enhanced OPRO Contributors",
    author_email="",
    description="Enhanced version of OPRO with configurable endpoints and improved dataset interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-deepmind/opro",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pydantic>=2.0.0",
        "tqdm>=4.65.0",
        "jsonlines>=3.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
            "mypy>=0.990",
        ],
    },
)
