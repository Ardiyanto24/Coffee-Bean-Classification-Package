"""Setup configuration for Coffee Bean Classification package."""

from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent

# 1. Baca isi README.md untuk deskripsi panjang
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# 2. Baca versi dari file version.py secara dinamis
version = {}
version_path = this_directory / "coffee_bean_classification" / "version.py"
if version_path.exists():
    exec(version_path.read_text(), version)
else:
    version["__version__"] = "1.0.0"

# 3. Baca dependensi dari requirements.txt
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# 4. Kebutuhan untuk Development dan MLOps
DEV_REQUIRED = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
]

EXTRAS = {
    "dev": DEV_REQUIRED,
    "mlflow": ["mlflow>=2.8.0"],
    "wandb": ["wandb>=0.16.0"],
    "all": DEV_REQUIRED + ["mlflow>=2.8.0", "wandb>=0.16.0"],
}

setup(
    name="coffee-bean-classification",
    version=version.get("__version__", "1.0.0"),
    author="Coffee Bean Classification Team",
    author_email="your.email@example.com",
    description="Production-ready coffee bean image classification using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/coffee-bean-classification",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require=EXTRAS,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=["deep-learning", "image-classification", "computer-vision", "coffee-bean"],
)
