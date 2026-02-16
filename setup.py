"""Setup configuration for Coffee Bean Classification package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Development requirements
dev_requirements = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
]

setup(
    name="coffee-bean-classification",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready coffee bean image classification using deep learning with clean OOP architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/coffee-bean-classification",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/coffee-bean-classification/issues",
        "Documentation": "https://github.com/yourusername/coffee-bean-classification/tree/main/docs",
        "Source Code": "https://github.com/yourusername/coffee-bean-classification",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "all": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "coffee-classify=coffee_bean_classification.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "deep-learning",
        "image-classification",
        "computer-vision",
        "coffee-bean",
        "tensorflow",
        "machine-learning",
        "cnn",
        "transfer-learning",
    ],
)
