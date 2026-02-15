"""Setup configuration for coffee_bean_classification package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')

# Read version
version = {}
version_path = this_directory / "coffee_bean_classification" / "version.py"
if version_path.exists():
    exec(version_path.read_text(), version)
else:
    version['__version__'] = '0.1.0'

# Core dependencies
REQUIRED = [
    'tensorflow>=2.15.0',
    'numpy>=1.24.0',
    'pandas>=2.0.0',
    'matplotlib>=3.7.0',
    'seaborn>=0.12.0',
    'scikit-learn>=1.3.0',
    'pillow>=10.0.0',
    'pyyaml>=6.0',
]

# Development dependencies
DEV_REQUIRED = [
    'pytest>=7.4.0',
    'pytest-cov>=4.1.0',
    'black>=23.0.0',
    'flake8>=6.0.0',
    'mypy>=1.5.0',
    'pylint>=2.17.0',
    'ipython>=8.14.0',
    'jupyter>=1.0.0',
]

# Optional dependencies
EXTRAS = {
    'dev': DEV_REQUIRED,
    'mlflow': ['mlflow>=2.8.0'],
    'wandb': ['wandb>=0.16.0'],
    'tensorboard': ['tensorboard>=2.15.0'],
    'all': DEV_REQUIRED + ['mlflow>=2.8.0', 'wandb>=0.16.0', 'tensorboard>=2.15.0'],
}

setup(
    name='coffee_bean_classification',
    version=version['__version__'],
    description='Production-ready Coffee Bean Classification with OOP Design',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Coffee Bean Classification Team',
    author_email='',
    url='https://github.com/yourusername/coffee_bean_classification',
    packages=find_packages(exclude=['tests', 'tests.*', 'examples']),
    include_package_data=True,
    python_requires='>=3.9',
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='machine-learning deep-learning computer-vision image-classification coffee',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/coffee_bean_classification/issues',
        'Source': 'https://github.com/yourusername/coffee_bean_classification',
        'Documentation': 'https://coffee-bean-classification.readthedocs.io',
    },
)
