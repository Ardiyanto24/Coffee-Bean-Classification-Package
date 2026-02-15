"""Version information for coffee_bean_classification package."""

__version__ = "0.1.0"
__author__ = "Coffee Bean Classification Team"
__description__ = "Production-ready Coffee Bean Classification with OOP Design"
__license__ = "MIT"

# Version info
VERSION_INFO = {
    "major": 0,
    "minor": 1,
    "patch": 0,
    "release": "alpha"
}

def get_version():
    """Get the current version string."""
    return __version__

def get_version_info():
    """Get detailed version information."""
    return VERSION_INFO
