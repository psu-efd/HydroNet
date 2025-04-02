from sys import version_info

# Define version based on Python version
if version_info >= (3, 8):
    # Python 3.8+ has importlib.metadata built-in
    try:
        from importlib.metadata import version as get_version
        __version__ = get_version("HydroNet")
    except Exception:
        __version__ = "unknown"
else:
    # For older Python versions, we need the backport
    try:
        from importlib_metadata import version as get_version
        __version__ = get_version("HydroNet")
    except ImportError:
        # If importlib_metadata is not installed
        __version__ = "unknown"


def get_HydroNet_version_info():
    return ", ".join(
        [
            f"HydroNet v{__version__}",
            f"Python {version_info.major}.{version_info.minor}.{version_info.micro}",
            "Copyright (c) 2025 Xiaofeng Liu"
        ]
    )