"""DLR3 glioblastoma progression-free survival research pipeline."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dlr3-gbm-pfs")
except PackageNotFoundError:
    __version__ = "0+local"

__all__ = ["__version__"]
