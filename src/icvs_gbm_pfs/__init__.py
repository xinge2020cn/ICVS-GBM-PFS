"""ICVS-GBM-PFS glioblastoma progression-free survival research pipeline."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("icvs-gbm-pfs")
except PackageNotFoundError:
    __version__ = "0+local"

__all__ = ["__version__"]
