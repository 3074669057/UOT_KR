"""Cross-chain Celer ETH↔BNB mapping toolkit (package root)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("celer-cross")
except PackageNotFoundError:
    __version__ = "0.0.0"

from cross.interfaces.cli import main

__all__ = ["__version__", "main"]
