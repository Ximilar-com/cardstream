"""cardstream — real-time trading-card identification.

The version is authored once, in pyproject.toml; everything else reads it
from the installed package metadata.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("cardstream")
except PackageNotFoundError:  # bare source tree on PYTHONPATH, no install
    __version__ = "0.0.0+uninstalled"
