import tomllib as toml
from pathlib import Path


def _get_version() -> str:
    """Get the version from pyproject.toml."""
    with open(
        Path(__file__).parent.parent / "pyproject.toml",
        "rb",
    ) as f:
        return toml.load(f)["tool"]["poetry"]["version"]


__version__ = _get_version()
