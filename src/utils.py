from pathlib import Path


def get_root_path() -> Path:
    """Return root directory of the repository."""
    return Path(__file__).parent.parent.absolute()
