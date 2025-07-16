import logging
import re
from keyword import iskeyword
from pathlib import Path


def get_root_path() -> Path:
    """Return root directory of the repository."""
    return Path(__file__).parent.parent.parent.absolute()


def setup_basic_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def sanitize_name(name: str, strict: bool = True) -> str:
    n = re.sub(r"[^0-9a-z]+", "_", name.lower())
    r = n.rstrip("_")
    if not r:
        raise ValueError(f"{name} does not contain any valid characters")
    if strict:
        r = r if not r[0].isdigit() else f"_{r}"
        r = r if r.isidentifier() and not iskeyword(r) else f"_{r}"
        r = r if not r[0] == "_" else f"f{r}"
    return r
