"""Central logging setup.

One ``configure_logging`` call sets the root level (DEBUG when the ``DEBUG``
env/setting is on, INFO otherwise) and every module grabs a named logger via
``get_logger(__name__)``.
"""

from __future__ import annotations

import logging

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(debug: bool = False) -> None:
    """Initialise root logging once. Idempotent — safe to call from both the app
    factory and ``main.py``; the first call wins, later calls only adjust level."""
    global _CONFIGURED
    level = logging.DEBUG if debug else logging.INFO
    if not _CONFIGURED:
        logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATEFMT)
        _CONFIGURED = True
    logging.getLogger().setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Names are trimmed of the leading package so the
    log prefix stays short (e.g. ``analysis.pipeline`` -> ``pipeline``)."""
    short = name.rsplit(".", 1)[-1]
    return logging.getLogger(short)
