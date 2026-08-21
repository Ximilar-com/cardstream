"""Where ``--store-images`` puts the images behind the calls you paid for.

One file per identify call, in one of two shapes (``--store-images-type``):

* ``object`` (the default) — the crop, written from the record's own
  ``_base64``, so the folder holds exactly what went on the wire, upscaling and
  JPEG quality included, not a re-encode of it. When a card comes back
  unmatched the question is always "what did the endpoint actually see", and a
  re-encoded crop cannot answer it.
* ``frame`` — the whole frame the crop was cut from, which is what you want
  when the question is about the *scene*: where the card was, what else was in
  shot, whether the detector boxed the right thing.

The mode lives here so both call sites stay unconditional — the identify path
offers every record's base64, the analyzer offers every frame, and the store
keeps whichever one it was asked for. Either way the numbering is one sequence.

Saving never raises. A full disk costs you the archive, not the show.
"""

from __future__ import annotations

import base64
import binascii
import os
import secrets
import threading
from pathlib import Path

import numpy as np

from cardstream.core.imaging import encode_jpeg
from cardstream.log import get_logger

logger = get_logger(__name__)

# --store-images-type values. OBJECT is the crop that was identified; FRAME is
# the whole picture it came out of.
OBJECT, FRAME = "object", "frame"
STORE_TYPES = (OBJECT, FRAME)


class ImageStore:
    """A folder of identify images, numbered in call order.

    Built only when the flag is given. The folder is created and proved
    writable in the constructor so a bad path fails at startup rather than two
    hours into a stream, and the ValueError it raises is the one the
    entrypoints already turn into a clean ``error:`` exit.
    """

    def __init__(self, folder: str | os.PathLike[str], kind: str = OBJECT) -> None:
        if kind not in STORE_TYPES:
            raise ValueError(
                f"--store-images-type {kind!r}: expected one of {', '.join(STORE_TYPES)}"
            )
        self.kind = kind
        self.folder = Path(folder).expanduser()
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"--store-images {self.folder}: {exc}") from exc
        if not os.access(self.folder, os.W_OK):
            raise ValueError(f"--store-images {self.folder}: not writable")
        # Identify runs in a background thread, and camera mode gives every
        # browser connection its own analyzer over one shared identify client.
        self._lock = threading.Lock()
        self._calls = 0

    def next_name(self) -> str:
        """``00001-9f3a2c.jpg``: the call number in this process, then a random
        suffix. The number sorts the folder in the order the cards were shown;
        the suffix is what stops a second run (or a second process pointed at
        the same folder) overwriting the first one's images."""
        with self._lock:
            self._calls += 1
            number = self._calls
        return f"{number:05d}-{secrets.token_hex(3)}.jpg"

    def save_b64(self, b64: str) -> Path | None:
        """Write one record's ``_base64`` payload — the crop as sent.

        A no-op in ``frame`` mode, so the identify path can offer every record
        without asking what the store is set to. Returns the path, or None if
        nothing was written — a failure is logged, never raised.
        """
        if self.kind != OBJECT:
            return None
        try:
            return self._write(base64.b64decode(b64, validate=True))
        except (binascii.Error, ValueError) as exc:
            logger.warning("[store] undecodable base64, not saved: %s", exc)
            return None

    def save_frame(self, frame_bgr: np.ndarray) -> Path | None:
        """Write the whole frame a crop was cut from. A no-op in ``object``
        mode. Encoded at the identify quality, so the two modes are comparable
        pictures of the same moment."""
        if self.kind != FRAME:
            return None
        data = encode_jpeg(frame_bgr)
        if data is None:
            logger.warning("[store] frame did not encode, not saved")
            return None
        return self._write(data)

    def _write(self, data: bytes) -> Path | None:
        """The tail both modes share: next name, write, survive a failed write."""
        path = self.folder / self.next_name()
        try:
            path.write_bytes(data)
        except OSError as exc:
            logger.warning("[store] %s: %s", path, exc)
            return None
        logger.debug("[store] wrote %s (%d bytes)", path, len(data))
        return path
