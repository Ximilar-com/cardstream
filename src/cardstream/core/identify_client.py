"""One paid identification call, start to finish.

``upscale_small`` → base64 → build the record → POST → parse the best match,
plus the auth headers; the ``{"records": [...]}`` envelope itself is
:meth:`IdentifyOptions.payload`, so a request-level flag lives with the rest
of the options. The client's
``DirectXimilarClient`` is a thin wrapper over this; where the options come
from and what type comes back are parameters, so a second caller never has to
copy the sequence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cardstream.core.identify_options import IdentifyOptions
from cardstream.core.image_store import ImageStore
from cardstream.core.imaging import encode_jpeg_b64, upscale_small
from cardstream.core.models import Identification
from cardstream.core.ximilar import (
    DEFAULT_TIERS,
    TierThresholds,
    full_image_card_object,
    parse_best_match,
    post_json,
)

# One default for every Ximilar call. The four call sites used to carry
# 15/15/20/20 for no stated reason.
DEFAULT_HTTP_TIMEOUT = 20.0

# How this client identifies itself to Ximilar. Every call carries it, so
# traffic from cardstream is distinguishable from a bare SDK or curl at the
# other end without anything else having to be passed along.
USER_AGENT = "CardStream"


def auth_headers(api_key: str) -> dict[str, str]:
    """The headers every Ximilar call carries: auth, content type, user agent.

    THE one place they are built — ``XimilarIdentifier`` holds the result for
    its lifetime and hands it to ``post_json``, so adding a header here reaches
    every request without touching a call site.
    """
    return {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }


class XimilarIdentifier:
    """Crop (or frame + objects) in, :class:`Identification` or None out.

    Stateless ``requests.post`` per call on purpose — Ximilar drops idle
    keep-alive sockets, so a Session buys nothing.
    """

    def __init__(
        self,
        api_key: str,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        tiers: TierThresholds = DEFAULT_TIERS,
        store: ImageStore | None = None,
    ) -> None:
        self._headers = auth_headers(api_key)
        self._timeout = timeout
        self._tiers = tiers
        # --store-images; None = keep nothing, which is the default.
        self._store = store

    def identify_crop(
        self, crop_bgr: np.ndarray, opts: IdentifyOptions
    ) -> Identification | None:
        """Identify a pre-cropped card.

        Upscales tiny crops first — small low-res cards make the id endpoints
        error more and match worse — and always sends one synthetic ``_objects``
        entry covering the whole image so the endpoint reuses our box instead
        of re-running its own detection.
        """
        image = upscale_small(crop_bgr)
        b64 = encode_jpeg_b64(image)
        if b64 is None:
            return None
        h, w = image.shape[:2]
        return self._post(opts.record(b64, [full_image_card_object(w, h)]), opts)

    def identify_frame(
        self,
        frame_bgr: np.ndarray,
        objects: list[dict[str, Any]],
        opts: IdentifyOptions,
    ) -> Identification | None:
        """Identify from the FULL frame plus boxes we already detected, so the
        endpoint reuses them instead of detecting again."""
        b64 = encode_jpeg_b64(frame_bgr)
        if b64 is None:
            return None
        return self._post(opts.record(b64, objects), opts)

    def _post(
        self, record: dict[str, Any], opts: IdentifyOptions
    ) -> Identification | None:
        # Before the POST, not after: an image whose call failed or matched
        # nothing is the one worth having on disk.
        if self._store is not None:
            self._store.save_b64(record["_base64"])
        body = post_json(
            opts.id_type.url,
            opts.payload(record),
            self._headers,
            self._timeout,
            tag=opts.id_type.key,
        )
        if body is None:
            return None
        return parse_best_match(body, self._tiers)
