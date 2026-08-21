"""Ximilar identification client — the one path a crop takes out of here.

The client does detection and the identity gate locally, so nothing sits
between it and the id endpoint: it POSTs the crop straight to
``collectibles/v2/{tcg,sport,slab,comics}_id`` using ``XIMILAR_API_KEY``.

The call itself is :class:`~cardstream.core.identify_client.XimilarIdentifier`
in ``core``, which is where it stays: the wire format and the flattened
identification dict are a property of the endpoint, not of this caller.
"""

from __future__ import annotations

import numpy as np

from cardstream.client.identify_target import IdentifyTarget
from cardstream.core.identify_client import DEFAULT_HTTP_TIMEOUT, XimilarIdentifier
from cardstream.core.identify_options import IdentifyOptions
from cardstream.core.image_store import ImageStore
from cardstream.core.ximilar import DEFAULT_TIERS, TierThresholds


class DirectXimilarClient(IdentifyTarget):
    """Straight to the id endpoint with our own API key."""

    def __init__(
        self,
        api_key: str,
        options: IdentifyOptions | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        tiers: TierThresholds = DEFAULT_TIERS,
        store: ImageStore | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "identification needs a Ximilar API key — set XIMILAR_API_KEY or "
                "pass --api-key"
            )
        super().__init__(options, timeout)
        self._identifier = XimilarIdentifier(api_key, timeout, tiers, store)

    def identify(self, crop_bgr: np.ndarray) -> dict | None:
        # Read once: the settings dialog may rebind .options mid-call.
        ident = self._identifier.identify_crop(crop_bgr, self.options)
        return ident.to_dict() if ident is not None else None
