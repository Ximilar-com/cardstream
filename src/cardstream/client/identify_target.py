"""Where the smart client sends one crop per distinct card.

One implementation today —
:class:`~cardstream.client.ximilar_api.DirectXimilarClient`, straight to
Ximilar with our own key. The ABC stays because it owns the part that is NOT
about transport: the mutable ``options`` bundle the settings dialog rebinds
mid-session, and the ``id_type`` property/setter that keeps a patched category
consistent. A subclass supplies :meth:`identify` and inherits all of that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from cardstream.core.identify_client import DEFAULT_HTTP_TIMEOUT
from cardstream.core.identify_options import IdentifyOptions


class IdentifyTarget(ABC):
    """One crop in, one flattened identification dict (or None) out."""

    def __init__(
        self,
        options: IdentifyOptions | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
    ) -> None:
        # Rebindable, not mutable: the settings dialog swaps the whole bundle
        # with ``target.options = target.options.with_(**patch)``, so a partly
        # applied change can never be observed by an in-flight identify.
        self.options = options if options is not None else IdentifyOptions()
        self._timeout = timeout

    @abstractmethod
    def identify(self, crop_bgr: np.ndarray) -> dict | None:
        """Identify one card crop. None = no match, or the call failed."""
