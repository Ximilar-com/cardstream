"""How sharp a card crop is — the one measurement behind ``--min-sharpness``.

The motion gate says the SCENE has settled; it cannot say the card is in focus.
A card that has just entered the frame sits still while the camera is still
hunting, the crop that goes out is a blur, and the endpoint answers with a
confident wrong card — and then a second paid call once the picture clears.
This scores the cut-out card itself, so the call can be held until there is
something worth paying for.

The score is a re-blur ratio (Crete et al., 2007): blur the crop once more and
see how much of its neighbour-to-neighbour contrast that removes. A sharp crop
loses most of it; one that is already blurred has little left to lose. Being a
ratio of the crop against ITSELF is the point. Variance of the Laplacian — the
usual answer — moves about 2x with the artwork and again with exposure and crop
size, so no single threshold holds across cards. Measured on real crops this
one sits at 0.75-0.81 for a card in focus whatever is printed on it, and at
0.46-0.63 for the out-of-focus ones that used to be sent.

What it does not do: sensor noise is fine detail too, so a very noisy picture
scores sharper than it is. The gate then lets more through rather than less —
it fails toward the old behaviour, never toward a card that cannot be named.
"""

from __future__ import annotations

import cv2
import numpy as np

from cardstream.core.imaging import fit_long_edge

# The crop is judged at this size, not at the camera's. Blur that does not
# survive the shrink is not blur the endpoint will see either, and it keeps the
# score — and so the threshold — the same at 720p and at 4K. Shrink-only: a
# small crop is measured as it is rather than marked down for being enlarged.
_LONG_EDGE = 256
# Width of the second blur, in pixels of that 256 px crop. Wide enough to wipe
# out a sharp card's fine print, so the ratio has room to fall as the crop's
# own blur approaches it.
_REBLUR_TAPS = 9


def sharpness(crop_bgr: np.ndarray) -> float:
    """0..1, higher is sharper; 0.0 for a crop with nothing in it to blur.

    The WORSE of the two axes: a card smeared sideways by a moving hand is
    still crisp top to bottom, and an average would let it through.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # float32 before any differencing — uint8 would wrap instead of going
    # negative, and every dark-to-light step would read as a bright one.
    small = fit_long_edge(gray, _LONG_EDGE).astype(np.float32)
    return min(_detail_lost(small, axis) for axis in (0, 1))


def _detail_lost(gray: np.ndarray, axis: int) -> float:
    """The share of pixel-to-pixel contrast along ``axis`` a further blur removes."""
    # cv2 kernel sizes are (width, height): axis 0 runs down the rows.
    ksize = (1, _REBLUR_TAPS) if axis == 0 else (_REBLUR_TAPS, 1)
    blurred = cv2.blur(gray, ksize)
    before = np.abs(np.diff(gray, axis=axis))
    after = np.abs(np.diff(blurred, axis=axis))
    total = float(before.sum())
    if total <= 0.0:
        # Featureless (or one pixel wide): nothing to lose is not "sharp".
        return 0.0
    return float(np.maximum(before - after, 0.0).sum()) / total
