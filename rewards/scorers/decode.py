"""Frame decoding for the reward scorers."""

def decode_uniform(path: str, n: int):
    """``n`` uniformly-spaced frames as RGB [n, H, W, 3], at the clip's own aspect ratio.

    VGGT's balanced preprocessing handles non-square input natively.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    raw = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        raw.append(f)
    cap.release()
    if not raw:
        return None
    idxs = np.linspace(0, len(raw) - 1, min(n, len(raw))).round().astype(int)
    return np.stack([cv2.cvtColor(raw[i], cv2.COLOR_BGR2RGB) for i in idxs], 0)
