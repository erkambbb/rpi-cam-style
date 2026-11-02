"""Side-by-side preview: input frame (left) vs camstyle output (right).

    python -m camstyle.preview INPUT.jpg OUTPUT_COMPARE.jpg [same options as the CLI]
"""
import sys

import cv2
import numpy as np

from .cli import build_parser, params_from_args, _read_yolo  # noqa: F401
from .core import CamStyle, load_reference
from .cli import _list_images


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        raise SystemExit("usage: python -m camstyle.preview INPUT.jpg COMPARE.jpg [options]")
    src, dst, rest = argv[0], argv[1], argv[2:]
    a = build_parser().parse_args(["_", "_"] + rest)
    p = params_from_args(a)
    ref = load_reference(_list_images(a.reference)[:300], ignore_top_px=p.ignore_top_px) if a.reference else None
    bgr = cv2.imread(src, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {src}")
    out = CamStyle(p, ref).stylize(bgr)
    H = 540
    left = cv2.resize(bgr, (int(bgr.shape[1] * H / bgr.shape[0]), H))
    right = cv2.resize(out, (int(out.shape[1] * H / out.shape[0]), H))
    for img, label in ((left, "INPUT"), (right, "CAMSTYLE")):
        cv2.putText(img, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imwrite(dst, np.hstack([left, np.full((H, 8, 3), 255, np.uint8), right]))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
