"""Command line interface.

    python -m camstyle IN_DIR OUT_DIR [options]

IN_DIR may be:
  * a plain folder of images           -> every image is stylized
  * a YOLO dataset (images/, labels/)  -> images/** are stylized and labels/**
                                          are transformed to the new geometry
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional

import cv2

from .core import CamStyle, CamStyleParams, load_reference

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _list_images(root: str) -> List[str]:
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(IMAGE_EXT):
                files.append(os.path.join(dp, fn))
    return sorted(files)


def _read_yolo(path: str):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5:
                rows.append((int(float(parts[0])), *map(float, parts[1:])))
    return rows


def _write_yolo(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for cls, cx, cy, w, h in rows:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def build_parser() -> argparse.ArgumentParser:
    d = CamStyleParams()
    ap = argparse.ArgumentParser(
        prog="camstyle",
        description="Make simulator frames look like they were shot with a real camera "
                    "(default profile: Raspberry Pi HQ camera, Sony IMX477, 12 mm lens).")
    ap.add_argument("input", help="input folder (images, or dataset with images/ and labels/)")
    ap.add_argument("output", help="output folder")

    g = ap.add_argument_group("camera / lens")
    g.add_argument("--focal-mm", type=float, default=d.focal_mm, help="lens focal length in mm")
    g.add_argument("--sensor-mm", type=str, default=f"{d.sensor_width_mm},{d.sensor_height_mm}",
                   help="sensor width,height in mm (IMX477 full array: 6.287,4.712)")
    g.add_argument("--k1", type=float, default=d.k1, help="radial distortion k1 (barrel < 0, pincushion > 0)")
    g.add_argument("--k2", type=float, default=d.k2, help="radial distortion k2")

    g = ap.add_argument_group("simulator and output size")
    g.add_argument("--source-hfov", type=float, default=d.source_hfov_deg,
                   help="horizontal FOV (deg) the simulator frames were rendered with. "
                        "Must be wider than the camera FOV.")
    g.add_argument("--out-size", type=str, default=f"{d.out_width}x{d.out_height}",
                   help="output size WxH, e.g. 1280x720 (use the camera's 16:9 mode)")

    g = ap.add_argument_group("exposure and color")
    g.add_argument("--ev", type=float, default=d.ev, help="exposure offset in stops")
    g.add_argument("--wb", type=str, default=",".join(map(str, d.white_balance)),
                   help="white balance RGB gains, e.g. 1.03,1.0,0.95")
    g.add_argument("--saturation", type=float, default=d.saturation)
    g.add_argument("--contrast", type=float, default=d.contrast)
    g.add_argument("--black-level", type=float, default=d.black_level)

    g = ap.add_argument_group("atmosphere")
    g.add_argument("--haze", type=float, default=d.haze, help="haze at the top of the frame (0..0.4)")
    g.add_argument("--haze-color", type=str, default=None,
                   help="haze RGB 0..1, e.g. 0.8,0.77,0.72 (default: from --reference, else warm grey)")
    g.add_argument("--vignette", type=float, default=d.vignette)

    g = ap.add_argument_group("optics and sensor")
    g.add_argument("--blur", type=float, default=d.blur_sigma, help="Gaussian softness sigma (px)")
    g.add_argument("--chroma-aberration", type=float, default=d.chroma_aberration)
    g.add_argument("--photon-scale", type=float, default=d.photon_scale,
                   help="shot-noise scale; lower = noisier (0 = off)")
    g.add_argument("--read-noise", type=float, default=d.read_noise)

    g = ap.add_argument_group("matching to real frames")
    g.add_argument("--reference", type=str, default=None,
                   help="folder of real camera frames (sample images from the camera)")
    g.add_argument("--match-strength", type=float, default=0.0,
                   help="0..1, histogram match to the reference colors (needs --reference)")
    g.add_argument("--ignore-top", type=int, default=d.ignore_top_px,
                   help="rows at the top of reference frames to ignore (overlay text)")

    g = ap.add_argument_group("output")
    g.add_argument("--jpeg-quality", type=int, default=d.jpeg_quality)
    g.add_argument("--seed", type=int, default=d.seed, help="random seed for noise (reproducible)")
    g.add_argument("--min-visible", type=float, default=0.3,
                   help="drop labels with less than this fraction visible after the crop")
    return ap


def params_from_args(a: argparse.Namespace) -> CamStyleParams:
    sw, sh = (float(x) for x in a.sensor_mm.split(","))
    ow, oh = (int(x) for x in a.out_size.lower().split("x"))
    wb = tuple(float(x) for x in a.wb.split(","))
    if len(wb) != 3:
        raise SystemExit("--wb needs three values, e.g. 1.03,1.0,0.95")
    haze = None
    if a.haze_color:
        haze = tuple(float(x) for x in a.haze_color.split(","))
    return CamStyleParams(
        sensor_width_mm=sw, sensor_height_mm=sh, focal_mm=a.focal_mm,
        k1=a.k1, k2=a.k2, source_hfov_deg=a.source_hfov,
        out_width=ow, out_height=oh, ev=a.ev, white_balance=wb,
        saturation=a.saturation, contrast=a.contrast, black_level=a.black_level,
        haze=a.haze, haze_color=haze, vignette=a.vignette,
        blur_sigma=a.blur, chroma_aberration=a.chroma_aberration,
        photon_scale=a.photon_scale, read_noise=a.read_noise,
        match_strength=a.match_strength, ignore_top_px=a.ignore_top,
        jpeg_quality=a.jpeg_quality, seed=a.seed)


def run(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    p = params_from_args(a)

    ref = None
    if a.reference:
        ref_files = _list_images(a.reference)[:300]
        if not ref_files:
            raise SystemExit(f"no images in --reference {a.reference}")
        ref = load_reference(ref_files, ignore_top_px=p.ignore_top_px)
        print(f"reference: {len(ref_files)} frames")
    elif a.match_strength > 0:
        raise SystemExit("--match-strength needs --reference")

    cs = CamStyle(p, ref)
    print(f"camera hfov {p.camera_hfov_deg:.2f} deg, out vfov {p.camera_vfov_deg:.2f} deg, "
          f"source hfov {p.source_hfov_deg} deg")

    is_dataset = os.path.isdir(os.path.join(a.input, "images"))
    img_root = os.path.join(a.input, "images") if is_dataset else a.input
    images = _list_images(img_root)
    if not images:
        raise SystemExit(f"no images found in {img_root}")

    os.makedirs(a.output, exist_ok=True)
    with open(os.path.join(a.output, "camstyle_params.json"), "w") as f:
        json.dump(p.to_dict(), f, indent=2)

    kept_boxes = dropped = 0
    for n, src in enumerate(images, 1):
        rel = os.path.relpath(src, img_root)
        dst = os.path.join(a.output, "images" if is_dataset else "", rel)
        dst = os.path.splitext(dst)[0] + ".jpg"
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        bgr = cv2.imread(src, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"skip unreadable {src}", file=sys.stderr)
            continue
        out = cs.stylize(bgr)
        cs.encode(out, dst)

        if is_dataset:
            lbl_in = os.path.join(a.input, "labels", os.path.splitext(rel)[0] + ".txt")
            rows = _read_yolo(lbl_in)
            new_rows = cs.transform_yolo_boxes(rows, bgr.shape[0], bgr.shape[1],
                                               min_visible=a.min_visible)
            dropped += len(rows) - len(new_rows)
            kept_boxes += len(new_rows)
            lbl_out = os.path.join(a.output, "labels", os.path.splitext(rel)[0] + ".txt")
            _write_yolo(lbl_out, new_rows)
        if n % 50 == 0:
            print(f"{n}/{len(images)}")

    print(f"done: {len(images)} images -> {a.output}")
    if is_dataset:
        print(f"labels kept {kept_boxes}, dropped (out of frame) {dropped}")
    return 0


def main() -> None:
    sys.exit(run())
