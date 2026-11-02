"""Core transform: make a rendered / simulated frame look like a real camera frame.

Pipeline (all steps optional, controlled by CamStyleParams):

1. Geometry   : match the field of view of the simulator to the real lens
                (crop / zoom), apply radial lens distortion.
2. Exposure   : EV offset, per-channel white balance.
3. Tone/color : saturation, contrast, black level, optional histogram matching
                to a folder of real camera frames.
4. Atmosphere : distance haze (stronger toward the top of the frame, i.e. toward
                the horizon), vignetting.
5. Optics     : softness (Gaussian blur), lateral chromatic aberration.
6. Sensor     : shot noise (Poisson) + read noise.
7. Output     : 8-bit quantization and JPEG compression.

Bounding boxes in YOLO format are transformed with the same geometry, so a
dataset can be stylized without re-labeling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class CamStyleParams:
    # --- camera / lens ---------------------------------------------------
    sensor_width_mm: float = 6.287      # Sony IMX477: 4056 px * 1.55 um
    sensor_height_mm: float = 4.712     # 3040 px * 1.55 um (full 4:3 array)
    focal_mm: float = 12.0              # lens focal length
    k1: float = -0.02                   # radial distortion (negative = barrel)
    k2: float = 0.0
    # --- simulator side --------------------------------------------------
    source_hfov_deg: float = 60.0       # horizontal FOV the simulator renders with
    out_width: int = 1280
    out_height: int = 720
    # --- exposure / color ------------------------------------------------
    ev: float = 0.0                     # exposure offset in stops
    white_balance: Tuple[float, float, float] = (1.03, 1.0, 0.95)  # RGB gains
    saturation: float = 0.90            # 1 = unchanged, 0 = grayscale
    contrast: float = 0.95              # 1 = unchanged
    black_level: float = 0.02           # lifts shadows (0..0.2 typical)
    # --- atmosphere ------------------------------------------------------
    haze: float = 0.06                  # max haze amount at the top of the frame
    haze_color: Optional[Tuple[float, float, float]] = None  # None = from references, else warm grey
    vignette: float = 0.18              # corner darkening (0..0.5 typical)
    # --- optics ----------------------------------------------------------
    blur_sigma: float = 0.6             # softness in output pixels (0 = off)
    chroma_aberration: float = 0.0008   # R/B radial scale offset (0 = off)
    # --- sensor ----------------------------------------------------------
    photon_scale: float = 1500.0        # higher = less shot noise
    read_noise: float = 0.006           # gaussian sigma, 0..1 range
    # --- reference matching (optional) -----------------------------------
    match_strength: float = 0.0         # 0 = off, 1 = full histogram match
    ignore_top_px: int = 50             # overlay band to ignore in references
    # --- output ----------------------------------------------------------
    jpeg_quality: int = 90
    seed: Optional[int] = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def camera_hfov_deg(self) -> float:
        """Horizontal field of view of the real lens on the real sensor."""
        return math.degrees(2 * math.atan(self.sensor_width_mm / (2 * self.focal_mm)))

    @property
    def camera_vfov_deg(self) -> float:
        """Vertical field of view of the output frame (aspect from out_width/out_height)."""
        f = self._out_focal_px()
        return math.degrees(2 * math.atan((self.out_height / 2) / f))

    def _out_focal_px(self) -> float:
        return (self.out_width / 2) / math.tan(math.radians(self.camera_hfov_deg) / 2)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class Geometry:
    """Maps output pixels to source pixels and back, for a given source size."""

    def __init__(self, src_h: int, src_w: int, p: CamStyleParams):
        self.src_h, self.src_w = src_h, src_w
        self.p = p
        self.f_out = p._out_focal_px()
        src_half = math.radians(p.source_hfov_deg) / 2
        self.f_src = (src_w / 2) / math.tan(src_half)
        self.cx_out = (p.out_width - 1) / 2
        self.cy_out = (p.out_height - 1) / 2
        self.cx_src = (src_w - 1) / 2
        self.cy_src = (src_h - 1) / 2

    def _check_coverage(self, map_x: np.ndarray, map_y: np.ndarray):
        # Output pixels that would need source pixels outside the frame.
        out_of = ((map_x < -1) | (map_x > self.src_w) | (map_y < -1) | (map_y > self.src_h))
        if out_of.mean() > 0.005:
            raise ValueError(
                f"Source FOV ({self.p.source_hfov_deg} deg) is narrower than the camera FOV "
                f"({self.p.camera_hfov_deg:.2f} deg) at the frame edges. Render the simulator "
                f"with a wider FOV (about {self.p.camera_hfov_deg + 4:.0f} deg or more).")

    def _distort(self, xn, yn):
        r2 = xn * xn + yn * yn
        d = 1 + self.p.k1 * r2 + self.p.k2 * r2 * r2
        return xn * d, yn * d

    def maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """float32 maps for cv2.remap: for each output pixel, the source pixel to sample."""
        u = np.arange(self.p.out_width, dtype=np.float32)
        v = np.arange(self.p.out_height, dtype=np.float32)
        xn = (u - self.cx_out) / self.f_out
        yn = (v - self.cy_out) / self.f_out
        XN, YN = np.meshgrid(xn, yn)
        XD, YD = self._distort(XN, YN)
        map_x = (self.cx_src + self.f_src * XD).astype(np.float32)
        map_y = (self.cy_src + self.f_src * YD).astype(np.float32)
        self._check_coverage(map_x, map_y)
        return map_x, map_y

    def src_to_out(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse mapping: source pixel coordinates -> output pixel coordinates."""
        xd = (np.asarray(xs, dtype=np.float64) - self.cx_src) / self.f_src
        yd = (np.asarray(ys, dtype=np.float64) - self.cy_src) / self.f_src
        xn, yn = xd.copy(), yd.copy()
        # Fixed-point iteration: xd = xn * (1 + k1 r^2 + k2 r^4)
        for _ in range(12):
            r2 = xn * xn + yn * yn
            d = 1 + self.p.k1 * r2 + self.p.k2 * r2 * r2
            xn, yn = xd / d, yd / d
        return self.cx_out + self.f_out * xn, self.cy_out + self.f_out * yn


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _histogram_match(img: np.ndarray, ref_channels: np.ndarray, strength: float) -> np.ndarray:
    """Per-channel quantile mapping of img (float 0..1, HxWx3 RGB) onto reference quantiles.

    ref_channels: (3, Q) array of reference quantile values per RGB channel.
    """
    q = np.linspace(0, 1, ref_channels.shape[1])
    out = np.empty_like(img)
    for c in range(3):
        src_q = np.quantile(img[..., c], q)
        # make source quantiles strictly increasing so np.interp is well defined
        src_q = src_q + np.arange(len(src_q)) * 1e-9
        mapped = np.interp(img[..., c], src_q, ref_channels[c])
        out[..., c] = mapped
    return (1 - strength) * img + strength * out


def compute_reference_stats(paths: Sequence[str], ignore_top_px: int = 50, q_count: int = 256):
    """Builds per-channel RGB quantiles and a haze color from real camera frames.

    Returns (quantiles (3, Q) float in 0..1, haze_color (3,) RGB in 0..1).
    """
    chans = [[], [], []]
    top_means = []
    for path in paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        body = rgb[ignore_top_px:]
        step = max(1, body.shape[0] * body.shape[1] // 200_000)  # subsample for speed
        flat = body.reshape(-1, 3)[::step]
        for c in range(3):
            chans[c].append(flat[:, c])
        top_band = rgb[ignore_top_px:ignore_top_px + max(1, rgb.shape[0] // 6)]
        top_means.append(top_band.reshape(-1, 3).mean(axis=0))
    if not chans[0]:
        raise ValueError("No readable reference images found")
    q = np.linspace(0, 1, q_count)
    quant = np.stack([np.quantile(np.concatenate(chans[c]), q) for c in range(3)])
    haze = np.mean(top_means, axis=0)
    return quant.astype(np.float32), haze.astype(np.float32)


@dataclass
class RefStats:
    quantiles: Optional[np.ndarray] = None   # (3, Q)
    haze_color: Optional[Tuple[float, float, float]] = None


# ---------------------------------------------------------------------------
# Main stylize function
# ---------------------------------------------------------------------------

class CamStyle:
    """Stylize frames and boxes. Create once, reuse for many frames of the same size."""

    def __init__(self, params: Optional[CamStyleParams] = None, reference: Optional[RefStats] = None):
        self.p = params or CamStyleParams()
        self.ref = reference or RefStats()
        # haze color: explicit param > reference frames > default warm grey
        if self.p.haze_color is None:
            if self.ref.haze_color is not None:
                self.p.haze_color = tuple(float(x) for x in self.ref.haze_color)
            else:
                self.p.haze_color = (0.80, 0.77, 0.72)
        self._rng = np.random.default_rng(self.p.seed)
        self._cache_key = None
        self._cache = None

    # -- geometry cache -----------------------------------------------------
    def geometry(self, src_h: int, src_w: int) -> Geometry:
        return Geometry(src_h, src_w, self.p)

    def _maps_for(self, src_h: int, src_w: int):
        key = (src_h, src_w)
        if self._cache_key != key:
            g = self.geometry(src_h, src_w)
            mx, my = g.maps()
            # precompute vignette and haze gradients for the output size
            H, W = self.p.out_height, self.p.out_width
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            xn = (xx - (W - 1) / 2) / (W / 2)
            yn = (yy - (H - 1) / 2) / (H / 2)
            rnorm = np.sqrt(xn * xn + yn * yn) / math.sqrt(2)
            vign = 1.0 - self.p.vignette * (rnorm ** 2)
            top = 1.0 - (yy / max(1, H - 1))          # 1 at top, 0 at bottom
            haze = self.p.haze * (top ** 1.5)
            self._cache = dict(geom=g, mx=mx, my=my, vign=vign.astype(np.float32),
                               haze=haze.astype(np.float32))
            self._cache_key = key
        return self._cache

    # -- frame --------------------------------------------------------------
    def stylize(self, bgr: np.ndarray) -> np.ndarray:
        """bgr uint8 HxWx3 -> bgr uint8 (out_height x out_width x 3)."""
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("expected HxWx3 BGR image")
        p = self.p
        c = self._maps_for(bgr.shape[0], bgr.shape[1])

        # 1. geometry: crop/zoom to the camera FOV + lens distortion
        img = cv2.remap(bgr, c["mx"], c["my"], interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 2. exposure + white balance
        rgb *= 2.0 ** p.ev
        rgb *= np.asarray(p.white_balance, dtype=np.float32)

        # 3. saturation, contrast, black level
        lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])[..., None]
        rgb = lum + (rgb - lum) * p.saturation
        rgb = 0.45 + (rgb - 0.45) * p.contrast
        rgb = p.black_level + (1 - p.black_level) * rgb

        # 4. optional histogram match to real camera frames
        if self.ref.quantiles is not None and p.match_strength > 0:
            rgb = _histogram_match(np.clip(rgb, 0, 1), self.ref.quantiles, p.match_strength)

        # 5. haze (toward horizon) and vignette
        haze_col = np.asarray(p.haze_color, dtype=np.float32)  # set in __init__
        rgb = rgb * (1 - c["haze"][..., None]) + haze_col * c["haze"][..., None]
        rgb *= c["vign"][..., None]

        # 6. optics: chromatic aberration, softness
        rgb = np.clip(rgb, 0, 1)
        if abs(p.chroma_aberration) > 0:
            rgb = self._chroma(rgb, p.chroma_aberration)
        if p.blur_sigma > 0:
            rgb = cv2.GaussianBlur(rgb, (0, 0), p.blur_sigma)

        # 7. sensor noise
        rgb = np.clip(rgb, 0, 1)
        if p.photon_scale > 0:
            shot = self._rng.poisson(rgb * p.photon_scale).astype(np.float32) / p.photon_scale
            rgb = shot
        if p.read_noise > 0:
            rgb = rgb + self._rng.normal(0, p.read_noise, rgb.shape).astype(np.float32)

        # 8. quantize to 8 bit
        out8 = np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
        return cv2.cvtColor(out8, cv2.COLOR_RGB2BGR)

    def _chroma(self, rgb: np.ndarray, ca: float) -> np.ndarray:
        H, W = rgb.shape[:2]
        cx, cy = (W - 1) / 2, (H - 1) / 2
        out = rgb.copy()
        for ch, s in ((0, 1 + ca), (2, 1 - ca)):
            M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
            out[..., ch] = cv2.warpAffine(rgb[..., ch], M, (W, H),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return out

    def encode(self, bgr: np.ndarray, path: str) -> None:
        """Write the frame; .jpg uses p.jpeg_quality, other extensions use lossless writers."""
        if path.lower().endswith((".jpg", ".jpeg")):
            ok = cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, int(self.p.jpeg_quality)])
        else:
            ok = cv2.imwrite(path, bgr)
        if not ok:
            raise IOError(f"could not write {path}")

    # -- labels -------------------------------------------------------------
    def transform_yolo_boxes(self, boxes: Iterable[Sequence[float]], src_h: int, src_w: int,
                             min_visible: float = 0.3):
        """Transform YOLO boxes (cls, cx, cy, w, h; normalized to the source frame).

        Returns a list of (cls, cx, cy, w, h) normalized to the output frame.
        Boxes that are less than `min_visible` inside the output frame are dropped.
        """
        g = self.geometry(src_h, src_w)
        out = []
        for cls, cx, cy, w, h in boxes:
            x1, x2 = (cx - w / 2) * src_w, (cx + w / 2) * src_w
            y1, y2 = (cy - h / 2) * src_h, (cy + h / 2) * src_h
            # sample a 3x3 grid inside the box so that distortion is handled
            gx, gy = np.meshgrid(np.linspace(x1, x2, 3), np.linspace(y1, y2, 3))
            ox, oy = g.src_to_out(gx.ravel(), gy.ravel())
            bx1, bx2 = float(ox.min()), float(ox.max())
            by1, by2 = float(oy.min()), float(oy.max())
            area_full = (bx2 - bx1) * (by2 - by1)
            cx1, cx2 = max(0.0, bx1), min(float(self.p.out_width), bx2)
            cy1, cy2 = max(0.0, by1), min(float(self.p.out_height), by2)
            if cx2 <= cx1 or cy2 <= cy1 or area_full <= 0:
                continue
            visible = ((cx2 - cx1) * (cy2 - cy1)) / area_full
            if visible < min_visible:
                continue
            W, H = self.p.out_width, self.p.out_height
            out.append((int(cls), (cx1 + cx2) / 2 / W, (cy1 + cy2) / 2 / H,
                        (cx2 - cx1) / W, (cy2 - cy1) / H))
        return out


def load_reference(paths: Sequence[str], ignore_top_px: int = 50) -> RefStats:
    quant, haze = compute_reference_stats(paths, ignore_top_px=ignore_top_px)
    return RefStats(quantiles=quant, haze_color=tuple(float(x) for x in haze))
