import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from camstyle import CamStyle, CamStyleParams, load_reference  # noqa: E402


def _frame(h=720, w=1280, seed=1):
    rng = np.random.default_rng(seed)
    base = np.linspace(40, 200, h, dtype=np.float32)[:, None, None] * np.ones((1, w, 3), np.float32)
    base += rng.normal(0, 8, (h, w, 3))
    return np.clip(base, 0, 255).astype(np.uint8)


def test_camera_fov_12mm_imx477():
    p = CamStyleParams(focal_mm=12.0)
    assert abs(p.camera_hfov_deg - 29.36) < 0.05
    assert 15.5 < p.camera_vfov_deg < 17.5  # 16:9 output


def test_output_shape_and_dtype():
    p = CamStyleParams(source_hfov_deg=60, out_width=640, out_height=360, seed=0)
    out = CamStyle(p).stylize(_frame(720, 1280))
    assert out.shape == (360, 640, 3) and out.dtype == np.uint8


def test_narrow_source_fov_is_rejected():
    p = CamStyleParams(source_hfov_deg=20.0)
    with pytest.raises(ValueError):
        CamStyle(p).stylize(_frame())


def test_deterministic_with_seed():
    p = CamStyleParams(source_hfov_deg=60, seed=7)
    a = CamStyle(p).stylize(_frame())
    b = CamStyle(CamStyleParams(source_hfov_deg=60, seed=7)).stylize(_frame())
    assert np.array_equal(a, b)


def test_matched_fov_is_identity_geometry_without_distortion():
    p = CamStyleParams(source_hfov_deg=29.36, k1=0.0, out_width=1280, out_height=720)
    p.source_hfov_deg = p.camera_hfov_deg
    from camstyle.core import Geometry
    g = Geometry(720, 1280, p)
    xs, ys = np.array([100.0, 640.0, 1200.0]), np.array([50.0, 360.0, 700.0])
    ox, oy = g.src_to_out(xs, ys)
    assert np.allclose(ox, xs, atol=1e-3) and np.allclose(oy, ys, atol=1e-3)


def test_inverse_mapping_matches_forward_maps():
    from camstyle.core import Geometry
    p = CamStyleParams(source_hfov_deg=60, k1=-0.08, out_width=640, out_height=360)
    g = Geometry(720, 1280, p)
    mx, my = g.maps()
    # pick output pixels, take their source positions, invert back
    u = np.array([10.0, 320.0, 600.0, 200.0])
    v = np.array([5.0, 180.0, 350.0, 300.0])
    sx = mx[v.astype(int), u.astype(int)].astype(float)
    sy = my[v.astype(int), u.astype(int)].astype(float)
    ox, oy = g.src_to_out(sx, sy)
    assert np.allclose(ox, u, atol=0.05) and np.allclose(oy, v, atol=0.05)


def test_boxes_follow_geometry_and_scale_up():
    # source 60 deg, camera 28.6 deg -> center crop zooms in by ~2x horizontally
    p = CamStyleParams(source_hfov_deg=60, k1=0.0, out_width=1280, out_height=720)
    cs = CamStyle(p)
    box = (0, 0.5, 0.5, 0.02, 0.02)  # tiny box at the center
    out = cs.transform_yolo_boxes([box], 720, 1280)
    assert len(out) == 1
    _, cx, cy, w, h = out[0]
    assert abs(cx - 0.5) < 0.01 and abs(cy - 0.5) < 0.01
    assert w > 0.02 * 1.5  # zoomed in


def test_box_outside_crop_is_dropped():
    p = CamStyleParams(source_hfov_deg=60, k1=0.0, out_width=1280, out_height=720)
    cs = CamStyle(p)
    out = cs.transform_yolo_boxes([(0, 0.02, 0.5, 0.02, 0.02)], 720, 1280)
    assert out == []


def test_reference_histogram_match_pulls_colors(tmp_path):
    # reference frames that are much darker and bluer than the source
    for i in range(3):
        ref = np.zeros((200, 300, 3), np.uint8)
        ref[..., 0] = 90 + i   # B
        ref[..., 1] = 70       # G
        ref[..., 2] = 60       # R
        cv2.imwrite(str(tmp_path / f"r{i}.png"), ref)
    import glob
    ref = load_reference(sorted(glob.glob(str(tmp_path / "*.png"))), ignore_top_px=0)
    src = _frame()
    base = CamStyle(CamStyleParams(source_hfov_deg=60, seed=0, match_strength=0.0)).stylize(src)
    matched = CamStyle(CamStyleParams(source_hfov_deg=60, seed=0, match_strength=1.0), ref).stylize(src)
    assert matched.mean() < base.mean()


def test_encode_jpeg(tmp_path):
    cs = CamStyle(CamStyleParams(source_hfov_deg=60))
    out = cs.stylize(_frame())
    path = str(tmp_path / "x.jpg")
    cs.encode(out, path)
    assert os.path.getsize(path) > 0
