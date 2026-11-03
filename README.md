# rpi-cam-style

Make rendered or simulated frames look like they were captured by a real camera.

The default profile is a **Raspberry Pi High Quality Camera** (Sony IMX477, 4056 × 3040, 1.55 µm pixels) with a **12 mm lens**. Other cameras are supported by changing the sensor size, focal length and output size.

It is meant for **domain adaptation**: simulator images (for example X-Plane screenshots) are turned into images that look like your camera's output, so a detector trained on them transfers better to real footage. Bounding-box labels in YOLO format are transformed with the same geometry, so a dataset can be converted without re-labeling.

## What it does

| Step | Effect | Parameters |
|---|---|---|
| Field of view | Crops / zooms the simulator frame to the camera's FOV (12 mm on IMX477 ≈ 29.4° horizontal, 16.8° vertical in 16:9) | `--focal-mm`, `--sensor-mm`, `--source-hfov`, `--out-size` |
| Lens distortion | Radial barrel/pincushion distortion | `--k1`, `--k2` |
| Exposure, white balance | EV offset, RGB gains | `--ev`, `--wb` |
| Tone and color | Saturation, contrast, black level | `--saturation`, `--contrast`, `--black-level` |
| Reference matching | Histogram match to real frames from your camera | `--reference`, `--match-strength`, `--ignore-top` |
| Atmosphere | Haze toward the top of the frame or over the whole frame, vignetting | `--haze`, `--haze-uniform`, `--haze-color`, `--vignette` |
| Optics | Softness, lateral chromatic aberration | `--blur`, `--chroma-aberration` |
| Sensor | Shot noise (Poisson) and read noise | `--photon-scale`, `--read-noise` |
| Output | 8-bit quantization, JPEG compression | `--jpeg-quality` |
| Variation | Random per-frame exposure, white balance, haze, softness, JPEG quality | `--jitter` |

## Install

Python 3.9+.

```bash
pip install numpy opencv-python
```

Clone the repository and run it from its folder, or install it with `pip install -e .`.

## Quick start

Stylize a folder of images:

```bash
python -m camstyle input_frames/ output_frames/ --source-hfov 60
```

Stylize a YOLO dataset (`images/` and `labels/` folders). Images are converted and labels are transformed; boxes that leave the frame are dropped:

```bash
python -m camstyle datasets/my_sim_set datasets/my_sim_set_cam --source-hfov 60 \
    --reference real_camera_samples/ --match-strength 0.3 --ignore-top 45
```

The output folder gets `camstyle_params.json` with the exact parameters used, so each dataset is reproducible.

## Important: simulator field of view

The simulator must render **wider** than the camera. The 12 mm camera needs about 29.4° horizontal. If the simulator FOV is narrower at the frame edges, the tool stops with an error.

- Set X-Plane (or your renderer) to a horizontal FOV of about **50–60°**. Then pass that value as `--source-hfov`.
- Use a **16:9** window or frame, and pass the same aspect with `--out-size 1280x720`.
- Labels that fall outside the 12 mm crop are dropped. Expect many drops if the simulator FOV is much wider than the camera's; this is normal.

## Parameters for your camera

All of these are command-line options. The defaults match the Raspberry Pi HQ camera with a 12 mm lens.

| Option | Default | Meaning |
|---|---|---|
| `--focal-mm` | 12 | Lens focal length (mm). Use the value printed on the lens. |
| `--sensor-mm` | 6.287,4.712 | Sensor width,height (mm). IMX477: 4056×1.55 µm and 3040×1.55 µm. |
| `--k1`, `--k2` | -0.02, 0 | Lens distortion. Negative = barrel. A 12 mm lens is usually weak, so -0.01 to -0.03. |
| `--source-hfov` | 60 | Simulator horizontal FOV in degrees. |
| `--out-size` | 1280x720 | Output size. Use your camera's mode. |
| `--ev` | 0 | Exposure offset in stops. |
| `--wb` | 1.03,1.0,0.95 | RGB white balance gains. |
| `--saturation` | 0.90 | 1 = unchanged, 0 = grayscale. |
| `--contrast` | 0.95 | 1 = unchanged. |
| `--black-level` | 0.02 | Lifts the shadows. |
| `--haze` | 0.06 | Haze at the top of the frame (the horizon side). |
| `--haze-color` | from reference, else warm grey | RGB 0..1, e.g. `0.8,0.77,0.72`. |
| `--vignette` | 0.18 | Corner darkening. |
| `--blur` | 0.6 | Softness sigma in pixels. 0 = sharp. |
| `--chroma-aberration` | 0.0008 | R/B radial offset. 0 = off. |
| `--photon-scale` | 1500 | Shot-noise scale. Lower = noisier. 0 = off. |
| `--read-noise` | 0.006 | Read noise sigma (0..1 range). |
| `--reference` | none | Folder of real camera frames. Used for histogram matching and haze color. |
| `--match-strength` | 0 | 0..1. Needs `--reference`. |
| `--ignore-top` | 50 | Rows at the top of reference frames to skip (overlay text). |
| `--jpeg-quality` | 90 | JPEG quality of the output. |
| `--seed` | 0 | Noise seed. The same seed gives the same output. |
| `--min-visible` | 0.3 | Minimum visible fraction of a label after the crop. |

### How to get good values

1. Take 5–10 real frames with the camera in the scene you care about (sky, ground, horizon). Include them in `--reference`. Crop out any overlay text from the top with `--ignore-top`.
2. Start with the defaults and `--match-strength 0.3`. Compare the output with a real frame side by side.
3. If the output is too washed out, lower `--haze` and `--match-strength`. If it is too dark, raise `--ev`. If there is too much color cast, change `--wb`.
4. Tune noise last: check the dark areas of a real frame and match `--photon-scale`.

Do not tune on one image. Check several frames.

### Recipe tuned on real Raspberry Pi HQ + 12 mm frames (hazy field footage)

Reference frames: hazy, warm, low contrast, soft video frames. This recipe gives output close to them
for X-Plane frames rendered at `--source-hfov 34`:

```bash
python -m camstyle SRC DST --focal-mm 12 --source-hfov 34 --out-size 1280x720 \
    --reference real_camera_samples/ --ignore-top 45 --match-strength 0.5 \
    --wb 1.05,1.0,0.93 --haze 0 --haze-uniform 0.07 --saturation 1.0 --contrast 0.95 \
    --blur 0.4 --photon-scale 1500 --read-noise 0.006 --jpeg-quality 70 --jitter 0.5
```

Notes:
- The simulator horizon is not at the top of the frame, so use `--haze-uniform` (whole frame) instead of `--haze`.
- `--match-strength` uses a smooth tone curve with a limited slope. It no longer makes stripes in clear sky.
- `--jitter 0.5` gives each frame a slightly different exposure, color, haze, softness and JPEG quality, so a detector does not overfit one look. Same `--seed` gives the same result.

## Using the Python API

```python
import cv2
from camstyle import CamStyle, CamStyleParams, load_reference

params = CamStyleParams(source_hfov_deg=60, out_width=1280, out_height=720, seed=1)
ref = load_reference(["real/frame1.jpg", "real/frame2.jpg"], ignore_top_px=45)
cs = CamStyle(params, ref)

frame = cv2.imread("sim.png")
out = cs.stylize(frame)                  # BGR uint8, 1280x720
cs.encode(out, "cam_like.jpg")

# labels: YOLO (cls, cx, cy, w, h), normalized to the source frame
boxes = [(0, 0.48, 0.52, 0.02, 0.01)]
new_boxes = cs.transform_yolo_boxes(boxes, frame.shape[0], frame.shape[1])
```

`CamStyle` keeps the geometry maps in a cache, so it is fast for many frames of the same size.

## Limits

- This is a **physically motivated approximation**. It does not model the real sensor's color matrix, demosaicing, ISP denoising or auto-exposure. Those are the parts that most change the look of a real camera.
- The haze model uses vertical position as a depth proxy. It does not know real scene depth.
- Reference matching learns global color and tone from your samples. If the references are all sky, the result will be too bright on the ground.
- Distortion is a simple radial model. The 12 mm lens is assumed to be a rectilinear lens.
- No real-camera validation is included. Check the output against your own frames before training on it.

## Tests

```bash
pip install pytest
python -m pytest -q tests
```

## Layout

```
camstyle/
  core.py    geometry, color, noise pipeline; label transform
  cli.py     command line interface
tests/
  test_camstyle.py
docs/
  index.html  project page
```

## Preview one image (input vs output)

```bash
python -m camstyle.preview path/to/sim_frame.png out/compare.jpg --source-hfov 60 \
    --reference real_camera_samples/ --match-strength 0.3 --ignore-top 45
```

Left: input, right: output. Generated files go into `out/` inside this repository (ignored by git).
