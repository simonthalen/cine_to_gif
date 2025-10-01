
"""
dicom_cine_to_gif.py

Create an animated GIF from a time-resolved DICOM cine.

"""

import os
import glob
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import imageio.v2 as imageio

def _to_uint8(img16, invert=False):
    """Percentile-based normalization to uint8 with optional inversion."""
    img = img16.astype(np.float32)
    lo, hi = np.percentile(img, (1, 99))
    if hi <= lo:
        hi = lo + 1.0
    img = np.clip((img - lo) / (hi - lo), 0, 1)
    if invert:
        img = 1.0 - img
    return (img * 255.0 + 0.5).astype(np.uint8)

def _read_multiframe(ds):
    """Return frames [N,H,W] uint8 and duration per frame in seconds."""
    arr = ds.pixel_array  # [N,H,W] or [H,W,N] depending on handler; pydicom -> [N,H,W]
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    # Apply VOI LUT if present (handles window center/width)
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    invert = str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1"
    frames_u8 = np.stack([_to_uint8(f, invert=invert) for f in arr], axis=0)

    # Frame timing
    frame_time_ms = ds.get("FrameTime", None)  # (0018,1063)
    cine_rate = ds.get("CineRate", None)      # (0018,0040) frames per second
    if frame_time_ms and frame_time_ms > 0:
        dt = float(frame_time_ms) / 1000.0
    elif cine_rate and cine_rate > 0:
        dt = 1.0 / float(cine_rate)
    else:
        dt = 1.0 / 25.0  # default 25 fps
    return frames_u8, dt

def _read_series(folder, fallback_fps=25.0):
    """Load a series of single-frame images from folder -> [N,H,W] uint8, dt."""
    files = []
    for ext in ("*.dcm", "*"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    # Keep only DICOMs
    dsets = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            # Skip non-image objects
            if getattr(ds, "SOPClassUID", ""):
                dsets.append((f, ds))
        except Exception:
            continue
    if not dsets:
        raise FileNotFoundError("No DICOM files found in folder.")

    # Sort by InstanceNumber, then by (AcquisitionTime/ContentTime) as fallback
    def sort_key(item):
        _, ds = item
        inst = getattr(ds, "InstanceNumber", 0)
        t = getattr(ds, "AcquisitionTime", None) or getattr(ds, "ContentTime", "")
        return (int(inst) if str(inst).isdigit() else 0, str(t))
    dsets.sort(key=sort_key)

    frames = []
    invert = None
    for path, _ds_meta in dsets:
        ds = pydicom.dcmread(path)
        if invert is None:
            invert = str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1"
        arr = ds.pixel_array
        if arr.ndim != 2:
            # If someone slipped a multiframe in the folder, take first frame
            arr = np.squeeze(arr)[..., :1]
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass
        frames.append(_to_uint8(arr, invert=invert))
    frames_u8 = np.stack(frames, axis=0)

    # Try to infer timing from (FrameTime) if present on any file, else fallback
    dt = None
    for _, ds in dsets:
        ft = ds.get("FrameTime", None)
        cr = ds.get("CineRate", None)
        if ft and ft > 0:
            dt = float(ft) / 1000.0
            break
        if cr and cr > 0:
            dt = 1.0 / float(cr)
            break
    if dt is None:
        dt = 1.0 / float(fallback_fps)
    return frames_u8, dt

def make_gif(input_path, output_gif, fps=None):
    """Create GIF from a multi-frame DICOM or a folder of single-frame DICOMs."""
    if os.path.isdir(input_path):
        frames, dt = _read_series(input_path, fallback_fps=fps or 25.0)
    else:
        ds = pydicom.dcmread(input_path)
        if hasattr(ds, "NumberOfFrames") and int(ds.NumberOfFrames) > 1:
            frames, dt = _read_multiframe(ds)
        else:
            # Treat the file's folder as a series
            frames, dt = _read_series(os.path.dirname(input_path), fallback_fps=fps or 25.0)

    if fps is not None and fps > 0:
        dt = 1.0 / float(fps)

    duration_per_frame = dt  # seconds
    imageio.mimsave(
        output_gif,
        [f for f in frames],  # list of 2D uint8 arrays
        format="GIF",
        duration=duration_per_frame,
        loop=0,  # 0 = infinite
    )
    print(f"Saved {frames.shape[0]} frames -> {output_gif}  (dt={duration_per_frame:.4f}s, fps≈{1/duration_per_frame:.2f})")

def main():
    in_path = "2ch.dcm"
    out_path = "2ch.gif"  # <-- must be a filename, not a folder
    make_gif(in_path, out_path, fps=25)
    

if __name__ == "__main__":
    main()