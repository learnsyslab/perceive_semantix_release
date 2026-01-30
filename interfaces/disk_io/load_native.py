import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from jaxtyping import Float


def check_database(
    path: Path, color_subpath: str = "color", depth_subpath: str = "depth", camera_params_file: str = "camera.csv"
) -> bool:
    """Check that the dataset is consistent.

    Requirements:
        - color and depth folders contain the same number of images named <timestamp>.png and <timestamp>.npy, respectively
        - camera_params_file has one row with flattend intrinsics matrix (9 entries) and flattend camera pose (16 entries) per timestamp
        - each timestamp has a color, depth, and camera pose entry
    """
    path = Path(path)
    color_dir = path / color_subpath
    depth_dir = path / depth_subpath
    camera_param_path = path / camera_params_file

    if not color_dir.is_dir() or not depth_dir.is_dir() or not camera_param_path.is_file():
        logging.error("Color or depth directory or pose file does not exist.")
        return False

    color_timestamps = {f.stem for f in color_dir.glob("*.png")}
    depth_timestamps = {f.stem for f in depth_dir.glob("*.npy")}
    if color_timestamps != depth_timestamps:
        logging.error("Color and depth directories have different timestamps.")
        return False

    try:
        with open(camera_param_path, "r") as f:
            _header = f.readline()
            param_lines = f.readlines()
    except Exception:
        logging.error(f"Failed to read pose file {camera_param_path.resolve()}.")
        return False

    param_timestamps = set()
    for line in param_lines:
        parts = line.strip().split(",")
        if len(parts) != (26):  # 1 timestamp + 9 intrinsics + 16 extrinsics
            return False
        param_timestamps.add(str(parts[0]).strip())

    if len(param_timestamps) != len(color_timestamps):
        logging.error("Number of camera parameter entries does not match number of color/depth images.")
        return False

    if color_timestamps != param_timestamps:
        logging.error("Mismatch between image timestamps and camera parameter timestamps.")
        return False

    return True


def timestamp_to_seconds(ts: str) -> float:
    """Convert a timestamp string in the format 'seconds-microseconds' to float seconds."""
    return float(ts.replace("-", ".").strip())


def iterate_database(
    path: Path,
    color_subpath: str = "color",
    depth_subpath: str = "depth",
    camera_params_file: str = "camera.csv",
) -> Iterator[
    tuple[
        float, Float[np.ndarray, "H W 3"], Float[np.ndarray, "H W"], Float[np.ndarray, "3 3"], Float[np.ndarray, "4 4"]
    ]
]:
    """Iterate over the dataset entries in timestamp order.

    Yields:
        output (tuple):
            - timestamp: float in seconds
            - color_image: np.ndarray (H, W, 3)
            - depth_image: np.ndarray (H, W)
            - camera_intrinsics: np.ndarray (3, 3)
            - camera_pose: np.ndarray (4, 4)

    """
    path = Path(path)
    color_dir = path / color_subpath
    depth_dir = path / depth_subpath
    pose_path = path / camera_params_file

    # Read camera poses
    poses = {}
    intrinsics = {}
    try:
        with open(pose_path, "r") as f:
            _ = f.readline()  # skip header
            for line in f:
                parts = line.strip().split(",")
                ts = parts[0].strip()
                try:
                    intrinsics_vals = np.array(parts[1:10], dtype=float)
                    pose_vals = np.array(parts[10:], dtype=float)
                    intrinsics[ts] = intrinsics_vals.reshape(3, 3)
                    poses[ts] = pose_vals.reshape(4, 4)
                except ValueError:
                    logging.warning(f"Invalid numeric values in pose line for timestamp {ts}")
    except Exception as e:
        logging.error(f"Failed to read pose file {pose_path.resolve()}: {e}")
        return

    # Iterate over all timestamps
    color_timestamps = sorted(f.stem for f in color_dir.glob("*.png"))
    for ts in color_timestamps:
        color_path = color_dir / f"{ts}.png"
        depth_path = depth_dir / f"{ts}.npy"

        if ts not in poses or ts not in intrinsics:
            logging.warning(f"No camera pose or intrinsics found for timestamp {ts}")
            continue

        color_img_bgr = cv2.imread(str(color_path), cv2.IMREAD_UNCHANGED)
        depth_img = np.load(str(depth_path))
        if color_img_bgr is None or depth_img is None:
            logging.warning(f"Failed to read images for timestamp {ts}")
            continue
        color_img_rgb = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2RGB)

        yield timestamp_to_seconds(ts), color_img_rgb, depth_img, intrinsics[ts], poses[ts]
