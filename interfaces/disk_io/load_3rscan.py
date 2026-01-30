import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from jaxtyping import Float, UInt8


def _scale_intrinsic_and_image(
    intrinsic: Float[np.ndarray, "3 3"] | UInt8[np.ndarray, "3 3"],
    image: Float[np.ndarray, "H W C"],
    target_width: int,
    target_height: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[
    Float[np.ndarray, "3 3"],
    Float[np.ndarray, "H' W' C"] | UInt8[np.ndarray, "H' W' C"],  # resized image
]:
    h, w = image.shape[:2]

    scale_x = target_width / w
    scale_y = target_height / h

    # Scale the intrinsic matrix
    scaled_intrinsic = intrinsic.copy()
    scaled_intrinsic[0, 0] *= scale_x  # fx
    scaled_intrinsic[1, 1] *= scale_y  # fy
    scaled_intrinsic[0, 2] *= scale_x  # cx
    scaled_intrinsic[1, 2] *= scale_y  # cy

    # Resize image
    scaled_image = cv2.resize(image, (target_width, target_height), interpolation=interpolation)

    if scaled_image.ndim == 2 and image.ndim == 3:
        scaled_image = np.expand_dims(scaled_image, axis=2)

    return scaled_intrinsic, scaled_image


def load_pose(file_path: Path) -> Float[np.ndarray, "4 4"]:
    """Load a 4x4 camera pose matrix from a text file."""
    pose = np.loadtxt(file_path, dtype=np.float32).reshape(4, 4)
    return pose


def check_database(*args) -> bool:
    """Check that the dataset is consistent."""
    return True


def iterate_database(
    path: Path,
) -> Iterator[
    tuple[
        float, UInt8[np.ndarray, "H W 3"], Float[np.ndarray, "H W"], Float[np.ndarray, "3 3"], Float[np.ndarray, "4 4"]
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
    info_file = path / "_info.txt"

    # Initialize variables
    data = {}

    # Read file
    with open(info_file, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                key = key.strip()
                value = value.strip()
                data[key] = value

    # Helper function to parse numeric lists
    def parse_matrix(value_str):
        return np.array([float(x) for x in value_str.split()])

    # Extract and convert to appropriate types
    version_number = int(data["m_versionNumber"])
    _sensor_name = data["m_sensorName"]
    color_width = int(data["m_colorWidth"])
    color_height = int(data["m_colorHeight"])
    depth_width = int(data["m_depthWidth"])
    depth_height = int(data["m_depthHeight"])
    depth_shift = float(data["m_depthShift"])
    _num_frames = int(data["m_frames.size"])

    # Intrinsics are 3x3, extrinsics are 4x4
    color_intrinsic = parse_matrix(data["m_calibrationColorIntrinsic"]).reshape(4, 4)
    color_extrinsic = parse_matrix(data["m_calibrationColorExtrinsic"]).reshape(4, 4)
    depth_intrinsic = parse_matrix(data["m_calibrationDepthIntrinsic"]).reshape(4, 4)
    depth_extrinsic = parse_matrix(data["m_calibrationDepthExtrinsic"]).reshape(4, 4)

    if np.any(color_intrinsic[3, :] != np.array([0, 0, 0, 1])) or np.any(
        color_intrinsic[:, 3] != np.array([0, 0, 0, 1])
    ):
        logging.error(msg := "Invalid color intrinsic matrix format.")
        raise ValueError(msg)
    color_intrinsic = color_intrinsic[:3, :3]

    if np.any(depth_intrinsic[3, :] != np.array([0, 0, 0, 1])) or np.any(
        depth_intrinsic[:, 3] != np.array([0, 0, 0, 1])
    ):
        logging.error(msg := "Invalid depth intrinsic matrix format.")
        raise ValueError(msg)
    depth_intrinsic = depth_intrinsic[:3, :3]

    if version_number != 4:
        logging.error(msg := f"Unsupported 3RScan version number: {version_number}")
        raise ValueError(msg)

    if np.any(color_extrinsic != np.eye(4)) or np.any(depth_extrinsic != np.eye(4)):
        logging.error(msg := "Non-identity calibration extrinsics are not supported.")
        raise ValueError(msg)

    target_width = max(color_width, depth_width)
    target_height = max(color_height, depth_height)

    color_files = sorted(path.glob("frame-*.color.jpg"))

    for color_path in color_files:
        frame_id = color_path.stem.split("-")[1].split(".")[0]  # e.g. '000115'
        frame_num = float(frame_id)

        # Derive other file paths
        depth_path = path / f"frame-{frame_id}.depth.pgm"
        pose_path = path / f"frame-{frame_id}.pose.txt"

        # Load images
        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        color = np.array(color_rgb, dtype=np.uint8)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth = np.expand_dims(np.array(depth, dtype=np.float32), axis=2) / depth_shift

        if color is None or depth is None or not pose_path.exists():
            logging.warning(f"Skipping incomplete frame: {frame_id}")
            continue  # skip incomplete frames

        # Scale intrinsic matrix
        scaled_intrinsic_color, scaled_color = _scale_intrinsic_and_image(
            color_intrinsic, color, target_width, target_height
        )
        scaled_intrinsic_depth, scaled_depth = _scale_intrinsic_and_image(
            depth_intrinsic, depth, target_width, target_height, interpolation=cv2.INTER_NEAREST
        )

        # Load pose
        pose = load_pose(pose_path)

        yield frame_num, scaled_color, scaled_depth, scaled_intrinsic_color, pose
