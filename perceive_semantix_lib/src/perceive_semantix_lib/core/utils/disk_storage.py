import gzip
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, Type

import cv2
import numpy as np
from jaxtyping import Float, UInt8

from perceive_semantix_lib.core.background_tracker import BackgroundTracker
from perceive_semantix_lib.core.geometry import GeometryType
from perceive_semantix_lib.core.geometry.point_cloud import PointCloud
from perceive_semantix_lib.core.input_types import InputData
from perceive_semantix_lib.core.object_tracker import ObjectTracker
from perceive_semantix_lib.legacy.io import load_scene_legacy

logger = logging.getLogger(__name__)


class DiskStorage:
    def __init__(self, root_path: Path) -> None:
        self.root = root_path
        self.color_path = self.root / "input" / "color"
        self.depth_path = self.root / "input" / "depth"
        self.camera_parameters_path = self.root / "input" / "camera.csv"
        self.detection_path = self.root / "debug"
        self.scene_path = self.root / "output"
        logger.info(f"Experiment storage at {self.root.resolve()}")
        self.ensure_dirs()

    @staticmethod
    def _create_timestamp_str(time_sec: float) -> str:
        return f"{time_sec:013.6f}".replace(".", "-")

    def _write_camera_params(
        self, intrinsics: Float[np.ndarray, "3 3"], camera_pose: Float[np.ndarray, "4 4"], timestamp: str
    ) -> None:
        newline = "\n"
        delimiter = ","

        intrinsics_strings = [f"{v:.18e}" for v in intrinsics.flatten()]
        pose_strings = [f"{v:.18e}" for v in camera_pose.flatten()]
        row_strings = [timestamp] + intrinsics_strings + pose_strings
        csv_line = ",".join(row_strings) + "\n"

        if not self.camera_parameters_path.exists():
            header = delimiter.join(
                ["timestamp"]
                + ["K_00", "K_01", "K_02", "K_10", "K_11", "K_12", "K_20", "K_21", "K_22"]
                + ["T_00", "T_01", "T_02", "T_03"]
                + ["T_10", "T_11", "T_12", "T_13"]
                + ["T_20", "T_21", "T_22", "T_23"]
                + ["T_30", "T_31", "T_32", "T_33"],
            )
            with open(self.camera_parameters_path, "w") as f:
                f.write(header + newline)
        with open(self.camera_parameters_path, "a") as f:
            f.write(csv_line)

    def save_input(self, input: InputData, time_sec: float) -> Tuple[Path, Path]:
        timestamp = DiskStorage._create_timestamp_str(time_sec)
        color_filename = f"{self.color_path}/{timestamp}.png"
        depth_filename = f"{self.depth_path}/{timestamp}.npy"
        bgr_image = cv2.cvtColor(input.color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(color_filename, bgr_image)
        np.save(depth_filename, input.depth, allow_pickle=False)
        self._write_camera_params(input.camera_intrinsics, input.pose, timestamp)
        return Path(color_filename), Path(depth_filename)

    def save_detectons(self, annotated_color: UInt8[np.ndarray, "H W 3"], time_sec: float) -> Path:
        timestamp = DiskStorage._create_timestamp_str(time_sec)
        annotated_color_filename = f"{self.detection_path}/{timestamp}.png"
        bgr_image = cv2.cvtColor(annotated_color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(annotated_color_filename, bgr_image)
        return Path(annotated_color_filename)

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.color_path.mkdir(parents=True, exist_ok=True)
        self.depth_path.mkdir(parents=True, exist_ok=True)
        self.camera_parameters_path.parent.mkdir(parents=True, exist_ok=True)
        self.detection_path.mkdir(parents=True, exist_ok=True)
        self.scene_path.mkdir(parents=True, exist_ok=True)

    def save_scene(
        self,
        object_tracker: ObjectTracker,
        background_tracker: BackgroundTracker,
        time_sec: float,
        camera_pose: Optional[Float[np.ndarray, "4 4"]],
    ) -> None:
        timestamp = DiskStorage._create_timestamp_str(time_sec)
        scene_filename = f"{self.scene_path}/{timestamp}.pkl"
        scene_dict = {
            "version": "0.0.1",
            "objects": object_tracker.to_dict(),
            "background": background_tracker.to_dict(),
            "time_sec": time_sec,
            "camera_pose": camera_pose,
        }
        with open(scene_filename, "wb") as f:
            pickle.dump(scene_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Pickled scene to {scene_filename}")

    @staticmethod
    def load_scene(
        path: Path, device: str = "cuda", geometry_type: Type[GeometryType] = PointCloud
    ) -> tuple[ObjectTracker, BackgroundTracker, float, Optional[Float[np.ndarray, "4 4"]]]:
        if not path.exists() or not path.is_file():
            logger.error(msg := f"Scene file not found: {path}")
            raise FileNotFoundError(msg)
        try:
            with open(path, "rb") as f:
                scene_dict = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to unpickle scene from {path.resolve()} with error {e}, trying gzip decompression")
            try:
                with gzip.open(path, "rb") as f:
                    scene_dict = pickle.load(f)
            except Exception as e2:
                logger.error(f"Failed to unpickle scene from {path.resolve()} with gzip decompression with error {e2}")
                raise e2 from e

        if "version" not in scene_dict or scene_dict["version"] != "0.0.1":
            logger.warning("Trying to load scene using legacy format")
            return load_scene_legacy(scene_dict, geometry_type=geometry_type)

        object_tracker = ObjectTracker.from_dict(scene_dict["objects"], device=device, geometry_type=geometry_type)
        background_tracker = BackgroundTracker.from_dict(scene_dict["background"], geometry_type=geometry_type)
        time_sec: float = scene_dict["time_sec"]
        camera_pose: Optional[Float[np.ndarray, "4 4"]] = scene_dict["camera_pose"]
        logger.info(f"Unpickled scene from {path.resolve()}")
        return object_tracker, background_tracker, time_sec, camera_pose
