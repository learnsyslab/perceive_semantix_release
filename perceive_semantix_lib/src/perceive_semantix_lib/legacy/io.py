import logging
from dataclasses import fields
from typing import Any, Optional
from uuid import UUID

import numpy as np
import torch
from jaxtyping import Float
from typeguard import check_type

from perceive_semantix_lib.core.background_tracker import BackgroundTracker
from perceive_semantix_lib.core.geometry import GeometryType, PointCloud
from perceive_semantix_lib.core.object_tracker import ObjectTracker
from perceive_semantix_lib.core.scene_object import SceneObject
from perceive_semantix_lib.core.utils.pocd_datatypes import (
    BetaDistributedValue,
    POCDObjectTypes,
    StandardDistributedValue,
)

logger = logging.getLogger(__name__)


def _load_object_legacy(
    obj: dict, device: str = "cuda", geometry_type: type[GeometryType] = PointCloud
) -> SceneObject[GeometryType]:
    point_cloud = geometry_type.from_serializable((obj["pcd_np"], obj["pcd_color_np"]))
    visual_feature = torch.tensor(obj["clip_ft"], dtype=torch.float32, device=device)

    scene_object: SceneObject[GeometryType] = SceneObject.__new__(SceneObject)

    # read from dict
    scene_object.id = UUID(obj["id"])
    scene_object.geometry = point_cloud
    scene_object.visual_feature = visual_feature
    scene_object.class_name = obj["class_name"]
    scene_object.observation_times = (np.arange(obj["num_detections"], dtype=int) + 1).tolist()
    scene_object.instance_color = tuple(np.array(obj["inst_color"]).tolist())

    statonarity_prior: str = obj.get("stationarity_prior", "DYNAMIC")
    if statonarity_prior == "STATIC":
        scene_object.stationarity_prior = POCDObjectTypes.STATIC
    else:
        scene_object.stationarity_prior = POCDObjectTypes.DYNAMIC

    scene_object.pocd_stationarity = BetaDistributedValue(alpha=float(obj["a"]), beta=float(obj["b"]))
    scene_object.pocd_geometric_change = StandardDistributedValue(mean=float(obj["mu"]), std=float(obj["sig"]))

    scene_object.pocd_inlier = bool(obj["inlier"])
    scene_object.time_of_disappearance = float(obj["time_of_disappearance"])
    scene_object.current_change = None
    scene_object.last_decay_fake_measurement_time = float(obj["last_decay_fakemeasurment_time"])
    scene_object.class_confidence_history_ = {int(obj["class_id"]): (1, 1.0)}

    # derived attributes
    scene_object.id_str = str(scene_object.id)[:4] + "_" + scene_object.class_name.replace(" ", "-")
    scene_object.bounding_box = scene_object.geometry.get_bounding_box()
    scene_object.ground_projection = {}
    scene_object.ground_projection_dirty = True
    scene_object._update_class()
    if scene_object.class_id != obj["class_id"]:
        logger.warning(
            f"Class ID mismatch when loading SceneObject from dict: {scene_object.class_id} != {obj['class_id']}"
        )
    scene_object.stationarity_confidence = scene_object.pocd_stationarity.mean()

    def validate(obj):
        for f in fields(obj):
            try:
                val = getattr(obj, f.name)
            except AttributeError as e:
                logger.fatal(msg := f"Missing attribute {f.name} in SceneObject deserialization: {e}")
                raise RuntimeError(msg)
            try:
                check_type(f.name, val, f.type)
            except TypeError as e:
                logger.fatal(msg := f"Type mismatch for attribute {f.name} in SceneObject deserialization: {e}")
                raise RuntimeError(msg)

    validate(scene_object)
    return scene_object


def load_scene_legacy(
    scene_dict: dict[str, Any], geometry_type: type[GeometryType] = PointCloud
) -> tuple[ObjectTracker, BackgroundTracker[GeometryType], float, Optional[Float[np.ndarray, "4 4"]]]:
    """Load a scene from a legacy pickle file."""
    camera_pose: Optional[Float[np.ndarray, "4 4"]] = scene_dict.get("camera_pose", None)
    frame_index: int = int(scene_dict.get("frame_index", 0))
    time_sec: float = scene_dict.get("frame_time", float(frame_index))

    active_objects: list = scene_dict["objects"]
    missing_objects: list = scene_dict.get("missing_objects", [])

    background_points: np.ndarray = scene_dict.get("background_points", np.empty((0, 3), dtype=np.float32))
    background_colors: Optional[np.ndarray] = scene_dict.get("background_colors", None)
    background_tracker = BackgroundTracker(geometry_type=geometry_type)
    background_tracker.geometry = geometry_type.from_serializable((background_points, background_colors))

    object_tracker = ObjectTracker()
    for obj_dict in active_objects:
        object_tracker.add_object(_load_object_legacy(obj_dict, device="cuda", geometry_type=geometry_type))

    for obj_dict in missing_objects:
        object_tracker.add_object(obj := _load_object_legacy(obj_dict, device="cuda", geometry_type=geometry_type))
        object_tracker.mark_missing(obj.id, time_sec=obj.time_of_disappearance)

    return object_tracker, background_tracker, time_sec, camera_pose
