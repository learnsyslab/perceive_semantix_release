import bisect
from dataclasses import dataclass, fields
from logging import getLogger
from typing import Any, ClassVar, Generic, Optional, Type
from uuid import UUID, uuid4

import numpy as np
import torch
from jaxtyping import Bool, Float, UInt8
from open3d.geometry import AxisAlignedBoundingBox  # pyright: ignore[reportMissingImports]
from scipy.special import gammaln
from scipy.stats import norm, uniform
from supervision.detection.core import Detections
from torch.nn import functional as F
from typeguard import check_type

from perceive_semantix_lib.core.geometry import GeometryType, PointCloud
from perceive_semantix_lib.core.geometry.tensor_point_cloud import TensorPointCloud
from perceive_semantix_lib.core.occupancy_grid import OccupancyGrid
from perceive_semantix_lib.core.utils.object_classes import get_global_object_classes
from perceive_semantix_lib.core.utils.pocd_datatypes import (
    BetaDistributedValue,
    POCDObjectTypes,
    StandardDistributedValue,
)

logger = getLogger(__name__)


@dataclass
class SceneObject(Generic[GeometryType]):
    id: UUID
    id_str: str

    geometry: GeometryType
    bounding_box: AxisAlignedBoundingBox
    visual_feature: Float[torch.Tensor, "N"]

    ground_projection: dict[str, Optional[OccupancyGrid]]
    ground_projection_dirty: bool

    class_id: int
    class_name: str
    class_confidence: float
    class_confidence_history_: dict[int, tuple[int, float]]

    observation_times: list[float]
    instance_color: tuple[float, float, float]

    stationarity_prior: POCDObjectTypes
    pocd_stationarity: BetaDistributedValue
    pocd_geometric_change: StandardDistributedValue
    pocd_inlier: bool
    stationarity_confidence: float

    time_of_disappearance: float

    current_change: Optional[StandardDistributedValue]
    last_decay_fake_measurement_time: float

    stationarity_prior_map: ClassVar[dict[str, POCDObjectTypes]] = {}

    def __init__(
        self,
        class_id: int,
        class_name: str,
        class_confidence: float,
        observation_time: float,
        goemetry: GeometryType,
        bounding_box: AxisAlignedBoundingBox,
        visual_feature: Float[torch.Tensor, "N"],
        id: Optional[UUID] = None,
        stationarity_prior: POCDObjectTypes = POCDObjectTypes.DYNAMIC,
    ):
        if id is None:
            self.id = uuid4()
        else:
            self.id = id

        self.geometry = goemetry
        self.bounding_box = bounding_box
        self.visual_feature = visual_feature

        self.ground_projection = {}
        self.ground_projection_dirty = True

        self.class_id = class_id
        self.class_name = class_name
        self.class_confidence = class_confidence
        self.class_confidence_history_ = {self.class_id: (1, class_confidence)}

        self.observation_times = [observation_time]
        self.instance_color = tuple(np.random.rand(3).tolist())

        self.stationarity_prior = stationarity_prior
        self.pocd_stationarity = BetaDistributedValue(alpha=2.0, beta=1.0)
        self.pocd_geometric_change = StandardDistributedValue(mean=0.0, std=0.5)
        self.pocd_inlier = True
        self.stationarity_confidence = self.pocd_stationarity.mean()

        self.time_of_disappearance = -1

        self.current_change = None
        self.last_decay_fake_measurement_time = -1
        self.id_str = str(self.id)[:4] + "_" + self.class_name.replace(" ", "-")

    def update_pocd_probabilities(self, eps: float = 1e-5, cap: float = 10) -> None:
        """Update the POCD probabilities based on the current change measurement.

        For details see:
            https://github.com/Viky397/TorWICDataset/blob/main/Qian_Chatrath_POCD_SuppMaterial.pdf
            and
            https://www.roboticsproceedings.org/rss18/p013.pdf
        """
        if self.current_change is None:
            logger.error("Cannot update POCD probabilities without a current change measurement.")
            return

        mu = self.pocd_geometric_change.mean
        sig = self.pocd_geometric_change.std
        a = self.pocd_stationarity.alpha
        b = self.pocd_stationarity.beta
        object_type = self.stationarity_prior
        inlier = self.pocd_inlier
        change = self.current_change.mean
        std_change = self.current_change.std

        s_weight = 1  # Higher weight means greater increase, lower decrease
        if object_type == POCDObjectTypes.DYNAMIC and not inlier:
            s_weight = 0  # Drop semi-fast
        elif object_type == POCDObjectTypes.DYNAMIC and inlier:
            s_weight = 2  # Rise fast
        elif object_type == POCDObjectTypes.STATIC and not inlier:
            s_weight = 0  # drop slow
        elif object_type == POCDObjectTypes.STATIC and inlier:
            s_weight = 3  # rise fast
        elif object_type == POCDObjectTypes.DISSAPEARED and not inlier:
            s_weight = 5  # drop very fast
        elif object_type == POCDObjectTypes.DISSAPEARED and inlier:
            s_weight = 0  # rise slow
        # object_type = min(1, object_type.value)
        object_type = 1

        tolerance = 20 * std_change

        s_sq = 1 / (1 / np.square(sig) + 1 / np.square(std_change))
        m = s_sq * (mu / np.square(sig) + change / np.square(std_change))

        k1, k2 = self.updateKSingle(k=s_weight, eps=eps)

        C1 = k1 * max(norm.pdf(change, loc=mu, scale=sig), eps)
        if abs(change) >= (tolerance - eps):
            C2 = k2 * uniform.pdf(tolerance, loc=0, scale=tolerance)
        else:
            C2 = k2 * uniform.pdf(abs(change), loc=0, scale=tolerance)
        C1 = max(eps, C1)
        C2 = max(eps, C2)
        C_norm = C1 + C2
        C1 /= C_norm
        C2 /= C_norm

        inlier = True if C1 >= C2 else False
        self.pocd_inlier = inlier

        mu_prime = C1 * m + C2 * mu
        sig = np.sqrt(C1 * (s_sq + np.square(m)) + C2 * (np.square(sig) + np.square(mu)) - np.square(mu_prime))

        gamma = (a + object_type * s_weight + 1) / (a + b + s_weight + 1)
        eta = (a + object_type * s_weight) / (a + b + s_weight + 1)
        theta = C1 * gamma + C2 * eta
        alpha = ((a + object_type * s_weight + 2) * (a + object_type * s_weight + 1)) / (
            (a + b + s_weight + 1) * (a + b + s_weight + 2)
        )
        beta = ((a + object_type * s_weight + 1) * (a + object_type * s_weight)) / (
            (a + b + s_weight + 1) * (a + b + s_weight + 2)
        )
        # nu = C1 * alpha+ C2 * beta

        theta_sq = np.square(theta)
        a = (C1 * theta * alpha + C2 * beta * theta - theta_sq) / (theta_sq - C1 * alpha - C2 * beta)
        b = (
            (C1 * theta * alpha + C2 * beta * theta - theta_sq)
            * (1 - theta)
            / ((theta_sq - C1 * alpha - C2 * beta) * theta)
        )

        if a > cap or b > cap:
            ratio = max(a, b) / cap
            a /= ratio
            b /= ratio
        self.pocd_stationarity.alpha = float(a)
        self.pocd_stationarity.beta = float(b)
        self.pocd_geometric_change.std = sig
        self.pocd_geometric_change.mean = mu_prime
        self.stationarity_confidence = self.pocd_stationarity.mean()

    def updateKSingle(self, k: float, eps: float = 1e-5) -> tuple[float, float]:
        """Beta Distribution calculation in posterior stationarity update rule."""
        a = self.pocd_stationarity.alpha
        b = self.pocd_stationarity.beta
        object_type = self.stationarity_prior.value

        lk1 = (gammaln(a + b) + gammaln(a + k * object_type + 1) + gammaln(b + k - k * object_type)) - (
            gammaln(a) + gammaln(b) + gammaln(a + b + k + 1)
        )
        lk2 = (gammaln(a + b) + gammaln(a + k * object_type) + gammaln(b + k - k * object_type + 1)) - (
            gammaln(a) + gammaln(b) + gammaln(a + b + k + 1)
        )

        k1 = np.exp(lk1)
        k2 = np.exp(lk2)

        ks = k1 + k2
        k1 /= ks
        k2 /= ks

        k1 = max(eps, k1)
        k2 = max(eps, k2)

        return k1, k2

    def first_observed_time(self) -> float:
        return self.observation_times[0]

    def last_observed_time(self) -> float:
        return self.observation_times[-1]

    def number_of_observations(self) -> int:
        return len(self.observation_times)

    def _add_class_observation(self, class_id: int, class_confidence: float) -> None:
        if class_id in self.class_confidence_history_:
            count, total_conf = self.class_confidence_history_[class_id]
            self.class_confidence_history_[class_id] = (count + 1, total_conf + class_confidence)
        else:
            self.class_confidence_history_[class_id] = (1, class_confidence)
        self._update_class()

    def _update_class(self) -> None:
        if len(self.class_confidence_history_) == 0:
            logger.error("Cannot update class without history.")
            return
        self.class_id = max(self.class_confidence_history_, key=lambda k: self.class_confidence_history_[k][1])  # type: ignore
        self.class_confidence = (
            self.class_confidence_history_[self.class_id][1] / self.class_confidence_history_[self.class_id][0]
        )  # type: ignore
        self.class_name = get_global_object_classes()[self.class_id]

    def merge_with_detection(
        self,
        class_id: int,
        class_confidence: float,
        point_cloud: GeometryType,
        visual_feature: Float[torch.Tensor, "N"],
        time_sec: float,
        replace_pointcloud: bool = False,
    ) -> None:
        """Merge the current SceneObject with a new detection.

        Args:
            class_id (int): Class ID of the new detection.
            class_confidence (float): Confidence of the new detection.
            point_cloud (GeometryType): Point cloud of the new detection.
            visual_feature (Float[torch.Tensor, "N"]): Visual feature of the new detection.
            time_sec (float): Time of the new detection.
            replace_pointcloud (bool): If True, replace the existing point cloud with the new one. If False, merge them.

        """
        self._add_class_observation(class_id, class_confidence)
        if replace_pointcloud:
            self.geometry = point_cloud
        else:
            self.geometry += point_cloud
        self.ground_projection_dirty = True
        self.bounding_box = self.geometry.get_bounding_box()

        if time_sec < self.last_observed_time():
            bisect.insort(self.observation_times, time_sec)
        else:
            self.observation_times.append(time_sec)
        self.time_of_disappearance = -1
        self.visual_feature = (self.visual_feature * self.number_of_observations() + visual_feature) / (
            self.number_of_observations() + 1
        )

    def merge_with_object(
        self,
        other: "SceneObject",
        replace_pointcloud: bool = False,
        use_greater_stationarity_estimate: bool = False,
    ) -> None:
        """Merge another SceneObject into the current SceneObject.

        Args:
            other (SceneObject): The SceneObject to merge with.
            replace_pointcloud (bool): If True, replace the existing point cloud with the other object's point cloud. If False, merge them.
            use_greater_stationarity_estimate (bool): If True, keep the stationarity estimate with the higher confidence.

        """
        if self.observation_times[-1] > other.observation_times[-1]:
            logger.warning("Merging an older object into a newer one.")
        self.observation_times.extend(other.observation_times)
        self.observation_times.sort()

        if replace_pointcloud:
            self.geometry = other.geometry
        else:
            self.geometry += other.geometry
        self.ground_projection_dirty = True
        self.bounding_box = self.geometry.get_bounding_box()
        self.visual_feature = (
            self.visual_feature * self.number_of_observations() + other.visual_feature * other.number_of_observations()
        ) / (self.number_of_observations() + other.number_of_observations())
        self.class_confidence_history_ = {
            key: (
                self.class_confidence_history_.get(key, (0, 0))[0]
                + other.class_confidence_history_.get(key, (0, 0))[0],
                self.class_confidence_history_.get(key, (0, 0))[1]
                + other.class_confidence_history_.get(key, (0, 0))[1],
            )
            for key in set(self.class_confidence_history_) | set(other.class_confidence_history_)
        }
        self._update_class()
        self.time_of_disappearance = -1

        if not (use_greater_stationarity_estimate and self.stationarity_confidence > other.stationarity_confidence):
            self.pocd_stationarity.alpha = other.pocd_stationarity.alpha
            self.pocd_stationarity.beta = other.pocd_stationarity.beta
            self.pocd_geometric_change.mean = other.pocd_geometric_change.mean
            self.pocd_geometric_change.std = other.pocd_geometric_change.std
            self.pocd_inlier = other.pocd_inlier
            self.stationarity_confidence = other.stationarity_confidence
            self.stationarity_prior = other.stationarity_prior

    def get_ground_projection(
        self,
        map_id: str,
        resolution: float = 0.1,
        occupied_height_bounds: tuple[float, float] = (0.0, 10.0),
        padding_meters: float = 0.0,
        padding_value: int = -1,
    ) -> Optional[OccupancyGrid]:
        if self.ground_projection_dirty or map_id not in self.ground_projection:
            occ = self.geometry.to_occupancy_grid(resolution, occupied_height_bounds)
            if padding_meters != 0.0 and occ is not None:
                occ.pad(padding_meters, padding_value)
            self.ground_projection[map_id] = occ
            self.ground_projection_dirty = False
        return self.ground_projection[map_id]

    @classmethod
    def generate_from_detections(
        cls, detection: Detections, clip_features: Float[torch.Tensor, "N clip"], time_sec: float
    ) -> list:
        """Generate a list of SceneObject instances from detections and corresponding CLIP features.

        Args:
            detection (Detections): Detections object containing detection data of N detections.
            clip_features (Float[torch.Tensor, "N clip"]): CLIP features corresponding to the N detections.
            time_sec (float): Observation time for the detections.

        Returns:
            list[SceneObject]: List of generated SceneObject instances.

        """
        if detection.data is None:
            logger.error("No detection data available to generate SceneObject.")
            return []
        if len(detection) != len(clip_features):
            logger.error(
                f"Mismatch between number of detections ({len(detection)}) and number of clip features ({len(clip_features)})."
            )
            return []
        if detection.class_id is None or detection.confidence is None:
            logger.error("Detection must have 'class_id' and 'confidence' to generate SceneObject.")
            return []
        if (
            "point_clouds" not in detection.data
            or "bounding_boxes" not in detection.data
            or "classes" not in detection.data
        ):
            logger.error(
                "Detection data must contain 'point_clouds', 'bounding_boxes', and 'classes' to generate SceneObject."
            )
            return []

        return [
            SceneObject(
                class_id=int(id),
                class_name=class_name,
                class_confidence=float(conf),
                observation_time=time_sec,
                goemetry=pcd,
                bounding_box=bbox,
                visual_feature=visual_feature,
                stationarity_prior=cls.stationarity_prior_map.get(class_name, POCDObjectTypes.DYNAMIC),
            )
            for id, conf, pcd, bbox, class_name, visual_feature in zip(
                detection.class_id,
                detection.confidence,
                detection.data["point_clouds"],
                detection.data["bounding_boxes"],
                detection.data["classes"],
                clip_features,
            )
        ]

    @staticmethod
    def project_onto_camera(
        objects: list,  # list["SceneObject"], but lookahead type not supported by jaxtyping
        camera_pose: Float[torch.Tensor, "4 4"],
        camera_intrinsics_torch: Float[torch.Tensor, "3 3"],
        img_height: int,
        img_width: int,
        min_depth: float,
        max_depth: float,
        occlusion_margin: float,
        occluding_depth_image: Optional[Float[torch.Tensor, "1 1 H W"]] = None,
        visibility_threshold: float = 0.35,
        preallocate_object_mask_count: int = 10,
        device: str = "cuda",
    ) -> tuple[list[UUID], list[float], Bool[torch.Tensor, "N H W"]]:
        """Project each object on the camera plane and return which objects are visible form the current camera pose.

        Args:
            objects (list[SceneObject]): List of SceneObject instances.
            camera_pose (Float[torch.Tensor, "4 4"]): Camera pose matrix.
            camera_intrinsics_torch (Float[torch.Tensor, "3 3"]): Camera intrinsic matrix.
            img_height (int): Image height to project on.
            img_width (int): Image width to project on.
            min_depth (float): Minimum depth to consider.
            max_depth (float): Maximum depth to consider.
            occlusion_margin (float): Margin in meters to consider for occlusion.
            occluding_depth_image (Optional[Float[torch.Tensor, "1 1 H W"]]): Raw depth image from the camera that is used to determine occlusion.
            visibility_threshold (float): Minimum ratio of visible points to total points to consider an object as expected.
            preallocate_object_mask_count (int): Number of object masks to preallocate.
            device (str): Device to use for computation.

        Returns:
            expected_object_ids (list[UUID]): List of UUIDs of expected objects.
            object_visibility_ratio (list[float]): List of visibility ratios for each expected object.
            object_projections (Float[torch.Tensor, "N H W"]): Binary masks of shape (N, H, W) for each expected object.

        """
        # somehow hacky way to do get the correct type (PointCloud or TensorPointCloud), this is necessary because batch_project_to_camera is a class method
        if occluding_depth_image is None:
            logger.warning("Occluding depth image is required for projecting objects onto camera.")
        else:
            occluding_depth_image = occluding_depth_image.squeeze(0)  # (1, 1, H, W) -> (1, H, W)
        geom_class = type(objects[0].geometry) if len(objects) > 0 else TensorPointCloud
        expected_object_indices, object_visibility_ratio, object_projections = geom_class.batch_project_to_camera(
            [obj.geometry for obj in objects],
            camera_pose,
            camera_intrinsics_torch,
            img_height,
            img_width,
            min_depth,
            max_depth,
            occlusion_margin,
            occluding_depth_image,
            device,
            preallocated_object_mask_count=preallocate_object_mask_count,
            expected_visibility_threshold=visibility_threshold,
        )

        expected_object_ids = [objects[i].id for i in expected_object_indices]
        object_projections = morphological_closing(object_projections, kernel_size=9)

        return expected_object_ids, object_visibility_ratio, object_projections

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "id_str": self.id_str,
            "point_cloud": self.geometry.to_serializable(),
            "visual_feature": self.visual_feature.cpu().numpy(),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "observation_times": self.observation_times,
            "instance_color": self.instance_color,
            "stationarity_prior": self.stationarity_prior.name,
            "pocd_stationarity": {
                "alpha": self.pocd_stationarity.alpha,
                "beta": self.pocd_stationarity.beta,
            },
            "pocd_geometric_change": {
                "mean": self.pocd_geometric_change.mean,
                "std": self.pocd_geometric_change.std,
            },
            "pocd_inlier": self.pocd_inlier,
            "time_of_disappearance": self.time_of_disappearance,
            "current_change": {
                "mean": self.current_change.mean if self.current_change else None,
                "std": self.current_change.std if self.current_change else None,
            },
            "last_decay_fake_measurement_time": self.last_decay_fake_measurement_time,
            "class_confidence_history": self.class_confidence_history_,
        }

    @classmethod
    def from_dict(
        cls, obj: dict[str, Any], device: str = "cuda", geometry_type: Type[GeometryType] = PointCloud
    ) -> "SceneObject":
        visual_feature = torch.tensor(obj["visual_feature"], dtype=torch.float32, device=device)

        scene_object = cls.__new__(cls)

        # read from dict
        scene_object.id = UUID(obj["id"])
        scene_object.id_str = obj["id_str"]
        scene_object.geometry = geometry_type.from_serializable(obj["point_cloud"])
        scene_object.visual_feature = visual_feature
        scene_object.class_name = obj["class_name"]
        scene_object.observation_times = obj["observation_times"]
        scene_object.instance_color = tuple(obj["instance_color"])
        scene_object.stationarity_prior = POCDObjectTypes[obj["stationarity_prior"]]
        scene_object.pocd_stationarity = BetaDistributedValue(
            alpha=obj["pocd_stationarity"]["alpha"], beta=obj["pocd_stationarity"]["beta"]
        )
        scene_object.pocd_geometric_change = StandardDistributedValue(
            mean=obj["pocd_geometric_change"]["mean"], std=obj["pocd_geometric_change"]["std"]
        )
        scene_object.pocd_inlier = obj["pocd_inlier"]
        scene_object.time_of_disappearance = obj["time_of_disappearance"]
        if obj["current_change"]["mean"] is not None and obj["current_change"]["std"] is not None:
            scene_object.current_change = StandardDistributedValue(
                mean=obj["current_change"]["mean"], std=obj["current_change"]["std"]
            )
        else:
            scene_object.current_change = None
        scene_object.last_decay_fake_measurement_time = obj["last_decay_fake_measurement_time"]
        scene_object.class_confidence_history_ = {int(k): tuple(v) for k, v in obj["class_confidence_history"].items()}

        # derived attributes
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

    def shift_all_recorded_timestamps(self, seconds: float):
        """Shift all recorded timestamps by a given number of seconds.

        Shifts ``observation_times``, ``time_of_disappearance``, and ``last_decay_fake_measurement_time``.

        Args:
            seconds (float): Number of seconds to shift the timestamps by. Can be negative.

        """
        self.observation_times = [t + seconds for t in self.observation_times]
        if self.time_of_disappearance > 0:
            self.time_of_disappearance += seconds
        if self.last_decay_fake_measurement_time > 0:
            self.last_decay_fake_measurement_time += seconds


def morphological_closing(masks: UInt8[torch.Tensor, "N H W"], kernel_size: int = 3) -> Bool[torch.Tensor, "N H W"]:
    """Perform morphological closing (dilation followed by erosion) on binary masks.

    Args:
        masks (Tensor): Input binary masks of shape (N, H, W), values 0 or 1 (uint8).
        kernel_size (int): Size of the structuring element (must be odd).

    Returns:
        Tensor: Output binary masks of shape (N, H, W) after closing.

    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"

    masks = masks.unsqueeze(1).float()  # (N, 1, H, W)
    padding = kernel_size // 2

    dilated = F.max_pool2d(masks, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-dilated, kernel_size, stride=1, padding=padding)
    closed = (eroded > 0.5).squeeze(1).to(torch.bool)
    return closed
