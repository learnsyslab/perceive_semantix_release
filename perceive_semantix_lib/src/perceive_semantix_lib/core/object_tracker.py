from logging import getLogger
from typing import Any, Iterable, Iterator, Optional, Sequence, Type, Union
from uuid import UUID

import numpy as np
import torch
from jaxtyping import Float
from supervision.detection.core import Detections
from torch.nn import functional as F

from perceive_semantix_lib.core.config import ObjectMatchingConfig
from perceive_semantix_lib.core.geometry import GeometryType
from perceive_semantix_lib.core.geometry.point_cloud import PointCloud
from perceive_semantix_lib.core.matching.matching import AssociationType, ObjectDetectionAdjacency
from perceive_semantix_lib.core.scene_object import SceneObject
from perceive_semantix_lib.core.utils.pocd_datatypes import POCDObjectTypes, StandardDistributedValue
from perceive_semantix_lib.core.utils.similarity import compute_overlap_matrix_general

logger = getLogger(__name__)


class ObjectTracker:
    def __init__(self):
        self.objects: dict[UUID, SceneObject] = {}
        self.active: list[UUID] = []
        self.missing: list[UUID] = []

    def __len__(self) -> int:
        """Get number of tracked objects."""
        return len(self.objects)

    def add_objects(self, objs: list[SceneObject]) -> None:
        for obj in objs:
            self.add_object(obj)

    def add_object(self, obj: SceneObject) -> None:
        if obj.id in self.objects:
            logger.error(f"Object with id {obj.id} already exists.")
            raise ValueError(f"Object with id {obj.id} already exists.")

        self.objects[obj.id] = obj
        if obj.id not in self.active:
            self.active.append(obj.id)

    def delete_object(self, obj_id: UUID) -> None:
        if obj_id not in self.objects:
            logger.error(f"Object with id {obj_id} does not exist.")
            raise ValueError(f"Object with id {obj_id} does not exist.")

        self.objects.pop(obj_id)
        if obj_id in self.active:
            self.active.remove(obj_id)
        if obj_id in self.missing:
            self.missing.remove(obj_id)

    def mark_missing(self, obj_id: UUID | SceneObject | Iterable[Union[UUID, SceneObject]], time_sec: float) -> None:
        if not isinstance(obj_id, Iterable):
            obj_id = [obj_id]
        for id in obj_id:
            if isinstance(id, SceneObject):
                id = id.id
            if id not in self.objects:
                logger.error(f"Object with id {id} does not exist.")
                raise ValueError(f"Object with id {id} does not exist.")
            if id in self.active:
                self.active.remove(id)
            if id not in self.missing:
                self.missing.append(id)
            self.objects[id].time_of_disappearance = time_sec

    def mark_active(self, obj_id: UUID) -> None:
        if obj_id not in self.objects:
            logger.error(f"Object with id {obj_id} does not exist.")
            raise ValueError(f"Object with id {obj_id} does not exist.")
        if obj_id in self.missing:
            self.missing.remove(obj_id)
        if obj_id not in self.active:
            self.active.append(obj_id)

    def active_objects(self) -> Iterator[SceneObject]:
        """Get iterator over active objects."""
        for obj_id in self.active:
            yield self.objects[obj_id]

    def missing_objects(self) -> Iterator[SceneObject]:
        """Get iterator over missing objects."""
        for obj_id in self.missing:
            yield self.objects[obj_id]

    def get_stacked_bounding_box(self, ids: Optional[list[UUID]] = None) -> Float[torch.Tensor, "N attr"]:
        if ids is None:
            ids = list(self.objects.keys())
        bbox_points = torch.stack(
            [torch.from_numpy(self.objects[obj_id].bounding_box.get_box_points()).to(torch.float32) for obj_id in ids]
        )
        return bbox_points

    def get_stacked_clip_features(self, ids: Optional[list[UUID]] = None) -> Float[torch.Tensor, "N feat"]:
        if ids is None:
            ids = list(self.objects.keys())
        clip_features = torch.stack([self.objects[obj_id].visual_feature for obj_id in ids])
        return clip_features

    def merge_from_adjacency(
        self,
        detections: Detections,
        detections_clip_features: Float[torch.Tensor, "N feat"],
        time_sec: float,
        adjacency: ObjectDetectionAdjacency,
        downsample_voxel_size: float = 0.005,
    ) -> None:
        """Merge detections into the tracker, updating existing objects and adding new ones. The `adjacency` matrix is modified in place with the newly added objects.

        Args:
            detections (Detections): Detections instance containing new detections.
            detections_clip_features (Float[torch.Tensor, "N feat"]): CLIP features for each of the N detections.
            time_sec (float): Time of the detections.
            adjacency (ObjectDetectionAdjacency): Adjacency matrix containing associations between detections and existing objects.
            downsample_voxel_size (float): Voxel size to downsample point clouds of modified objects.

        """
        if detections.class_id is None or detections.confidence is None:
            logger.error("Detections must have 'class_id' and 'confidence' to merge.")
            raise ValueError("Detections must have 'class_id' and 'confidence' to merge.")
        if len(detections) != detections_clip_features.shape[0]:
            logger.error("Number of detections must match number of clip features.")
            raise ValueError("Number of detections must match number of clip features.")

        dirty_object_ids: set[UUID] = set()

        # First, merge detections into existing objects
        detections_as_new_objects: list[int] = []
        for i, (class_id, confidence, clip_features, pcd) in enumerate(
            zip(
                detections.class_id,
                detections.confidence,
                detections_clip_features,
                detections.data["point_clouds"],
            )
        ):
            matched_id, match_type, _ = adjacency.get_merge_target(i)
            if matched_id is None or match_type is None:
                detections_as_new_objects.append(i)
            else:
                self.objects[matched_id].merge_with_detection(
                    int(class_id),
                    float(confidence),
                    pcd,
                    clip_features,
                    time_sec,
                    match_type == AssociationType.TRANSLATE,
                )
                self.mark_active(matched_id)
                dirty_object_ids.add(matched_id)

        # Traverse adjacency matrix in depth-first manner to merge objects that have been matched to each other (excluding detections as these have been processed above)
        # Add all the objects which are only merged into as root nodes
        stack = [
            (obj, False)
            for obj in self.objects.values()
            if not adjacency.is_merged_source(obj.id, include_detections=False)
        ]
        merge_counter = 0
        while len(stack) > 0:
            obj, childs_added = stack.pop()
            if childs_added:
                # All children have been processed, now merge this into the parent
                parent_id, _, _ = adjacency.get_merge_target(obj.id, include_detections=False)
                if parent_id is not None:
                    self.objects[parent_id].merge_with_object(obj, replace_pointcloud=True)
                    if obj.id in self.active:
                        self.mark_active(parent_id)
                    self.delete_object(obj.id)
                    merge_counter += 1
                    dirty_object_ids.add(parent_id)
                    dirty_object_ids.discard(obj.id)
            else:
                # Add children to stack
                stack.append((obj, True))
                for child_id, _, _ in adjacency.get_merge_sources(obj.id, include_detections=False):
                    if (self.objects[child_id], False) in stack or (self.objects[child_id], True) in stack:  # type: ignore (child_id is UUID since we cleared detection edges)
                        raise ValueError("Cycle detected in object-object adjacency graph.")
                    stack.append((self.objects[child_id], False))  # type: ignore (child_id is UUID since we cleared detection edges)

        # Add unmatched detections as new objects
        self.add_objects(
            new_objects := SceneObject.generate_from_detections(
                detection=detections[detections_as_new_objects],  # type: ignore
                clip_features=detections_clip_features[detections_as_new_objects, :],
                time_sec=time_sec,
            )
        )
        adjacency.add_object_nodes(obj.id for obj in new_objects)
        for detection, object in zip(detections_as_new_objects, new_objects):
            adjacency.add_edge(detection, object.id, AssociationType.MERGE, None)

        dirty_object_ids.update(obj.id for obj in new_objects)

        # Downsample point clouds of dirty objects to save memory
        for id in dirty_object_ids:
            self.objects[id].geometry.postprocess_geometry(downsample_voxel_size)

        logger.debug(
            f"Added {len(new_objects)} new objects, merged {len(detections) - len(new_objects)} detections with objects, merged {merge_counter} existing objects."
        )

    def updated_measured_change(
        self,
        detections: Detections,
        inview_object_ids: Sequence[UUID],
        adjacency: ObjectDetectionAdjacency,
        config: ObjectMatchingConfig = ObjectMatchingConfig(),
    ) -> None:
        """Measure change for objects based on new detections.

        Detected objects have their change measured using ICP between the stored point cloud and the detection point cloud.
        Object expected in the current view ('inview_object_ids') receive the default change value if not detected.
        Objects not expected in view receive the change measurement 'None'.

        Args:
            detections (Detections): Detections instance containing new detections.
            inview_object_ids (list[UUID]): List of UUIDs of objects expected to be in view.
            adjacency (ObjectDetectionAdjacency): Adjacency matrix containing associations between detections and existing objects.
            config (ObjectMatchingConfig): Configuration parameters for object matching and change measurement. Used parameters include:
                - pocd_default_change_meters_mean
                - pocd_default_change_meters_standard_deviation
                - icp_threshold
                - icp_max_iteration

        Returns:
            None: The function modifies the 'latest_change' attribute of SceneObject instances in place.

        """
        if detections.data is None or "point_clouds" not in detections.data:
            logger.error("Detections must have 'point_clouds' in their data to measure change.")
            raise ValueError("Detections must have 'point_clouds' in their data to measure change.")

        if "matching_transformations" not in detections.data:
            detections.data["matching_transformations"] = [None] * len(detections)

        for objs in self.objects.values():
            objs.current_change = None
        for obj_id in inview_object_ids:
            self.objects[obj_id].current_change = StandardDistributedValue(
                mean=config.pocd_default_change_meters_mean, std=config.pocd_default_change_meters_standard_deviation
            )
        for i, (detection_point_cloud, transformation) in enumerate(
            zip(detections.data["point_clouds"], detections.data["matching_transformations"])
        ):
            object_id, _, transformation = adjacency.get_merge_target(i)
            if object_id is None:
                continue
            # Object has been observed. Evaluate change magnitude with ICP
            if transformation is None:
                registration_results = detection_point_cloud.register_geometries(
                    self.objects[object_id].geometry, np.eye(4), config.icp_threshold, config.icp_max_iteration
                )
                transformation = registration_results.transformation
                adjacency.add_edge(i, object_id, AssociationType.MERGE, torch.tensor(transformation))
            self.objects[object_id].current_change = StandardDistributedValue(
                mean=np.linalg.norm(transformation[:3, 3]),  # type: ignore
                std=config.pocd_default_change_meters_standard_deviation,
            )

    def inject_fake_measured_change(
        self,
        pocd_decay_stationarity_threshold: float,
        fake_changes: dict[POCDObjectTypes, tuple[float, StandardDistributedValue]],
        time_sec: float,
    ) -> list[UUID]:
        """Inject fake change measurements for objects that have not been measured recently (these are objects which the camera did not look at for some time).

        Args:
            pocd_decay_stationarity_threshold (float): Minimum stationarity confidence to consider injecting fake change measurements.
            fake_changes (dict[POCDObjectTypes, tuple[float, StandardDistributedValue]]): Dictionary mapping object types to a tuple of (period of applying the fake change measurements, fake change measurement).
            time_sec (float): Current time in seconds.

        Returns:
            list[UUID]: List of UUIDs of objects for which fake change measurements were injected.

        """
        injected_ids = []
        for obj in self.active_objects():
            if obj.current_change is not None:
                continue
            if obj.stationarity_confidence <= pocd_decay_stationarity_threshold:
                continue
            if obj.stationarity_prior not in fake_changes:
                continue
            min_time_diff, change = fake_changes[obj.stationarity_prior]
            if (
                time_sec - obj.last_observed_time() > min_time_diff
                and time_sec - obj.last_decay_fake_measurement_time > min_time_diff
            ):
                obj.current_change = change
                obj.last_decay_fake_measurement_time = time_sec
                injected_ids.append(obj.id)
        logger.debug(f"Injected fake change measurement for {len(injected_ids)} objects.")
        return injected_ids

    def update_stationarity_confidences(self) -> None:
        for obj in self.active_objects():
            if obj.current_change is None:
                continue
            obj.update_pocd_probabilities()

    def filter_objects(
        self,
        min_observations: int = 2,
        min_points: int = 10,
    ):
        remove_ids = []
        for obj_id, obj in self.objects.items():
            if obj.number_of_observations() < min_observations:
                remove_ids.append(obj_id)
            elif len(obj.geometry) < min_points:
                remove_ids.append(obj_id)

        logger.debug(f"Filtered out {len(remove_ids)} objects.")
        for obj_id in remove_ids:
            self.delete_object(obj_id)

    def merge_overlapping_objects(
        self,
        downsample_voxel_size: float = 0.01,
        merge_overlap_threshold: float = 0.7,
        merge_visual_threshold: float = 0.8,
    ) -> None:
        """Merge active objects that have a high geometric overlap and high visual similarity.

        Args:
            downsample_voxel_size (float): Voxel size to downsample point clouds for overlap computation.
            merge_overlap_threshold (float): Minimum overlap ratio to consider merging two objects.
            merge_visual_threshold (float): Minimum visual similarity to consider merging two objects.

        """
        if len(self.active) <= 1:
            return

        overlap_matrix = compute_overlap_matrix_general(
            point_clouds_a=[obj.geometry for obj in self.active_objects()],
            bounding_boxes_a=[obj.bounding_box for obj in self.active_objects()],
            downsample_voxel_size=downsample_voxel_size,
        )

        # make upper triangle zero such that each match is only considered once
        overlap_matrix = torch.tril(overlap_matrix)

        x, y = overlap_matrix.nonzero().T
        overlap_ratio = overlap_matrix[x, y]

        # Sort indices of overlap ratios in descending order
        sort = torch.argsort(overlap_ratio, descending=True)
        x = x[sort]
        y = y[sort]
        overlap_ratio = overlap_ratio[sort]

        # ids of objects
        x_id = [self.active[i] for i in x]
        y_id = [self.active[i] for i in y]

        already_merged = set()
        for id_0, id_1, ratio in zip(x_id, y_id, overlap_ratio):
            if ratio < merge_overlap_threshold:
                break  # stop processing if overlap is below threshold (list is sorted)

            if id_0 in already_merged:
                continue

            visual_sim = F.cosine_similarity(
                self.objects[id_0].visual_feature,
                self.objects[id_1].visual_feature,
                dim=0,
            )
            if visual_sim < merge_visual_threshold:
                continue

            target_id, source_id = (
                (id_1, id_0)
                if self.objects[id_1].first_observed_time() < self.objects[id_0].first_observed_time()
                else (id_0, id_1)
            )
            self.objects[target_id].merge_with_object(self.objects[source_id], use_greater_stationarity_estimate=True)
            already_merged.add(source_id)
            logger.debug(
                f"Merged overlapping objects {self.objects[source_id].id_str} and {self.objects[target_id].id_str}"
            )

        for id in already_merged:
            self.delete_object(id)

    def to_dict(self) -> dict[str, Any]:
        obj_ids = [obj.id for obj in self.objects.values()]
        return {
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "active": [obj_ids.index(obj_id) for obj_id in self.active],
            "missing": [obj_ids.index(obj_id) for obj_id in self.missing],
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], device: str = "cuda", geometry_type: Type[GeometryType] = PointCloud
    ) -> "ObjectTracker":
        obj = cls.__new__(cls)
        obj.objects = {
            UUID(obj_data["id"]): SceneObject.from_dict(obj_data, device=device, geometry_type=geometry_type)
            for obj_data in data["objects"]
        }
        obj.active = [UUID(data["objects"][int(idx)]["id"]) for idx in data["active"]]
        obj.missing = [UUID(data["objects"][int(idx)]["id"]) for idx in data["missing"]]
        return obj
