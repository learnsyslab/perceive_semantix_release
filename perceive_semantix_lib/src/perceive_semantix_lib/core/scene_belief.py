import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional
from uuid import UUID

import numpy as np
import torch
from jaxtyping import Bool, Float, UInt8, install_import_hook
from open3d import geometry
from supervision.detection.core import Detections

if os.getenv("ENABLE_RUNTIME_TYPECHECKING", "0") == "1":
    hook = install_import_hook("perceive_semantix_lib", "typeguard.typechecked")
else:
    hook = None
from perceive_semantix_lib.core.background_tracker import BackgroundTracker
from perceive_semantix_lib.core.config import Config
from perceive_semantix_lib.core.geometry import GeometryBase, PointCloud, TensorPointCloud
from perceive_semantix_lib.core.input_types import InputData, InputDataStamped, TorchInputData
from perceive_semantix_lib.core.matching.matching import (
    ObjectDetectionAdjacency,
    SimilarityMatrix,
    compute_spatial_similarities,
    compute_visual_similarities,
    match_detections_to_objects,
    match_detections_to_objects_semantically_greedy_icp,
    match_objects_to_objects,
)
from perceive_semantix_lib.core.object_detector import ObjectDetector
from perceive_semantix_lib.core.object_tracker import ObjectTracker
from perceive_semantix_lib.core.scene_object import SceneObject
from perceive_semantix_lib.core.utils.disk_storage import DiskStorage
from perceive_semantix_lib.core.utils.object_classes import ObjectClasses, init_global_object_classes
from perceive_semantix_lib.core.utils.optional_rerun_wrapper import (
    OptionalReRun,
    orr_log_annotated_image,
    orr_log_associations,
    orr_log_background_point_cloud,
    orr_log_camera,
    orr_log_detections_matches,
    orr_log_expected_object_projection,
    orr_log_point_clouds_bounding_boxes,
    orr_log_scene_objects,
    orr_log_stationarity,
)
from perceive_semantix_lib.core.utils.pocd_datatypes import POCDObjectTypes, StandardDistributedValue
from perceive_semantix_lib.core.utils.visualize_helpers import vis_result_fast
from perceive_semantix_lib.llm.open_ai_client import OpenAIParallelClient
from perceive_semantix_lib.llm.stationarity_prior_client import StationarityPriorOpenAIAsyncClient

if hook is not None:
    hook.uninstall()

logger = logging.getLogger(__name__)


class Scene:
    config: Config

    def __init__(
        self,
        config: Config,
        initial_scene_path: Optional[Path] = None,
        initial_scene_overwrite_time_sec: Optional[float] = None,
    ) -> None:
        self.config = config
        self.pocd_decay_config = {
            POCDObjectTypes.STATIC: (
                self.config.matching.pocd_decay_static_period_seconds,
                StandardDistributedValue(
                    mean=self.config.matching.pocd_decay_static_change_mean_meters,
                    std=self.config.matching.pocd_decay_static_change_std_meters,
                ),
            ),
            POCDObjectTypes.DYNAMIC: (
                self.config.matching.pocd_decay_dynamic_period_seconds,
                StandardDistributedValue(
                    mean=self.config.matching.pocd_decay_dynamic_change_mean_meters,
                    std=self.config.matching.pocd_decay_dynamic_change_std_meters,
                ),
            ),
        }

        if os.getenv("ENABLE_RUNTIME_TYPECHECKING", "0") == "1":
            logger.warning("Runtime type checking is enabled. This may impact performance.")
        if self.config.geometry_type == "pointcloud":
            logger.warning(
                "Using deprecated 'pointcloud' geometry type (old Open3D PointClouds). Consider switching to 'tensor_pointcloud' (GPU supported Open3D PointClouds)."
            )

        self.orr = OptionalReRun()
        self.orr.set_use_rerun(self.config.debug.enable_rerun)
        self.orr.init("perceive_semantix_scene")
        self.orr.spawn()

        self.disk = DiskStorage(Path(self.config.debug.output_root))

        init_global_object_classes(
            ObjectClasses(
                self.config.detection.object_class_names_file,
                bg_classes=self.config.detection.background_class_names,
                skip_bg=self.config.detection.skip_background_classes,
            )
        )
        if self.config.geometry_type == "pointcloud":
            self.geometry_type = PointCloud
        elif self.config.geometry_type == "tensor_pointcloud":
            self.geometry_type = TensorPointCloud
        else:
            raise ValueError(f"Unsupported geometry type: {self.config.geometry_type}")
        self.detector = ObjectDetector(
            self.config.detection, self.config.device, self.geometry_type, self.config.model_weight_dir
        )
        logger.info("ObjectDetector initialized.")

        self._load_initial_map(initial_scene_path, initial_scene_overwrite_time_sec)
        logger.info(f"Scene initialized with config: {self.config}")

        if self.config.semantic_stationarity_prior and OpenAIParallelClient.api_key_is_valid():
            logger.info("Querying LLM for stationarity priors of object classes.")
            SceneObject.stationarity_prior_map = StationarityPriorOpenAIAsyncClient.query_semantic_stationarity(
                self.detector.object_classes.get_classes_arr()
            )

        self.last_frame_time_sec: Optional[float] = None
        self.frame_count: int = 0

    def _load_initial_map(self, path: Optional[Path] = None, shift_to_time_sec: Optional[float] = None) -> None:
        """Load an initial map from the specified path.

        Args:
            path (Path): The file path to load the initial map from.
            shift_to_time_sec (float, optional): If provided, shifts all timestamps in the loaded map to align the last frame time with this value.
                Can be used to synchronize the loaded map with the current time in a live system.

        """
        if path is None:
            self.objects: ObjectTracker = ObjectTracker()
            self.background = BackgroundTracker(geometry_type=self.geometry_type)
            self.previous_camera_pose: Optional[Float[np.ndarray, "4 4"]] = None
        else:
            self.objects, self.background, self.last_frame_time_sec, self.previous_camera_pose = DiskStorage.load_scene(
                path, device=self.config.device, geometry_type=self.geometry_type
            )
            if shift_to_time_sec is not None:
                time_shift = shift_to_time_sec - self.last_frame_time_sec
                for obj in self.objects.objects.values():
                    obj.shift_all_recorded_timestamps(time_shift)
                self.last_frame_time_sec = shift_to_time_sec
            self.orr.set_time("scene", timestamp=self.last_frame_time_sec)
            orr_log_stationarity(self.objects.objects.values())
            orr_log_scene_objects(self.objects.active_objects(), self.detector.object_classes)
            detections = Detections.empty()
            orr_log_associations(
                self.objects.objects,
                self.objects.active,
                detections,
                ObjectDetectionAdjacency(detections, list(self.objects.objects.keys())),
                self.last_frame_time_sec,
            )

    def _log_input(self, time_sec: float, input_data: Optional[InputData]) -> None:
        self.orr.set_time("scene", timestamp=time_sec)
        if input_data is not None:
            if self.config.debug.store_input_each_frame:
                color, _ = self.disk.save_input(input_data, time_sec)
            else:
                color = input_data.color
            orr_log_camera(
                intrinsics=input_data.camera_intrinsics,
                pose=input_data.pose,
                image=color,
                depth_image=input_data.depth.squeeze(-1),
                img_width=self.config.input_image_width,
                img_height=self.config.input_image_height,
                time_sec=time_sec,
                prev_pose=self.previous_camera_pose,
            )

    def _log_detections(
        self,
        color_image: UInt8[np.ndarray, "H W 3"],
        detections: Detections,
        time_sec: float,
    ) -> None:
        if self.config.debug.rerun_visualize_detections_2d and self.config.debug.enable_rerun:
            annotated_color_image = vis_result_fast(
                color_image, detections, self.detector.object_classes.get_classes_arr()
            )
            annotated_color_path = self.disk.save_detectons(annotated_color_image, time_sec)
            orr_log_annotated_image(annotated_color_path)
        if self.config.debug.rerun_visualize_detections:
            orr_log_point_clouds_bounding_boxes(detections, time_sec, self.detector.object_classes)

    def _log_detection_background_points(self, background_points: geometry.PointCloud) -> None:
        if self.config.debug.rerun_visualize_detection_background_points:
            orr_log_background_point_cloud(background_points, log_as_detection=True)

    def _log_background_points(self, background_points: GeometryBase) -> None:
        if self.config.debug.rerun_visualize_background_points:
            orr_log_background_point_cloud(background_points, log_as_detection=False)

    def _log_expected_objects_projections(self, object_projections: Bool[torch.Tensor, "N H W"]) -> None:
        if self.config.debug.rerun_visualize_expected_objects and object_projections.shape[0] > 0:
            orr_log_expected_object_projection(object_projections[0, ...])

    def _log_detections_matches(
        self,
        detections: Detections,
        time_sec: float,
        objects: dict[UUID, SceneObject],
        adjacency: ObjectDetectionAdjacency,
    ) -> None:
        if self.config.debug.rerun_visualize_object_matches:
            orr_log_detections_matches(detections, time_sec, objects, adjacency)

    def _match_detections_to_objects(
        self,
        tuple_detections_clip_features: tuple[Detections, Float[torch.Tensor, "candidates clip"]],
        inview_object_ids: List[UUID],
        inview_visibility_ratios: List[float],
        inview_projections: Optional[Bool[torch.Tensor, "N H W"]],
        invalid_depth_measurement_mask: Optional[Bool[torch.Tensor, "H W"]],
        adjacency: ObjectDetectionAdjacency,
    ) -> tuple[ObjectDetectionAdjacency, SimilarityMatrix, SimilarityMatrix]:
        logger.debug("Computing spatial similarities between existing objects and new detections.")
        inview_spatial_similarity = compute_spatial_similarities(
            [self.objects.objects[k] for k in inview_object_ids],
            tuple_detections_clip_features[0],
            objects_visibility_ratios=inview_visibility_ratios,
            downsample_voxel_size=self.config.downsample_voxel_size,
            compensate_invalid_depth=self.config.matching.compensate_invalid_depth,
            object_projections=inview_projections,
            invalid_depth_measurement=invalid_depth_measurement_mask,
        )

        spatial_similarity = SimilarityMatrix(
            self.objects.active,
            torch.full(
                (len(self.objects.active), len(tuple_detections_clip_features[0])),
                0,
                dtype=inview_spatial_similarity.dtype,
                device=inview_spatial_similarity.device,
            ),
        )
        indices = [self.objects.active.index(id) for id in inview_object_ids]
        spatial_similarity.data[indices, :] = inview_spatial_similarity

        logger.debug("Computing visual similarities between existing objects and new detections.")
        visual_similarity = compute_visual_similarities(
            list(self.objects.active_objects()), tuple_detections_clip_features[1]
        )

        match_detections_to_objects(
            tuple_detections_clip_features[0],
            spatial_similarity,
            visual_similarity,
            spatial_threshold=self.config.matching.spatial_similarity_threshold,
            visual_threshold=self.config.matching.visual_similarity_threshold,
            prioritize_semantic_sim=self.config.matching.prioritize_semantic_similarity,
            matching_method=self.config.matching.matching_method,
            adjacency=adjacency,
        )
        return adjacency, spatial_similarity, visual_similarity

    def get_objects(self, min_number_of_observations: int = 0) -> Iterator[SceneObject]:
        """Get all objects in the scene with at least `min_number_of_observations` observations.

        Args:
            min_number_of_observations (int, optional): Minimum number of observations an object must have to be included. Defaults to 0.

        Returns:
            Iterator[SceneObject]: An iterator over the objects meeting the criteria.

        """
        yield from (
            obj for obj in self.objects.active_objects() if len(obj.observation_times) >= min_number_of_observations
        )

    def get_background(self) -> GeometryBase:
        """Get the current background geometry.

        Returns:
            GeometryBase: The current background geometry.

        """
        with self.background.lock:
            return self.background.geometry

    def step(self, posed_rgbd: InputDataStamped) -> None:
        """Process a new input frame, updating the scene's belief state.

        Args:
            posed_rgbd (InputDataStamped): The input data for the current frame. If `input_posed_rgbd.data` is
                `None` the statonarity confidences of existing objects will still be updated (i.e. decay over time).

        """
        logger.info(f"################### frame {self.frame_count} ({posed_rgbd.time_sec:0.2f})")
        logger.debug(f"active objects: {[o.id_str for o in self.objects.active_objects()]}")
        self._log_input(posed_rgbd.time_sec, posed_rgbd.data)

        if posed_rgbd.data is not None:
            if not posed_rgbd.data.verify():
                logger.error("Input data verification failed. Skipping this frame.")
                return
            torch_input_data = TorchInputData.from_input_data(posed_rgbd.data)

            # Obtain current-view detections
            detections, clip_features, background_pc = self.detector.detect_3d(
                torch_input_data.color,
                torch_input_data.depth,
                torch_input_data.pose,
                torch_input_data.camera_intrinsics,
            )

            self.background.async_add_background_points(
                background_pc,
                self.config.downsample_background_voxel_size,
                current_xy=tuple(torch_input_data.pose[:2, 3].cpu().tolist()),
            )
            detections = self.detector.post_process_3d_detections(detections, self.config.downsample_voxel_size)
            self._log_detections(posed_rgbd.data.color, detections, posed_rgbd.time_sec)
            self._log_detection_background_points(background_pc)

            # Extract which objects from the map are expected to be in view
            inview_object_ids, inview_visibility_ratios, inview_projections = SceneObject.project_onto_camera(
                list(self.objects.active_objects()),
                torch_input_data.pose,
                torch_input_data.camera_intrinsics,
                self.config.input_image_height,
                self.config.input_image_width,
                self.config.detection.min_depth_m,
                self.config.detection.max_depth_m,
                device=self.config.device,
                visibility_threshold=self.config.pocd_visibility_threshold,
                preallocate_object_mask_count=self.config.preallocate_visible_objects_mask_count,
            )
            self._log_expected_objects_projections(inview_projections)
            logger.error(
                f"Expected objects in view: {[str(id)[:4] + self.objects.objects[id].class_name for id in inview_object_ids]}"
            )

            invalid_depth_measurement_mask = torch.logical_or(
                torch.logical_or(
                    torch_input_data.depth[0, 0, ...] == 0,
                    torch_input_data.depth[0, 0, ...] > self.config.detection.max_depth_m,
                ),
                torch_input_data.depth[0, 0, ...] < self.config.detection.min_depth_m,
            )

        else:
            logger.debug("Got empty input data")
            detections = Detections.empty()
            detections.data = {"classes": [], "point_clouds": [], "bounding_boxes": []}
            clip_features = torch.empty((0, 0))
            inview_object_ids: list[UUID] = []
            inview_visibility_ratios: list[float] = []
            inview_projections = torch.zeros(
                (0, self.config.input_image_height, self.config.input_image_width),
                dtype=torch.bool,
                device=self.config.device,
            )
            invalid_depth_measurement_mask = None

        object_adjacency = ObjectDetectionAdjacency(detections, list(self.objects.objects.keys()))

        # Match detections to existing objects (while not allowing spatial displacement)
        object_adjacency, spatial_similarity, visual_similarity = self._match_detections_to_objects(
            (detections, clip_features),
            inview_object_ids,
            inview_visibility_ratios,
            inview_projections,
            invalid_depth_measurement_mask,
            object_adjacency,
        )
        self._log_detections_matches(detections, posed_rgbd.time_sec, self.objects.objects, object_adjacency)

        self.objects.updated_measured_change(detections, inview_object_ids, object_adjacency, self.config.matching)
        objects_with_injected_changes = self.objects.inject_fake_measured_change(
            self.config.matching.pocd_decay_stationarity_threshold, self.pocd_decay_config, posed_rgbd.time_sec
        )

        self.objects.update_stationarity_confidences()
        for obj in (self.objects.objects[id] for id in objects_with_injected_changes):
            obj.stationarity_confidence = max(
                obj.stationarity_confidence, self.config.matching.pocd_decay_stationarity_threshold
            )

        # Mark active objects which are expected in the current view and have not been matched to a detection as "disappeared"
        # Try to associate these with detections using a strategy which allows for spatial displacement
        disappeared_objects = [
            self.objects.objects[id] for id in inview_object_ids if not object_adjacency.is_merged_target(id)
        ]
        for obj in disappeared_objects:
            obj.time_of_disappearance = posed_rgbd.time_sec
        match_detections_to_objects_semantically_greedy_icp(
            disappeared_objects,
            detections,
            object_adjacency,
            visual_similarity,
            self.config.matching,
        )

        # Try to match objects which are potentially translated to other existing objects
        potentially_translated_objects = [
            obj
            for obj in self.objects.active_objects()
            if obj.stationarity_confidence < self.config.matching.pocd_transformation_threshold
            and obj.stationarity_confidence >= self.config.matching.pocd_removal_threshold
            and obj not in disappeared_objects
            and not object_adjacency.is_merged_target(obj.id)
        ]
        logger.debug(f"number of potentially_translated_objects: {len(potentially_translated_objects)}")
        match_objects_to_objects(
            potentially_translated_objects,
            tuple(self.objects.active_objects()),
            object_adjacency,
            self.last_frame_time_sec,
            self.config.matching,
        )

        # Update list of missing objects and try to reidentify them
        new_missing_objects = [
            obj
            for obj in self.objects.active_objects()
            if obj.stationarity_confidence < self.config.matching.pocd_removal_threshold
            and not object_adjacency.is_merged_target(obj.id)
            and not object_adjacency.is_merged_source(obj.id)
        ]
        self.objects.mark_missing(new_missing_objects, posed_rgbd.time_sec)
        logger.debug(f"number of active_objects: {len(self.objects.active)}")
        logger.debug(f"missing_objects: {[o.id_str for o in self.objects.missing_objects()]}")
        match_objects_to_objects(
            tuple(self.objects.missing_objects()),
            tuple(o for o in self.objects.active_objects() if not object_adjacency.is_merged_source(o.id)),
            object_adjacency,
            self.last_frame_time_sec,
            self.config.matching,
        )

        self.objects.merge_from_adjacency(
            detections, clip_features, posed_rgbd.time_sec, object_adjacency, self.config.downsample_voxel_size
        )
        orr_log_associations(
            self.objects.objects, self.objects.active, detections, object_adjacency, posed_rgbd.time_sec
        )

        # Apply post-processing steps periodically
        self.frame_count += 1
        if self.frame_count % self.config.merging_interval_frames == 0:
            before_merging = len(self.objects.active)
            self.objects.merge_overlapping_objects(
                downsample_voxel_size=self.config.downsample_voxel_size,
                merge_overlap_threshold=self.config.merge_overlap_threshold,
                merge_visual_threshold=self.config.merge_visual_threshold,
            )
            logger.debug(
                f"Merged {before_merging - len(self.objects.active)} overlapping objects. Active objects count: {len(self.objects.active)}"
            )

        if self.frame_count % self.config.filtering_interval_frames == 0:
            self.objects.filter_objects(self.config.filtering_min_observations, self.config.filtering_min_points)

        orr_log_stationarity(self.objects.objects.values())
        orr_log_scene_objects(self.objects.active_objects(), self.detector.object_classes)
        self.background.wait_until_processed()
        with self.background.lock:
            self._log_background_points(self.background.geometry)

        self.last_frame_time_sec = posed_rgbd.time_sec
        if posed_rgbd.data is not None:
            self.previous_camera_pose = posed_rgbd.data.pose

        if self.config.debug.store_scene_each_frame:
            self.disk.save_scene(
                self.objects,
                self.background,
                posed_rgbd.time_sec,
                posed_rgbd.data.pose if posed_rgbd.data is not None else None,
            )
