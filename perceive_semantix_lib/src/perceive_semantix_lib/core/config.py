from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

# TODO In the future these could be used together with hydra as structured configurations (https://hydra.cc/docs/tutorials/structured_config/intro/)


@dataclass(frozen=True)
class DetectionConfig:
    """Parameters for extracting objects from RGB-D frames.

    Attributes:
        blur_threshold (float): Frames with variance of Laplacian below this threshold are considered blurry.
            Default is 60.0.

        yolo_model_name (str, optional): Name of the object detection model.
            See https://docs.ultralytics.com/models for available models.
            Default is "yolov8l-world".

        object_class_names_file (Path, optional): Path to a file containing object class names (one per line). For the object detection model.
            If None, raises an error. Default is None.

        background_class_names (list[str]): List of class names considered as background.
            Default is ["wall", "floor", "ceiling"].

        skip_background_classes (bool): Whether to skip detections of background classes.
            Default is True.

        object_min_detection_confidence (float): Minimum confidence for object detections.
            Detections with confidence below this threshold are ignored. Default is 0.25.

        max_bbox_area_ratio (float, optional): Maximum bounding box area ratio.
            Bounding boxes covering more than this fraction of the image area are ignored. Default is 0.5.

        mask_erosion_radius (int): Radius for erosion (in pixels) of object masks (by how much the masks are shrunk) to avoid artifacts at the edges.
            Depth information near the edges of the masks are often noisy and can lead to faulty 3D object reconstructions. Default is 3.

        sam_model_name (str): Name of the object segmentation model.
            See https://docs.ultralytics.com/models for available models. Default is "mobile_sam".

        mask_area_threshold (float): Minimum mask area threshold. Masks with less pixels are ignored. Default is 25.

        clip_model_name (str): Name of the CLIP model for visual feature extraction.
            See https://github.com/mlfoundations/open_clip for available models. Default is "ViT-H-14".

        clip_pretrained (str): Pretrained weights for the CLIP model. Default is "laion2b_s32b_b79k".

        min_points_threshold (int): Objects with less than this number of pixels with valid depth measurements (not downsampled) are ignored.
            Default is 100.

        obj_pcd_max_points (int): Maximum points per object point cloud. Larger objects are downsampled. Default is 5000.

        min_bbox_volume (float): Objects with 3d bounding box volume below this threshold are ignored. Default is 1e-6m^3.

        min_depth_m (float): Minimum depth in meters. Only consider points within this depth range. Default is 0.5m.

        max_depth_m (float): Maximum depth in meters. Only consider points within this depth range. Default is 2.0m.

    """

    blur_threshold: float = 60.0

    yolo_model_name: str = "yolov8l-world"
    object_class_names_file: Optional[Path] = None
    background_class_names: list[str] = field(default_factory=lambda: ["wall", "floor", "ceiling"])
    skip_background_classes: bool = True
    object_min_detection_confidence: float = 0.25
    max_bbox_area_ratio: Optional[float] = 0.5
    mask_erosion_radius: int = 3

    sam_model_name: str = "mobile_sam"
    mask_area_threshold: float = 25

    clip_model_name: str = "ViT-H-14"
    clip_pretrained: str = "laion2b_s32b_b79k"

    min_points_threshold: int = 100
    obj_pcd_max_points: int = 5000
    min_bbox_volume: float = 1e-6

    min_depth_m: float = 0.5
    max_depth_m: float = 2.0


@dataclass(frozen=True)
class ObjectMatchingConfig:
    """Parameters for matching newly detected objects to existing objects in the scene and for re-identification.

    Attributes:
        compensate_invalid_depth (bool): Whether to compensate for invalid depth measurements during matching. If True, pixels where no depth measurement could be made are accounted for in the geometric similarity metric.
            Default is True.

        spatial_similarity_threshold (float): Threshold for spatial similarity when matching objects. Default is 0.4.

        visual_similarity_threshold (float): Threshold for visual similarity when matching objects. Default is 0.6.

        visual_similarity_threshold_reidentification (float): Threshold for visual similarity during re-identification.

        prioritize_semantic_similarity (bool): Whether to prioritize semantic similarity over visual similarity during matching of detections to objects. Default is False.

        matching_method (str): Method used for matching objects. Currently only "sep_thresh" (separate thresholds for spatial and visual similarity which both must be surpassed) is supported. Default is "sep_thresh".

        icp_threshold (float): Maximum correspondence distance for ICP algorithm (see https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.registration_icp.html). Default is 0.025.

        icp_max_iteration (int): Maximum number of iterations for ICP algorithm. Default is 100.

        icp_inlier_rmse_threshold (float): Inlier RMSE threshold for ICP algorithm. Default is 100.

        pocd_default_change_meters_mean (float): Default POCD change mean (which is applied when object disappears) in meters. Default is 30.

        pocd_default_change_meters_standard_deviation (float): Default POCD change standard deviation (which is applied when object disappears) in meters. Default is 0.5.

        pocd_decay_stationarity_threshold (float): Only objects with stationarity probability above this threshold will have their POCD decay applied. Default is 0.6.

        pocd_decay_static_period_seconds (float): Duration for which static POCD decay is applied in seconds. Default is 1800.
        pocd_decay_static_change_mean_meters (float): Mean of the Gaussian for static POCD decay change in meters. Default is 2.
        pocd_decay_static_change_std_meters (float): Standard deviation of the Gaussian for static POCD decay change in meters. Default is 0.5.

        pocd_decay_dynamic_period_seconds (float): Default is 15.
        pocd_decay_dynamic_change_mean_meters (float): Default is 2.
        pocd_decay_dynamic_change_std_meters (float): Default is 0.5.

        pocd_removal_threshold (float): Objects with POCD below this threshold will be removed from the scene. Default is 0.5.
        pocd_transformation_threshold (float): Objects with POCD below this threshold are considered to have transformed. Default is 0.6.

        reidentification_lookforward_seconds (float): Duration in seconds to look forward for re-identifying objects. Default is 30.
        reidentification_lookback_seconds (float): Duration in seconds to look back for re-identifying objects. Default is 30.0.
        reidentification_merge_vs_translate_threshold_meters (float): During re-identification, when two objects which are matched to
            each other are within this distance they are merged, otherwise they are translated. Default is 0.05.

    """

    compensate_invalid_depth: bool = True
    spatial_similarity_threshold: float = 0.4
    visual_similarity_threshold: float = 0.6
    visual_similarity_threshold_reidentification: float = 0.7
    prioritize_semantic_similarity: bool = False
    matching_method: str = "sep_thresh"

    icp_threshold: float = 0.025
    icp_max_iteration: int = 100
    icp_inlier_rmse_threshold: float = 100  # TODO check this parameter

    pocd_default_change_meters_mean: float = 30
    pocd_default_change_meters_standard_deviation: float = 0.5

    pocd_decay_stationarity_threshold: float = 0.6
    pocd_decay_static_period_seconds: float = 1800
    pocd_decay_static_change_mean_meters: float = 2
    pocd_decay_static_change_std_meters: float = 0.5

    pocd_decay_dynamic_period_seconds: float = 15
    pocd_decay_dynamic_change_mean_meters: float = 2
    pocd_decay_dynamic_change_std_meters: float = 0.5

    pocd_removal_threshold: float = 0.5
    pocd_transformation_threshold: float = 0.6

    reidentification_lookforward_seconds: float = 30
    reidentification_lookback_seconds: float = 30.0
    reidentification_merge_vs_translate_threshold_meters: float = 0.05


@dataclass(frozen=True)
class BackgroundConfig:
    """Parameters for the OctoMap-backed background occupancy tracker.

    Attributes:
        max_range_m (float): Maximum ray length (in meters) considered when raycasting background points into
            the occupancy octree. Passing -1 disables the limit (the full ray from sensor origin to each measured
            point is always inserted, regardless of length). Background points are not otherwise range-limited
            (unlike `DetectionConfig.min_depth_m`/`max_depth_m`, which only apply to per-object detections).
            Default is -1.0 (unlimited).

        store_color (bool): Whether to store per-voxel RGB color (averaged from the input frames) in the
            background occupancy octree, backed by a `ColorOcTree` instead of a plain `OcTree`. Increases memory
            usage and octree serialization size. When disabled, background voxels carry no color information and
            are rendered in a single flat gray when logged to rerun. Default is False.

        resolution_m (float): Resolution (in meters) of the background occupancy octree, i.e. minimum voxel size. Default is 0.05.

        lazy_eval (bool): Whether to merge collapse inner nodes whose children are all occupied or all free,
            lazily (True) after each frame's background point cloud is inserted (False). This can reduce memory
            usage and octree serialization size, but increases the time taken to insert each frame's background points.
            Default is False.

        rerun_chunk_size_m (float): Size (in meters, along the XY plane only - each chunk spans the full Z
            range) of the square chunks used to split the background octree into independently-addressed rerun
            entities, so that only chunks whose content changed in the current frame need to be re-logged.
            Default is 2.0.

        prob_hit (float, optional): Sensor model parameter for octomap P(occupied | hit). The probability that a voxel is actually
            occupied, given that a ray's endpoint (a depth measurement) landed inside it (a "hit"). Higher values
            make voxels become confidently occupied faster. Passing None uses pyoctomap's default (0.7). Default is None.

        prob_miss (float, optional): Sensor model parameter for octomap P(occupied | miss). The probability that a voxel is actually
            occupied, given that a ray passed through it as free space (a "miss"). Lower values carve voxels free
            faster. Passing None uses pyoctomap's default (0.4). Default is None.

    """

    max_range_m: float = 5.0
    store_color: bool = False
    resolution_m: float = 0.05
    lazy_eval: bool = False
    rerun_chunk_size_m: float = 1.0
    prob_hit: Optional[float] = None
    prob_miss: Optional[float] = 0.3


@dataclass(frozen=True)
class DebugConfig:
    """Debugging and visualization parameters.

    Attributes:
        output_root (Path = "./scene_timestamp"): Root directory for storing output data.
        store_input_each_frame (bool = False): Whether to store the input data for each frame.
        store_scene_each_frame (bool = False): Whether to store the scene state for each frame.

        enable_rerun (bool = True): Whether to enable rerun visualizations.
        rerun_launch_mode (Literal["spawn", "serve_grpc", "connect_grpc"] = "serve_grpc"): How to launch/connect to rerun.
            "spawn" opens a new local rerun viewer window. "serve_grpc" starts a gRPC server that a separately-launched
            viewer can connect to. "connect_grpc" connects to an already-running rerun gRPC server/proxy at
            `rerun_grpc_connect_url`. Default is "serve_grpc" ("spawn" is currently broken with rerun-sdk 0.34.1: it
            launches the viewer with `RERUN_APP_ONLY=true` set, which crashes rerun's own CLI entry point with
            `ImportError: cannot import name '_dec_active_tracing_sessions'` - an upstream bug, not fixable here).
        rerun_grpc_connect_url (str = "rerun+http://127.0.0.1:9876/proxy"): URL of the rerun gRPC server/proxy to
            connect to when `rerun_launch_mode` is "connect_grpc". Unused otherwise.
        rerun_visualize_detections_2d (bool = False): Whether to visualize 2D detections (segmented input image) in rerun.
        rerun_visualize_detections (bool = False): Whether to visualize 3D detections in rerun.
        rerun_visualize_detection_background_points (bool = False): Whether to visualize background points of detections in rerun.
        rerun_visualize_background_points (bool = False): Whether to visualize the background occupancy octree (as chunked point clouds) in rerun.
        rerun_visualize_expected_objects (bool = False): Whether to visualize the projection of expected objects onto the camera in rerun.
        rerun_visualize_object_matches (bool = False): Whether to visualize object matches in rerun.

    """

    output_root: Path = field(default_factory=lambda: Path("./scene_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    store_input_each_frame: bool = False
    store_scene_each_frame: bool = False

    enable_rerun: bool = True
    rerun_launch_mode: Literal["spawn", "serve_grpc", "connect_grpc"] = "spawn"
    rerun_grpc_connect_url: str = "rerun+http://127.0.0.1:9876/proxy"
    rerun_visualize_detections_2d: bool = False
    rerun_visualize_detections: bool = False
    rerun_visualize_detection_background_points: bool = False
    rerun_visualize_background_points: bool = False
    rerun_visualize_expected_objects: bool = False
    rerun_visualize_object_matches: bool = False


@dataclass(frozen=True)
class Config:
    r"""Main configuration class for the system.

    Attributes:
        input_image_height (int = 640): Height of the input images.
        input_image_width (int = 360): Width of the input images.

        load_inital_map_path (str = ""): Path to load an initial map from. If not empty, must point to a directory produced by a previous run with setting ``debug.store_scene_each_frame = True``.

        pocd_visibility_threshold (float = 0.1): When `pocd_visibility_threshold` of the object's points are visible in the current camera fov it is considered visible.
        preallocate_visible_objects_mask_count (int = 10): For each expected (i.e., in-view) object a mask of its projection is returned. This parameter controls how many mask tensors are preallocated. Affects computation speed.
        downsample_voxel_size (float = 0.025): Minimum distance between point cloud points in meters.
        downsample_background_voxel_size (float = 0.05): Resolution (in meters) of the background occupancy octree,
            i.e. the minimum distinguishable background geometry size.

        semantic_stationarity_prior (bool = True): Whether to query LLM for stationarity prior of object classes.

        device (str = "cuda"): Device to run computations on.

        model_weight_dir (Path = "./\_\_model_cache\_\_"): Directory to cache model weights.

        filtering_interval_frames (int = 5): Apply filtering every N frames.
        filtering_min_observations (int = 2): Objects observed less than this number of times are removed during filtering.
        filtering_min_points (int = 5): Objects with less than this number of points are removed during filtering.

        merging_interval_frames (int = 5): Apply merging every N frames.
        merge_overlap_threshold (float = 0.7): Minimum geometric overlap ratio between two objects to consider merging them.
        merge_visual_threshold (float = 0.8): Minimum visual similarity between two objects to consider merging them.

        geometry_type (Literal["pointcloud", "tensor_pointcloud"] = "tensor_pointcloud"): Type of geometry representation to use. "pointcloud" is deprecated Open3D point clouds, "tensor_pointcloud" is Open3D's new GPU supported PointClouds.

        detection (DetectionConfig): Configuration parameters for object detection.
        debug (DebugConfig): Configuration parameters for debugging and visualization.
        matching (ObjectMatchingConfig): Configuration parameters for object matching and re-identification.
        background (BackgroundConfig): Configuration parameters for the background occupancy octree.

    """

    # Input parameters
    input_image_height: int = 640
    input_image_width: int = 360
    load_inital_map_path: str = ""

    pocd_visibility_threshold: float = 0.25
    preallocate_visible_objects_mask_count: int = 10
    downsample_voxel_size: float = 0.025

    semantic_stationarity_prior: bool = True

    device: str = "cuda"

    model_weight_dir: Path = field(default_factory=lambda: Path("./__model_cache__"))

    filtering_interval_frames: int = 5
    filtering_min_observations: int = 2
    filtering_min_points: int = 5

    merging_interval_frames: int = 5
    merge_overlap_threshold: float = 0.7
    merge_visual_threshold: float = 0.8
    occlusion_margin_m: float = 0.05

    geometry_type: Literal["pointcloud", "tensor_pointcloud"] = "tensor_pointcloud"

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    matching: ObjectMatchingConfig = field(default_factory=ObjectMatchingConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
