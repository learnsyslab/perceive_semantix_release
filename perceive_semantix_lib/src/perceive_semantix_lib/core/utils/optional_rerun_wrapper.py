import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Literal, Optional, Union
from uuid import UUID

import numpy as np
import torch
from jaxtyping import Bool, Float, UInt8
from open3d import geometry
from pyoctomap import ColorOcTree, OcTree  # pyright: ignore[reportAttributeAccessIssue]
from scipy.spatial.transform import Rotation
from supervision.detection.core import Detections
from typeguard import typeguard_ignore

from perceive_semantix_lib.core.geometry import GeometryBase, GeometryType, PointCloud, TensorPointCloud
from perceive_semantix_lib.core.matching.matching import ObjectDetectionAdjacency
from perceive_semantix_lib.core.occupancy_grid import OccupancyGrid, SparseVoxelGrid
from perceive_semantix_lib.core.scene_object import SceneObject
from perceive_semantix_lib.core.utils.object_classes import ObjectClasses

logger = logging.getLogger(__name__)


class _NoOpProxy:
    """Swallows arbitrary attribute access and calls, so chains like `orr.blueprint.Grid(...)` are no-ops when rerun is disabled or unavailable."""

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return None


class OptionalReRun:
    """A singleton class that optionally wraps the 'rerun' library.

    Example usage:
        orr = OptionalReRun()
        orr.set_use_rerun(config.enable_rerun)
        orr.init("my_app")  # This will call rerun.init if rerun is installed and enabled
        orr.log("my_data", data)  # This will call rerun.log if rerun is installed and enabled

        At any point in the code the same (=singelton) instance can be accessed via:
        orr_same = OptionalReRun()
        orr_same.log(...)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config_use_rerun = None
            cls._instance._rerun = None
        return cls._instance

    def set_use_rerun(self, config_use_rerun):
        self._config_use_rerun = config_use_rerun
        if self._config_use_rerun and self._rerun is None:
            try:
                import rerun as rr

                self._rerun = rr
                logger.info("Rerun is installed. Using rerun for logging.")
            except ImportError:
                logger.info("Rerun is not installed. Not using rerun for logging.")
        else:
            logger.info("Rerun functionality is disabled in the config. Not using rerun for logging.")

    def spawn(
        self,
        *,
        port: int = 9876,
        connect: bool = True,
        memory_limit: str = "75%",
        server_memory_limit: str = "1GiB",
        hide_welcome_screen: bool = False,
        detach_process: bool = True,
        executable_name: str = "rerun",
        executable_path: Optional[str] = None,
        default_blueprint=None,
        recording=None,
    ) -> None:
        """Spawn the rerun viewer, forwarding all arguments to rerun's own spawn/connect_grpc.

        Works around a bug in rerun-sdk==0.34.1: rr.spawn() sets RERUN_APP_ONLY=true on the
        viewer subprocess's environment.
        """
        if not self._config_use_rerun or self._rerun is None:
            logger.debug(
                "Skipping optional rerun call to 'spawn' because rerun usage is disabled or rerun is not installed."
            )
            return None

        import rerun_bindings

        rerun_bindings.spawn(
            port=port,
            memory_limit=memory_limit,
            server_memory_limit=server_memory_limit,
            hide_welcome_screen=hide_welcome_screen,
            detach_process=detach_process,
            executable_name=executable_name,
            executable_path=executable_path,
            extra_env=[],
        )

        if connect:
            self._rerun.connect_grpc(
                f"rerun+http://127.0.0.1:{port}/proxy",
                recording=recording,
                default_blueprint=default_blueprint,
            )

    def __getattr__(self, name):
        """Forward attributes and method calls to the rerun library if available and enabled.

        Non-callable attributes (e.g. the `blueprint` submodule) are returned as-is so that
        chained access like `orr.blueprint.Grid(...)` resolves directly against the real rerun
        module instead of being wrapped.
        """
        if self._config_use_rerun and self._rerun:
            attr = getattr(self._rerun, name, None)
            if attr is None:
                logger.debug(f"'{name}' is not a valid rerun attribute.")
                return _NoOpProxy()
            return attr
        else:
            if not self._config_use_rerun:
                logger.debug(f"Skipping optional rerun call to '{name}' because rerun usage is disabled.")
            elif self._rerun is None:
                logger.debug(f"Skipping optional rerun call to '{name}' because rerun is not installed.")
            return _NoOpProxy()


if TYPE_CHECKING:
    import rerun as orr
else:
    orr = OptionalReRun()
prev_logged_entities: set[str] = set()


@typeguard_ignore
def orr_set_blueprint(make_active: bool = True) -> Optional[orr.blueprint.Container]:
    """Set the rerun blueprint for the application."""
    if not orr._config_use_rerun:
        return None
    blueprint = orr.blueprint.Grid(
        orr.blueprint.Spatial3DView(
            name="Map",
            origin="/world",
            overrides={
                "world/camera/depth_image": orr.blueprint.EntityBehavior(visible=False),
                "world/objects/bbox": orr.blueprint.EntityBehavior(visible=False),
                "world/objects/pcd_rgb": orr.blueprint.EntityBehavior(visible=False),
                "world/occupancy_grid": orr.blueprint.EntityBehavior(visible=False),
            },
        ),
        orr.blueprint.Vertical(
            orr.blueprint.TimeSeriesView(
                name="Stationarity",
                origin="/stationarity",
                axis_y=orr.blueprint.ScalarAxis(range=(0.0, 1.0), zoom_lock=True),
                axis_x=orr.blueprint.TimeAxis(
                    view_range=orr.blueprint.TimeRange(
                        start=orr.blueprint.TimeRangeBoundary.cursor_relative(
                            seconds=-5,
                        ),
                        end=orr.blueprint.TimeRangeBoundary.cursor_relative(seconds=0),
                    ),
                    zoom_lock=True,
                ),
            ),
            orr.blueprint.Spatial2DView(
                origin="world/camera",
                overrides={"world/camera/depth_image": orr.blueprint.EntityBehavior(visible=False)},
            ),
        ),
        column_shares=[2, 1],
    )
    orr.send_blueprint(blueprint, make_active=make_active, make_default=True)
    return blueprint


def orr_log_camera(
    intrinsics: Float[np.ndarray, "3 3"],
    pose: Float[np.ndarray, "4 4"],
    image: Union[Path, UInt8[np.ndarray, "N M 3"]],
    img_width: int,
    img_height: int,
    time_sec: float,
    depth_image: Optional[Float[np.ndarray, "N M"]] = None,
    prev_pose: Optional[Float[np.ndarray, "4 4"]] = None,
) -> None:
    """Log camera rgb (and depth) image and pose to rerun, along with trajectory if previous pose is provided.

    Args:
        intrinsics (Float[np.ndarray, "3 3"]): Camera intrinsic matrix.
        pose (Float[np.ndarray, "4 4"]): Camera pose matrix.
        image (Union[Path, UInt8[np.ndarray, "N M 3"]]): RGB image or path to image file.
        img_width (int): Image width.
        img_height (int): Image height.
        time_sec (float): Timestamp in seconds.
        depth_image (Optional[UInt8[np.ndarray, "N M 3"]], optional): Depth image. Defaults to None.
        prev_pose (Optional[Float[np.ndarray, "4 4"]], optional): Previous camera pose for trajectory logging. Defaults to None.

    """
    if not orr._config_use_rerun:
        return
    # Extract intrinsic camera parameters
    focal_length = [intrinsics[0, 0].item(), intrinsics[1, 1].item()]
    principal_point = [intrinsics[0, 2].item(), intrinsics[1, 2].item()]
    resolution = [img_width, img_height]  # Width x Height from the RGB image

    # Log camera intrinsics and resolution
    orr.log(
        "world/camera",
        orr.Pinhole(
            resolution=resolution,
            focal_length=focal_length,
            principal_point=principal_point,
            image_plane_distance=0.3,
        ),
    )

    # Convert the current adjusted pose to translation and quaternion for logging
    translation = pose[:3, 3].tolist()

    orr.log("world/camera", orr.Transform3D(translation=translation, mat3x3=pose[:3, :3], from_parent=False))

    # Log RGB image
    orr_log_rgb_image(image)

    # Log depth image
    if depth_image is not None:
        orr_log_depth_image(depth_image)

    # Log trajectory if not the first frame
    if prev_pose is not None:
        prev_translation = prev_pose[:3, 3].tolist()

        # Log a line strip from the previous to the current camera pose
        orr.log(
            f"world/camera_trajectory/{time_sec:.3f}",
            orr.LineStrips3D([np.vstack([prev_translation, translation]).tolist()], colors=[[255, 0, 0]]),
        )


def orr_log_rgb_image(image: Union[Path, UInt8[np.ndarray, "N M 3"]]):
    """Log camera RGB image to rerun."""
    if isinstance(image, Path):
        orr.log("world/camera/rgb_image", orr.EncodedImage(path=str(image.resolve())))
    else:
        orr.log("world/camera/rgb_image", orr.Image(image, color_model="RGB"))


def orr_log_expected_object_projection(image: Bool[torch.Tensor, "N M"]):
    """Log expected object projection image to rerun."""
    orr.log("world/camera/object_projection", orr.Image(image.cpu() * 255))


def orr_log_depth_image(image: Float[np.ndarray, "N M"], meter: float = 1.0):
    """Log camera depth image to rerun."""
    orr.log("world/camera/depth_image", orr.DepthImage(image, meter=meter))


def orr_log_annotated_image(image: Union[Path, UInt8[np.ndarray, "N M 3"]]):
    """Log annotated RGB image to rerun."""
    if isinstance(image, Path):
        orr.log("world/camera/rgb_image_annotated", orr.EncodedImage(path=str(image.resolve())))
    else:
        orr.log("world/camera/rgb_image_annotated", orr.Image(image, color_model="RGB"))


def orr_log_point_clouds_bounding_boxes(
    detections: Detections, time_sec: float, object_classes: Optional[ObjectClasses] = None
) -> None:
    """Log point clouds and bounding boxes from detections to rerun. Requires detections to have 'point_clouds', 'bounding_boxes', and 'classes' in its 'data' dictonary.

    If object_classes is provided, an additional log entry will be made where the point clouds are be colored according to the class colors.
    """
    if not orr._config_use_rerun:
        return

    logger.debug("Logging point clouds and bounding boxes to rerun.")

    # Remove previously logged entities that are no longer present
    base_entity_path = "world/detections/objects"
    orr.log(base_entity_path, orr.Clear(recursive=True))

    if detections.data is None:
        return
    if (
        "point_clouds" not in detections.data
        or "bounding_boxes" not in detections.data
        or "classes" not in detections.data
    ):
        logger.warning("Detections data does not contain required fields for logging point clouds and bounding boxes.")
        return

    new_logged_entities = set()
    for i, (pcd, bbox, unmodified_class_name) in enumerate(
        zip(detections.data["point_clouds"], detections.data["bounding_boxes"], detections.data["classes"])
    ):
        obj_label = f"det_{i}"

        if isinstance(pcd, PointCloud):
            positions = np.asarray(pcd.pcd.points)
            if hasattr(pcd.pcd, "colors") and len(pcd.pcd.colors) > 0:
                colors = np.asarray(pcd.pcd.colors) * 255
                colors = colors.astype(np.uint8)
            else:
                colors = None
        elif isinstance(pcd, TensorPointCloud):
            if len(pcd) == 0:
                positions = np.zeros((0, 3))
                colors = None
            else:
                positions = pcd.pcd.point.positions.cpu().numpy()
                if "colors" in pcd.pcd.point:
                    colors = (pcd.pcd.point.colors.cpu().numpy() * 255).astype(np.uint8)
                else:
                    colors = None
        else:
            raise ValueError("Only point cloud scene objects are supported for rerun logging.")

        rgb_pcd_entity = base_entity_path + "/rgb_pcd/" + obj_label
        orr.log(
            rgb_pcd_entity,
            orr.Points3D(
                positions,
                colors=colors,
                radii=0.005,
            ),
        )
        new_logged_entities.add(rgb_pcd_entity)

        if object_classes is not None:
            curr_obj_color = object_classes.get_class_color(unmodified_class_name)
            seg_pcd_entity = base_entity_path + "/seg_pcd/" + obj_label
            orr.log(
                seg_pcd_entity,
                orr.Points3D(
                    positions,
                    colors=[curr_obj_color],
                    radii=0.005,
                ),
            )
            new_logged_entities.add(seg_pcd_entity)


def orr_log_background_point_cloud(pcd: GeometryBase, log_as_detection: bool = False) -> None:
    """Log background point clouds to rerun."""
    logger.debug("Logging point clouds and bounding boxes to rerun.")
    if not orr._config_use_rerun:
        return
    if log_as_detection:
        base_entity_path = "world/detections/background"
    else:
        base_entity_path = "world/background"

    if isinstance(pcd, PointCloud):
        positions = np.asarray(pcd.pcd.points)
        if hasattr(pcd.pcd, "colors") and len(pcd.pcd.colors) > 0:
            colors = np.asarray(pcd.pcd.colors) * 255
            colors = colors.astype(np.uint8)
        else:
            colors = None
    elif isinstance(pcd, TensorPointCloud):
        if len(pcd) == 0:
            positions = np.zeros((0, 3))
            colors = None
        else:
            positions = pcd.pcd.point.positions.cpu().numpy()
            if "colors" in pcd.pcd.point:
                colors = (pcd.pcd.point.colors.cpu().numpy() * 255).astype(np.uint8)
            else:
                colors = None
    else:
        raise ValueError("Only point cloud scene objects are supported for rerun logging.")

    orr.log(
        base_entity_path,
        orr.Points3D(
            positions,
            colors=colors,
            radii=0.005,
        ),
    )


_BACKGROUND_OCTREE_OCCUPIED_DEFAULT_COLOR = np.array([160, 160, 160], dtype=np.uint8)
_BACKGROUND_OCTREE_EMPTY_COLOR = np.array([80, 140, 200], dtype=np.uint8)


def _extract_leaf_points(
    tree: Union[OcTree, ColorOcTree],
    bbx_min: Float[np.ndarray, "3"],
    bbx_max: Float[np.ndarray, "3"],
    store_color: bool,
    log_empty: bool,
) -> tuple[
    tuple[Optional[Float[np.ndarray, "M 3"]], Optional[UInt8[np.ndarray, "M 3"]], Optional[Float[np.ndarray, "M"]]],
    Optional[tuple[Float[np.ndarray, "K 3"], Float[np.ndarray, "K"]]],
]:
    """Collect one point per leaf within a bounding box, at the leaf's own center and radius.

    `tree.begin_leafs_bbx(bbx_min, bbx_max)` is a real bounded tree descent (the primitive
    `extractPointCloud()` is built on internally), so this only visits leaves near the box rather than the
    whole tree. Unlike a voxel grid, `Points3D` takes a radius per point, so a pruned/merged leaf (occupied or
    free) is logged as a single larger point at its true size instead of being expanded into sub-voxels.

    Returns ((occupied_points, occupied_colors, occupied_radii), empty) where each of the first tuple's entries
    is None if no occupied leaves were found. `empty` is None if `log_empty` is False, otherwise
    `(empty_points, empty_radii)` or None if no free leaves were found.
    """
    occ_points, occ_colors, occ_radii = [], [], []
    empty_points, empty_radii = [], []

    for leaf in tree.begin_leafs_bbx(bbx_min, bbx_max):
        coord = np.array(leaf.getCoordinate())
        radius = leaf.getSize() / 2.0

        if tree.isNodeOccupied(leaf):
            occ_points.append(coord)
            occ_radii.append(radius)
            color = (
                np.array(leaf.getColor(), dtype=np.uint8) if store_color else _BACKGROUND_OCTREE_OCCUPIED_DEFAULT_COLOR
            )
            occ_colors.append(color)
        elif log_empty:
            empty_points.append(coord)
            empty_radii.append(radius)

    occupied = (np.stack(occ_points), np.stack(occ_colors), np.array(occ_radii)) if occ_points else (None, None, None)
    empty = None
    if log_empty:
        empty = (np.stack(empty_points), np.array(empty_radii)) if empty_points else None
    return occupied, empty


def orr_log_background_octree(
    tree: Union[OcTree, ColorOcTree],
    dirty_bbox: Optional[tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3"]]],
    chunk_size_m: float = 2.0,
    entity_base: str = "world/background_octree",
    store_color: bool = False,
    log_empty: bool = False,
) -> None:
    """Log the background occupancy octree to rerun as a set of chunked point clouds.

    Rerun archetypes fully replace an entity's content on every `orr.log()` call, so the octree is split into
    fixed-size XY chunks (each spanning the tree's full Z extent), each logged under its own entity path. Only
    chunks overlapping `dirty_bbox` (the AABB of points and sensor origin touched by the most recent update) are
    re-logged; untouched chunks keep showing their last-logged state without being resent. A touched chunk is
    still re-sent in full (not just its changed voxels), since anything omitted from a log call is dropped from
    the display -- `dirty_bbox` only controls which chunks get re-queried, not how much of a touched chunk is
    included.

    Args:
        tree (Union[OcTree, ColorOcTree]): The background occupancy octree.
        dirty_bbox (Optional[tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3"]]]): The (min, max) world-frame
            AABB touched by the most recent update, or None if nothing changed since the last call.
        chunk_size_m (float): Size (in meters, XY only) of the chunks used to split the octree for logging.
        entity_base (str): Base rerun entity path under which chunk entities (`<entity_base>/chunk_<cx>_<cy>`)
            are logged.
        store_color (bool): Whether `tree` is a `ColorOcTree` carrying per-voxel color. When True, each voxel is
            logged with its stored color; otherwise every voxel is logged with a single flat gray, since no
            per-voxel color is available.
        log_empty (bool): Whether to also log free (non-occupied) leaves, under a `<chunk_entity>/empty` child
            entity, in a single flat color distinct from occupied voxels. Free leaves are frequently pruned into
            much larger blocks than occupied ones (entire empty rooms can collapse into one leaf), so this can
            log a handful of very large points rather than a dense fill of the free volume. Off by default.

    """
    if not orr._config_use_rerun:
        return
    if dirty_bbox is None:
        return

    min_xyz, max_xyz = dirty_bbox
    tree_z_range = (tree.getMetricMin()[2], tree.getMetricMax()[2])

    cx0, cx1 = int(np.floor(min_xyz[0] / chunk_size_m)), int(np.floor(max_xyz[0] / chunk_size_m))
    cy0, cy1 = int(np.floor(min_xyz[1] / chunk_size_m)), int(np.floor(max_xyz[1] / chunk_size_m))

    for cx in range(cx0, cx1 + 1):
        for cy in range(cy0, cy1 + 1):
            chunk_min = np.array([cx * chunk_size_m, cy * chunk_size_m, tree_z_range[0]])
            chunk_max = np.array([(cx + 1) * chunk_size_m, (cy + 1) * chunk_size_m, tree_z_range[1]])
            entity_path = f"{entity_base}/chunk_{cx}_{cy}"

            (occ_points, occ_colors, occ_radii), empty = _extract_leaf_points(
                tree, chunk_min, chunk_max, store_color, log_empty
            )
            if occ_points is None:
                orr.log(entity_path, orr.Clear(recursive=True))
            else:
                orr.log(entity_path, orr.Points3D(occ_points, colors=occ_colors, radii=occ_radii))

            if log_empty:
                empty_entity_path = f"{entity_path}/empty"
                if empty is None:
                    orr.log(empty_entity_path, orr.Clear(recursive=True))
                else:
                    empty_points, empty_radii = empty
                    orr.log(
                        empty_entity_path,
                        orr.Points3D(
                            empty_points,
                            colors=np.tile(_BACKGROUND_OCTREE_EMPTY_COLOR, (empty_points.shape[0], 1)),
                            radii=empty_radii,
                        ),
                    )


def orr_log_scene_objects(objects: Iterator[SceneObject[GeometryType]], object_classes: ObjectClasses) -> None:
    """Log scene objects to rerun. Each object will have its point cloud and bounding box logged. The bounding box will be colored according to the object's class color determined from 'object_classes'.

    Args:
        objects (Iterator[SceneObject]): An iterator of SceneObject instances to log.
        object_classes (ObjectClasses): An ObjectClasses instance to determine class colors.

    """
    if not orr._config_use_rerun:
        return
    global prev_logged_entities

    logger.debug("Logging scene objects to rerun.")
    base_entity_path = "world/objects"
    new_logged_entities = set()
    for object in objects:
        obj_class = object.id_str.split("_")[1]
        obj_label = object.id_str
        obj_color = object_classes.get_class_color(object.class_name)
        obj_instance_color = (np.array(object.instance_color) * 255).astype(np.uint8).tolist()

        if isinstance(object.geometry, PointCloud):
            positions = np.asarray(object.geometry.pcd.points)
            if hasattr(object.geometry.pcd, "colors") and len(object.geometry.pcd.colors) > 0:
                colors = np.asarray(object.geometry.pcd.colors) * 255
                colors = colors.astype(np.uint8)
            else:
                colors = None
        elif isinstance(object.geometry, TensorPointCloud):
            if len(object.geometry) == 0:
                positions = np.zeros((0, 3))
                colors = None
            else:
                positions = object.geometry.pcd.point.positions.cpu().numpy()
                if "colors" in object.geometry.pcd.point:
                    colors = (object.geometry.pcd.point.colors.cpu().numpy() * 255).astype(np.uint8)
                else:
                    colors = None
        else:
            raise ValueError("Only point cloud scene objects are supported for rerun logging.")

        pcd_entity = base_entity_path + "/pcd_rgb/" + obj_class + "/" + obj_label
        orr.log(
            pcd_entity,
            orr.Points3D(
                positions,
                colors=colors,
                radii=0.005,
            ),
        )
        new_logged_entities.add(pcd_entity)

        pcd_entity = base_entity_path + "/pcd_instance/" + obj_class + "/" + obj_label
        orr.log(
            pcd_entity,
            orr.Points3D(
                positions,
                colors=[obj_instance_color],
                radii=0.005,
                labels=[object.class_name],
            ),
        )
        new_logged_entities.add(pcd_entity)

        if isinstance(object.bounding_box, geometry.AxisAlignedBoundingBox):
            centers = [object.bounding_box.get_center()]
            half_sizes = [object.bounding_box.get_extent() / 2]
            rotation = [Rotation.identity().as_quat()]
        else:
            centers = [object.bounding_box.center]
            half_sizes = [object.bounding_box.extent / 2]
            rotation = [Rotation.from_matrix(object.bounding_box.R).as_quat()]
        bbox_entity = base_entity_path + "/bbox/" + obj_class + "/" + obj_label
        orr.log(
            bbox_entity,
            orr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                rotations=rotation,
                colors=[obj_color],
                labels=[f"{str(object.id)[:4]}_{object.class_name}"],
            ),
        )
        new_logged_entities.add(bbox_entity)

    for entity_path in prev_logged_entities.difference(new_logged_entities):
        orr.log(entity_path, orr.Clear(recursive=True))
    prev_logged_entities = new_logged_entities


def orr_log_detections_matches(
    detections: Detections,
    time_sec: float,
    objects: dict[UUID, SceneObject[GeometryType]],
    adjacency: ObjectDetectionAdjacency,
) -> None:
    """Log arrows from detected objects to matched scene objects."""
    if not orr._config_use_rerun:
        return
    timestamp = f"{time_sec:.3f}".replace(".", "-")

    logger.debug("Logging detections matches to rerun.")

    base_entity_path = "world/detections/objects"

    if detections.data is None:
        return
    if "point_clouds" not in detections.data or "classes" not in detections.data:
        logger.warning("Detections data does not contain required field point_clouds or classes.")
        return

    for i, (pcd, unmodified_class_name) in enumerate(zip(detections.data["point_clouds"], detections.data["classes"])):
        id, _, _ = adjacency.get_merge_target(i)
        if id is None or id not in objects:
            continue

        class_name = unmodified_class_name.replace(" ", "_")
        obj_label = f"{timestamp}_{i}_{class_name}"

        detection_centroid = pcd.get_center()
        match_centroid = np.array(objects[id].geometry.get_center())

        entity = base_entity_path + "/matches/" + obj_label
        orr.log(
            entity,
            orr.Arrows3D(
                vectors=[match_centroid - detection_centroid],
                origins=[detection_centroid],
            ),
        )


logged_series = set()


def orr_log_stationarity(objects: Iterable[SceneObject[GeometryType]]):
    """Log stationarity confidence of scene objects to rerun.

    Args:
        objects (Iterable[SceneObject[GeometryType]]): An iterable of SceneObject instances whose stationarity confidence to log.

    """
    if not orr._config_use_rerun:
        return
    global logged_series
    base_path = "stationarity/"
    for obj in objects:
        obj_class = obj.id_str.split("_")[1]
        entity = base_path + obj_class + "/" + obj.id_str
        color = (np.array(obj.instance_color) * 255).astype(np.uint8).tolist()

        if obj.id_str not in logged_series:
            orr.log(entity, orr.SeriesLines(colors=color, names=obj.id_str), static=True)
            logged_series.add(obj.id_str)

        orr.log(
            entity,
            orr.Scalars([obj.stationarity_confidence]),
        )


node_ids: set[Union[str, UUID]] = set()
node_id_strings: dict[UUID, str] = {}
edge_ids: set[tuple[Union[str, UUID], Union[str, UUID]]] = set()
start_time: Optional[float] = None


def orr_log_associations(
    objects: dict[UUID, SceneObject[GeometryType]],
    active_object_ids: list[UUID],
    detections: Detections,
    object_adjacency: ObjectDetectionAdjacency,
    time_sec: float,
    inview_object_ids: list[UUID] = [],
    log_detections: bool = False,
):
    """Log object-detection and object-objects associations as a graph to rerun.

    Args:
        objects (dict[UUID, SceneObject[GeometryType]]): A dictionary of scene objects keyed by their UUIDs.
        active_object_ids (list[UUID]): A list of UUIDs of active scene objects.
        inview_object_ids (list[UUID]): A list of UUIDs of scene objects in view.
        detections (Detections): The current detections.
        object_adjacency (ObjectDetectionAdjacency): The object-detection adjacency information, mapping which detections where associated with which objects, and which objects were associated with which other objects.
        time_sec (float): The current timestamp in seconds.
        log_detections (bool, optional): Whether to include detections as nodes in the graph. Defaults to False.

    """
    if not orr._config_use_rerun:
        return

    global start_time
    if start_time is None:
        start_time = time_sec
    time_sec -= start_time
    timestamp = f"{time_sec:.3f}".replace(".", "-")

    new_node_ids, new_edge_ids = object_adjacency.get_nodes_edges(detection_prefix=f"{timestamp}_det_")

    if not log_detections:
        new_node_ids = [id for id in new_node_ids if isinstance(id, UUID)]
        new_edge_ids = [e for e in new_edge_ids if isinstance(e[0], UUID) and isinstance(e[1], UUID)]

    global node_ids, edge_ids
    node_ids = node_ids.union(new_node_ids)
    edge_ids = edge_ids.union(new_edge_ids)

    def formatter(id: Union[str, UUID]) -> str:
        if isinstance(id, str):
            return id
        elif id in node_id_strings:
            return node_id_strings[id]
        else:
            node_id_strings[id] = objects[id].id_str
            return objects[id].id_str

    nodes: list[str] = [formatter(id) for id in node_ids]
    edges: list[tuple[str, str]] = [(formatter(e[0]), formatter(e[1])) for e in edge_ids]

    detection_radi = 0.4
    object_radi = 0.5
    radii = [detection_radi if isinstance(id, int) else object_radi for id in node_ids]

    active_color = [0, 255, 0]
    inview_color = [255, 165, 0]
    inactive_color = [100, 0, 0]
    deleted_color = [80, 80, 80]
    detection_color = [0, 100, 100]
    colors = []
    for id in node_ids:
        if isinstance(id, UUID):
            # in-view objects are yellow, active objects are green, inactive objects are dark red, deleted objects are gray
            # in view objects will override active objects
            if id in inview_object_ids:
                colors.append(inview_color)
            elif id in active_object_ids:
                colors.append(active_color)
            elif id in objects.keys():
                colors.append(inactive_color)
            else:
                colors.append(deleted_color)
        else:
            colors.append(detection_color)

    base_path = "associations"
    orr.log(
        base_path,
        orr.GraphNodes(node_ids=nodes, colors=colors, radii=radii),
        orr.GraphEdges(
            edges=edges,
            graph_type="directed",
        ),
    )


def log_occupancy_grid(
    occupancy_grid: OccupancyGrid, entity_path: str, colormap: Literal["occupancy", "raw"] = "occupancy"
) -> None:
    """Log occupancy grid to rerun.

    Args:
        occupancy_grid (OccupancyGrid): The occupancy grid to log.
        entity_path (str): The entity path in rerun.
        colormap (Literal["occupancy", "raw"]): The colormap to use for visualization. "occupancy" maps occupied, free, and unknown cells to specific colors. "raw" uses the raw grid values from -128 (black) to 127 (white). Defaults to "occupancy".

    """
    if not orr._config_use_rerun:
        return

    if occupancy_grid is None:
        return

    image = occupancy_grid.grid.astype(np.float32)
    if colormap == "occupancy":
        occupied = image > 0
        free = image == 0
        unknown = image < 0
        image[occupied] = 0.0
        image[free] = 255.0
        image[unknown] = 127.0
    elif colormap == "raw":
        image = image + 128.0

    # grid is [x, y]; GridMap wants row-major [height(y), width(x)] data with row 0 at the top and rows
    # extending downward (-y) from `translation`, so flip vertically to keep the lower-left corner anchored.
    pixels = np.flipud(image.T).clip(0, 255).astype(np.uint8)
    height, width = pixels.shape

    orr.log(
        entity_path,
        orr.GridMap(
            data=pixels.tobytes(),
            format=orr.components.ImageFormat(width=width, height=height, color_model="L", channel_datatype="U8"),
            cell_size=occupancy_grid.resolution,
            translation=[float(occupancy_grid.origin[0]), float(occupancy_grid.origin[1]), -0.01],
            colormap=orr.components.Colormap.Grayscale,
        ),
    )


def log_sparse_voxel_grid(voxel_grid: SparseVoxelGrid, entity_path: str) -> None:
    """Log a sparse 3D voxel priority grid to rerun as a rainbow-colored point cloud.

    Point color follows rerun's own Turbo colormap over the [0, 1] priority probability (blue = low, red = high), and alpha is set to that same probability so low-priority voxels fade out.

    Args:
        voxel_grid (SparseVoxelGrid): The voxel grid to log.
        entity_path (str): The entity path in rerun.

    """
    if not orr._config_use_rerun:
        return

    if voxel_grid is None:
        return

    if len(voxel_grid.coords) == 0:
        orr.log(entity_path, orr.Clear(recursive=True))
        return

    if orr._rerun is None:
        return

    from rerun.utilities._turbo import turbo_colormap_data

    positions = voxel_grid.origin + (voxel_grid.coords + 0.5) * voxel_grid.resolution
    probabilities = (voxel_grid.values.astype(np.float32) + 128.0) / 255.0

    lut_indices = (probabilities * (len(turbo_colormap_data) - 1)).astype(int)
    colors = (turbo_colormap_data[lut_indices] * 255).astype(np.uint8)

    max_radius = voxel_grid.resolution * 0.5
    min_radius = max_radius * 0.01
    radii = np.clip(probabilities * max_radius, min_radius, max_radius)

    orr.log(
        entity_path,
        orr.Points3D(positions, colors=colors, radii=radii),
    )
