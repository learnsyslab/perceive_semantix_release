import threading
from logging import getLogger
from typing import Any, Generic, Optional, Type

from open3d import geometry

from perceive_semantix_lib.core.geometry import GeometryType, PointCloud, TensorPointCloud
from perceive_semantix_lib.core.occupancy_grid import OccupancyGrid

logger = getLogger(__name__)


class BackgroundTracker(Generic[GeometryType]):
    """Manages the background point cloud providing methods to asynchronously (in a seperate thread) update it."""

    geometry: GeometryType

    ground_projection: dict[str, Optional[OccupancyGrid]]
    ground_projection_dirty: bool

    def __init__(self, geometry_type: Type[GeometryType] = PointCloud) -> None:
        self.geometry = geometry_type()
        self._thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        self.ground_projection = {}
        self.ground_projection_dirty = True

    def get_ground_projection(
        self, map_id: str, resolution: float = 0.1, occupied_height_bounds: tuple[float, float] = (0.0, 10.0)
    ) -> Optional[OccupancyGrid]:
        if self.ground_projection_dirty or map_id not in self.ground_projection:
            self.ground_projection[map_id] = self.geometry.to_occupancy_grid(resolution, occupied_height_bounds)
            self.ground_projection_dirty = False
        return self.ground_projection[map_id]

    def async_add_background_points(
        self, new_points: GeometryType, downsample_voxel_size: float, current_xy: Optional[tuple[float, float]] = None
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.error("Background tracker thread has not finished executing but new call requested")
            raise RuntimeError("Background tracker  thread has not finished executing but new call requested")
        self._thread = threading.Thread(
            target=self._add_background_points,
            args=(new_points, downsample_voxel_size, current_xy),
            daemon=True,
        )
        self._thread.start()

    def wait_until_processed(self, timeout: Optional[float] = None) -> None:
        if self._thread is None:
            logger.warning("No thread running, but waiting requested")
            return
        if self._thread.is_alive():
            logger.debug("Waiting until BackgroundTracker thread is done")
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("Timout occured while waiting for BackgroundTracker thread to finish")

    def _add_background_points(
        self,
        new_points: GeometryType,
        downsample_voxel_size: float,
        addition_center_xy: Optional[tuple[float, float]] = None,
        addition_extent: float = 2.5,
    ) -> None:
        if addition_center_xy is not None:
            crop_bbox = geometry.AxisAlignedBoundingBox(
                min_bound=(addition_center_xy[0] - addition_extent, addition_center_xy[1] - addition_extent, -10.0),
                max_bound=(addition_center_xy[0] + addition_extent, addition_center_xy[1] + addition_extent, 10.0),
            )
            new_points.crop(crop_bbox)

        with self.lock:
            self.geometry = self.geometry.merge_with(new_points)
            self.geometry.postprocess_geometry(downsample_voxel_size)
            if isinstance(self.geometry, PointCloud):
                self.geometry.pcd, _ = self.geometry.pcd.remove_radius_outlier(nb_points=10, radius=0.5)
            if isinstance(self.geometry, TensorPointCloud):
                self.geometry.pcd, _ = self.geometry.pcd.remove_radius_outliers(nb_points=10, search_radius=0.5)
            self.ground_projection_dirty = True

    def to_dict(self) -> dict:
        return {"point_cloud": self.geometry.to_serializable()}

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], geometry_type: Type[GeometryType] = PointCloud
    ) -> "BackgroundTracker[GeometryType]":
        obj = BackgroundTracker(geometry_type)
        obj.geometry = geometry_type.from_serializable(data["point_cloud"])
        return obj
