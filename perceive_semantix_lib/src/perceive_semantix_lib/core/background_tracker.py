import tempfile
import threading
from logging import getLogger
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from jaxtyping import Float
from pyoctomap import ColorOcTree, OcTree  # pyright: ignore[reportAttributeAccessIssue]

from perceive_semantix_lib.core.config import BackgroundConfig
from perceive_semantix_lib.core.occupancy_grid import OccupancyGrid

logger = getLogger(__name__)


class BackgroundTracker:
    """Manages the background occupancy octree, providing methods to asynchronously update it."""

    tree: Union[OcTree, ColorOcTree]
    resolution: float
    store_color: bool
    lazy_eval: bool

    ground_projection: dict[str, Optional[OccupancyGrid]]
    ground_projection_dirty: bool

    last_update_bbox: Optional[tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3"]]]

    def __init__(self, config: BackgroundConfig) -> None:
        self.store_color = config.store_color
        self.lazy_eval = config.lazy_eval
        self.tree = ColorOcTree(config.resolution_m) if config.store_color else OcTree(config.resolution_m)
        if config.prob_hit is not None:
            self.tree.setProbHit(config.prob_hit)
        if config.prob_miss is not None:
            self.tree.setProbMiss(config.prob_miss)
        self.resolution = config.resolution_m
        self._thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        self.ground_projection = {}
        self.ground_projection_dirty = True
        self.last_update_bbox = None

    def get_ground_projection(
        self, map_id: str, resolution: float = 0.1, occupied_height_bounds: tuple[float, float] = (0.0, 10.0)
    ) -> Optional[OccupancyGrid]:
        if self.ground_projection_dirty or map_id not in self.ground_projection:
            with self.lock:
                self.ground_projection[map_id] = self._compute_occupancy_grid(resolution, occupied_height_bounds)
            self.ground_projection_dirty = False
        return self.ground_projection[map_id]

    def _compute_occupancy_grid(
        self, resolution: float, occupied_height_bounds: tuple[float, float]
    ) -> Optional[OccupancyGrid]:
        if self.tree.getNumLeafNodes() == 0:
            return None

        min_xy = self.tree.getMetricMin()[:2]
        max_xy = self.tree.getMetricMax()[:2]
        origin = np.floor(min_xy / resolution) * resolution
        grid_size = np.ceil((max_xy - origin) / resolution).astype(int) + 1
        grid = np.full(grid_size, -1, dtype=np.int8)

        # Mirror the old point-cloud based ground projection: only cells that were ever hit by an
        # occupied observation are classified (as occupied or, if outside the height band, free); cells
        # that were only ever seen as free (ray-carved) space stay unknown, since a raw point cloud never
        # encoded free space at all.
        for leaf in self.tree.begin_leafs():
            if not self.tree.isNodeOccupied(leaf):
                continue
            coord = leaf.getCoordinate()
            size = leaf.getSize()
            half = size / 2.0
            value = 1 if occupied_height_bounds[0] <= coord[2] <= occupied_height_bounds[1] else 0

            ix0 = max(int(np.floor((coord[0] - half - origin[0]) / resolution)), 0)
            ix1 = min(int(np.ceil((coord[0] + half - origin[0]) / resolution)), grid.shape[0])
            iy0 = max(int(np.floor((coord[1] - half - origin[1]) / resolution)), 0)
            iy1 = min(int(np.ceil((coord[1] + half - origin[1]) / resolution)), grid.shape[1])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            grid[ix0:ix1, iy0:iy1] = np.maximum(grid[ix0:ix1, iy0:iy1], value)

        return OccupancyGrid(origin=origin, resolution=resolution, grid=grid)

    def async_add_background_points(
        self,
        points_world: Float[np.ndarray, "N 3"],
        sensor_origin_world: Float[np.ndarray, "3"],
        max_range: float = -1.0,
        colors_world: Optional[Float[np.ndarray, "N 3"]] = None,
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.error("Background tracker thread has not finished executing but new call requested")
            raise RuntimeError("Background tracker  thread has not finished executing but new call requested")
        self._thread = threading.Thread(
            target=self._add_background_points,
            args=(points_world, sensor_origin_world, max_range, colors_world),
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
        points_world: Float[np.ndarray, "N 3"],
        sensor_origin_world: Float[np.ndarray, "3"],
        max_range: float,
        colors_world: Optional[Float[np.ndarray, "N 3"]] = None,
    ) -> None:
        if points_world.shape[0] == 0:
            return
        points_world = np.ascontiguousarray(points_world, dtype=np.float64)
        sensor_origin_world = np.ascontiguousarray(sensor_origin_world, dtype=np.float64)

        with self.lock:
            if self.store_color and colors_world is not None:
                self.tree.insertPointCloud(
                    points_world,
                    sensor_origin_world,
                    max_range=max_range,
                    lazy_eval=self.lazy_eval,
                    discretize=False,
                    colors=np.ascontiguousarray(colors_world, dtype=np.float64),
                )
            else:
                self.tree.insertPointCloud(
                    points_world, sensor_origin_world, max_range=max_range, lazy_eval=self.lazy_eval, discretize=False
                )
            self.tree.updateInnerOccupancy()
            self.ground_projection_dirty = True
            self.last_update_bbox = (
                np.minimum(points_world.min(axis=0), sensor_origin_world),
                np.maximum(points_world.max(axis=0), sensor_origin_world),
            )

    def to_dict(self) -> dict:
        # Written to a real path rather than in-memory: ColorOcTree.writeBinary requires a filename since it
        # appends a color trailer to the file after the core library writes the occupancy data.
        with tempfile.NamedTemporaryFile(suffix=".bt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.tree.writeBinary(tmp_path)
            octomap_binary = Path(tmp_path).read_bytes()
        finally:
            Path(tmp_path).unlink()
        return {
            "octomap_binary": octomap_binary,
            "resolution": self.resolution,
            "store_color": self.store_color,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackgroundTracker":
        config = BackgroundConfig(resolution_m=data["resolution"], store_color=data.get("store_color", False))
        obj = cls(config)
        with tempfile.NamedTemporaryFile(suffix=".bt", delete=False) as tmp:
            tmp.write(data["octomap_binary"])
            tmp_path = tmp.name
        try:
            obj.tree.readBinary(tmp_path)
        finally:
            Path(tmp_path).unlink()
        obj.ground_projection_dirty = True
        return obj
