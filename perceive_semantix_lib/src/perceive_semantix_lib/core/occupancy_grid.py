from dataclasses import dataclass
from typing import Optional

import numpy as np
from jaxtyping import Float, Int8
from open3d import geometry


@dataclass
class OccupancyGrid:
    origin: Float[np.ndarray, "2"]
    resolution: float

    # 0 = free, 1 = occupied, -1 = unknown
    grid: Int8[np.ndarray, "x_length y_length"]

    def pad(self, padding_meters: float, padding_value: int = -1) -> None:
        """Pad the occupancy grid by the given amount of meters on all sides.

        The actual padding in cells is computed as ``floor(padding_meters / resolution)``.

        Args:
            padding_meters (float): The amount of padding in meters.
            padding_value (int): The value to use for the padded cells.

        """
        padding_cells = int(padding_meters / self.resolution)
        self.origin -= padding_cells * self.resolution
        self.grid = np.pad(
            self.grid,
            pad_width=padding_cells,
            mode="constant",
            constant_values=padding_value,
        )

    @staticmethod
    def from_point_cloud(
        point_cloud: geometry.PointCloud,
        resolution: float = 0.1,
        occupied_height_bounds: tuple[float, float] = (0.0, 10.0),
    ) -> Optional["OccupancyGrid"]:
        """Create an occupancy grid from a point cloud.

        Args:
            point_cloud (geometry.PointCloud): The input point cloud.
            resolution (float): The resolution of the occupancy grid in meters. The grid origin will be aligned to this resolution.
            occupied_height_bounds (tuple[float, float]): The height bounds (min, max) to consider points as occupied.

        Returns:
            Optional[OccupancyGrid]: An OccupancyGrid if there are any points in the point cloud, otherwise None.

        """
        points_np = np.asarray(point_cloud.points)
        if len(points_np) == 0:
            return None

        min_xy = points_np[:, :2].min(axis=0)
        max_xy = points_np[:, :2].max(axis=0)
        origin = np.floor(min_xy / resolution) * resolution
        grid_size = np.ceil((max_xy - origin) / resolution).astype(int) + 1

        grid = np.full(grid_size, -1, dtype=np.int8)

        occupied_mask = (points_np[:, 2] >= occupied_height_bounds[0]) & (points_np[:, 2] <= occupied_height_bounds[1])
        indices = ((points_np[:, :2] - origin) / resolution).astype(int)
        grid[indices[~occupied_mask, 0], indices[~occupied_mask, 1]] = 0
        grid[indices[occupied_mask, 0], indices[occupied_mask, 1]] = 1
        return OccupancyGrid(origin=origin, resolution=resolution, grid=grid)
