from itertools import chain
from typing import Optional

import numpy as np
from jaxtyping import Int8
from scipy.ndimage import median_filter

from perceive_semantix_lib.core.occupancy_grid import OccupancyGrid
from perceive_semantix_lib.core.scene_belief import Scene
from perceive_semantix_lib.core.utils.optional_rerun_wrapper import log_occupancy_grid


class OccupancyMap:
    """Is attached to a Scene to compute a 2D occupancy grid for the entire scene (including background)."""

    def __init__(
        self, scene: Scene, resolution: float = 0.1, occupied_height_bounds: tuple[float, float] = (0.0, 10.0)
    ) -> None:
        self.scene = scene
        self.resolution = resolution
        self.occupied_height_bounds = occupied_height_bounds

    @staticmethod
    def _median_filter(grid: Int8[np.ndarray, "x_length y_length"], size: int = 3) -> np.ndarray:
        """Apply a median filter to the occupancy grid while preserving unknown values (-1)."""
        # Create a mask for known values
        known_mask = grid != -1
        temp = np.where(known_mask, grid, 0)
        filtered = median_filter(temp, size=size)
        filtered[~known_mask] = -1
        return filtered.astype(np.int8)

    def get_occupancy_grid(
        self,
        apply_median_filter: bool = True,
        included_position: Optional[tuple[float, float]] = None,
        included_position_padding: float = 0.5,
    ) -> Optional[OccupancyGrid]:
        """Compute a 2D occupancy grid for the entire scene (including background).

        It will update the internal occupancy grids of dirty objects and dirty background.

        Args:
            apply_median_filter (bool): Whether to filter out noise with a median filter.
            included_position (Optional[tuple[float, float]]): If provided, the return occupancy grid will definitily cover this position.
            included_position_padding (float): Minimum distance between the included position and the occupancy grid borders.

        Returns:
            occupancy_grid (Optional[OccupancyGrid]): An OccupancyGrid. None if the size of the occupancy grid cannot be inferred.

        """
        if included_position is None:
            included_position = (np.inf, np.inf)
        min_xy = np.array(included_position) - included_position_padding
        max_xy = np.array(included_position) + included_position_padding
        params = (self.resolution, self.occupied_height_bounds)

        for obj in chain(self.scene.get_objects(), [self.scene.background]):
            occ_grid = obj.get_ground_projection("occ", *params)
            if occ_grid is not None:
                min_xy = np.minimum(min_xy, occ_grid.origin)
                max_xy = np.maximum(
                    max_xy,
                    occ_grid.origin + np.array(occ_grid.grid.shape) * occ_grid.resolution,
                )

        if np.any(min_xy == np.array([np.inf, np.inf])) or np.any(max_xy == np.array([-np.inf, -np.inf])):
            return None

        origin = np.floor(min_xy / self.resolution) * self.resolution
        grid_size = np.ceil((max_xy - origin) / self.resolution).astype(int)
        grid = np.full(grid_size, -1, dtype=np.int8)

        for obj in chain(self.scene.get_objects(), [self.scene.background]):
            if (occ_grid := obj.get_ground_projection("occ", *params)) is None:
                continue
            ox, oy = ((occ_grid.origin - origin) / self.resolution).astype(int)
            h, w = occ_grid.grid.shape
            grid[ox : ox + h, oy : oy + w] = np.maximum(grid[ox : ox + h, oy : oy + w], occ_grid.grid)

        if apply_median_filter:
            grid = self._median_filter(grid, size=2)

        grid = OccupancyGrid(origin=origin, resolution=self.resolution, grid=grid)
        log_occupancy_grid(grid, "/world/occupancy_grid")
        return grid
