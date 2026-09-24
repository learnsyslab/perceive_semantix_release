from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import Iterable, Literal, Optional, Union

import numpy as np
from jaxtyping import Bool, Float
from scipy.ndimage import gaussian_filter
from scipy.stats import beta

from perceive_semantix_lib.core.geometry import GeometryType
from perceive_semantix_lib.core.occupancy_grid import OccupancyGrid, SparseVoxelGrid
from perceive_semantix_lib.core.scene_object import SceneObject
from perceive_semantix_lib.core.utils.optional_rerun_wrapper import log_occupancy_grid, log_sparse_voxel_grid
from perceive_semantix_lib.llm.similarity_client import SimilarityOpenAIAsyncClient

logger = getLogger(__name__)


class ExplorationPriority(Enum):
    STATIONARITY = 0
    SEMANTIC_SIMILARITY = 1


class SimilarityMeasure(Enum):
    SAME_LABEL = 0
    SEMANTIC_LLM = 1


@dataclass
class ExplorationConfig:
    similarity_measure: SimilarityMeasure = SimilarityMeasure.SEMANTIC_LLM
    device: str = "cuda"
    minimum_relevancy: float = 0.6
    map_padding_meters: float = 0.2
    resolution_meters: float = 0.1

    measurement_std: float = 0.05
    search_radius: float = 0.2
    search_radius_stationarity: float = 0.6
    dimensions: Literal[2, 3] = 2


class ExplorationPriorityMap:
    """Implements the exploration priority map as described in https://arxiv.org/abs/2509.19851 .

    For the operation mode `ExplorationPriority.STATIONARITY`, the exploration priority is based on the stationarity confidence of objects.
    For the operation mode `ExplorationPriority.SEMANTIC_SIMILARITY`, the exploration priority is based on the semantic similarity of objects to the query text.
    """

    def __init__(
        self,
        object_list: list[str],
        config: ExplorationConfig = ExplorationConfig(),
    ) -> None:
        self.config = config
        if self.config.dimensions not in (2, 3):
            raise ValueError(f"Invalid dimensions {self.config.dimensions}, must be 2 or 3")
        self.projection_params = {
            "resolution": self.config.resolution_meters,
            "occupied_height_bounds": (float("-inf"), float("inf")),
            "padding_meters": 0.4,
        }

        self.object_list: list[str] = object_list
        self.query: Optional[str] = None
        self.exploration_priority: Optional[ExplorationPriority] = None
        self.obj_classes_relevancy: dict[str, float] = {}

    def set_query(self, query: str) -> None:
        """Set the query string for the exploration priority map.

        Args:
            query (str): Query string to set the exploration priority map.
                If the query is "maintain", the exploration priority is set to `ExplorationPriority.STATIONARITY`.
                Otherwise, it is set to `ExplorationPriority.SEMANTIC_SIMILARITY`.

        """
        if len(query) == 0:
            logger.warning("Query is empty, cannot set exploration priority")
            return

        self.query = query
        if self.query == "maintain":
            self.exploration_priority = ExplorationPriority.STATIONARITY
        else:
            self.exploration_priority = ExplorationPriority.SEMANTIC_SIMILARITY
        logger.info(f"Set exploration priority to {self.exploration_priority} with query '{self.query}'")

        if self.exploration_priority == ExplorationPriority.SEMANTIC_SIMILARITY:
            logger.info("Querying LLM for object relevancy")
            self._get_object_relevancy_semantic(self.query)

    def _get_object_relevancy_semantic(self, text_query: str) -> None:
        self.obj_classes_relevancy = SimilarityOpenAIAsyncClient.query_semantic_similarity(text_query, self.object_list)
        if "person" in self.obj_classes_relevancy:
            self.obj_classes_relevancy["person"] = 0.0

    @staticmethod
    def _get_object_relevancy_stationarity(
        objects: Iterable[SceneObject[GeometryType]], alpha_param: float = 4, beta_param: float = 2.8
    ) -> Float[np.ndarray, "N"]:
        confidences = np.array([obj.stationarity_confidence for obj in objects])
        relevancy_scores = beta.pdf(confidences, alpha_param, beta_param)
        return relevancy_scores

    def _build_2d_priority_map(
        self,
        min_xy: Float[np.ndarray, "2"],
        max_xy: Float[np.ndarray, "2"],
        objects: Iterable[SceneObject[GeometryType]],
        obj_relevancy_scores: Float[np.ndarray, "N"],
        standard_deviations: Float[np.ndarray, "N"],
    ) -> OccupancyGrid:
        origin = np.floor(min_xy / self.config.resolution_meters) * self.config.resolution_meters
        grid_size = np.ceil((max_xy - origin) / self.config.resolution_meters).astype(int)
        grid = np.full(grid_size, 0.0)

        for relevancy, obj, std in zip(obj_relevancy_scores, objects, standard_deviations):
            if relevancy == 0:
                continue
            if (occ_grid := obj.get_ground_projection("exp", **self.projection_params)) is None:
                continue

            smoothed_grid = gaussian_filter(
                (occ_grid.grid > 0).astype(float), sigma=std / self.config.resolution_meters
            )
            smoothed_grid *= relevancy / np.sum(smoothed_grid) + 1e-10

            ox, oy = ((occ_grid.origin - origin) / self.config.resolution_meters).astype(int)
            h, w = occ_grid.grid.shape
            grid[ox : ox + h, oy : oy + w] += smoothed_grid

        grid_max = _max if (_max := np.max(grid)) > 0 else 1.0
        grid = (grid / grid_max * 255 - 128).astype(np.int8)
        return OccupancyGrid(origin=origin, resolution=self.config.resolution_meters, grid=grid)

    @staticmethod
    def _voxelize_points(
        points: Float[np.ndarray, "N 3"], resolution: float, padding_meters: float
    ) -> tuple[Float[np.ndarray, "3"], Bool[np.ndarray, "x y z"]]:
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        origin = np.floor(min_xyz / resolution) * resolution
        grid_size = np.ceil((max_xyz - origin) / resolution).astype(int) + 1

        grid = np.zeros(grid_size, dtype=bool)
        indices = ((points - origin) / resolution).astype(int)
        grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        padding_cells = int(padding_meters / resolution)
        origin -= padding_cells * resolution
        grid = np.pad(grid, pad_width=padding_cells, mode="constant", constant_values=False)
        return origin, grid

    def _build_3d_priority_map(
        self,
        min_xyz: Float[np.ndarray, "3"],
        voxelized_objects: list[Optional[tuple[Float[np.ndarray, "3"], Bool[np.ndarray, "x y z"]]]],
        obj_relevancy_scores: Float[np.ndarray, "N"],
        standard_deviations: Float[np.ndarray, "N"],
    ) -> SparseVoxelGrid:
        origin = np.floor(min_xyz / self.config.resolution_meters) * self.config.resolution_meters

        all_coords = []
        all_values = []
        for relevancy, std, voxelized in zip(obj_relevancy_scores, standard_deviations, voxelized_objects):
            if relevancy == 0 or voxelized is None:
                continue
            local_origin, occupied = voxelized

            smoothed_grid = gaussian_filter(occupied.astype(float), sigma=std / self.config.resolution_meters)
            smoothed_grid *= relevancy / np.sum(smoothed_grid) + 1e-10

            local_coords = np.argwhere(smoothed_grid > 0)
            if len(local_coords) == 0:
                continue
            offset = ((local_origin - origin) / self.config.resolution_meters).astype(int)
            all_coords.append(local_coords + offset)
            all_values.append(smoothed_grid[tuple(local_coords.T)])

        if not all_coords:
            return SparseVoxelGrid(
                origin=origin,
                resolution=self.config.resolution_meters,
                coords=np.zeros((0, 3), dtype=int),
                values=np.zeros((0,), dtype=np.int8),
            )

        coords, inverse = np.unique(np.concatenate(all_coords), axis=0, return_inverse=True)
        values = np.zeros(len(coords))
        np.add.at(values, inverse, np.concatenate(all_values))

        value_max = _max if (_max := np.max(values)) > 0 else 1.0
        values = (values / value_max * 255 - 128).astype(np.int8)
        return SparseVoxelGrid(origin=origin, resolution=self.config.resolution_meters, coords=coords, values=values)

    def get(
        self,
        objects: Iterable[SceneObject[GeometryType]],
    ) -> Optional[Union[OccupancyGrid, SparseVoxelGrid]]:
        """Compute the exploration priority map.

        Args:
            objects (Iterable[SceneObject[GeometryType]]): Iterable of SceneObject to consider for the exploration priority map.

        Returns:
            exploration_priority_map (Optional[Union[OccupancyGrid, VoxelGrid]]): Exploration priority map (2D or 3D depending on `build_3d`) with values between -128 (lowest priority) and 127 (highest), or None if it cannot be computed.

        """
        if self.exploration_priority is None or self.query is None:
            logger.warning("Exploration priority not set, cannot compute exploration priority map")
            return None

        if self.exploration_priority == ExplorationPriority.STATIONARITY:
            obj_relevancy_scores = self._get_object_relevancy_stationarity(objects)
        else:
            obj_relevancy_scores = np.array([self.obj_classes_relevancy.get(obj.class_name, 0.0) for obj in objects])
            perfect_match_mask = np.array([self.query == obj.class_name for obj in objects])
            if np.any(perfect_match_mask):
                logger.debug("Perfect match found, setting relevancy scores accordingly")
                obj_relevancy_scores[perfect_match_mask] = 1.0
                obj_relevancy_scores[~perfect_match_mask] = 0.0

            perfect_relevancy_found = np.any(obj_relevancy_scores == 1.0)
            if perfect_relevancy_found:
                logger.debug("Perfect (= 1.0) relevancy found. Zeroing other relevancy scores.")
                obj_relevancy_scores[obj_relevancy_scores < 1.0] = 0.0
            obj_relevancy_scores = 1 / (1 + np.exp(-obj_relevancy_scores))

        confidences = np.array([obj.stationarity_confidence for obj in objects])
        standard_deviations = (1 / confidences - 1) / (1 / self.config.search_radius_stationarity - 1) * (
            self.config.search_radius - self.config.measurement_std
        ) + self.config.measurement_std

        relevancy_threshold = self.config.minimum_relevancy
        if np.all(obj_relevancy_scores < self.config.minimum_relevancy):
            relevancy_threshold /= 2.0
        obj_relevancy_scores[obj_relevancy_scores < relevancy_threshold] = 0.0

        if self.config.dimensions == 3:
            voxelized_objects: list[Optional[tuple[Float[np.ndarray, "3"], Bool[np.ndarray, "x y z"]]]] = []
            min_xyz = np.array([np.inf, np.inf, np.inf])
            for obj in objects:
                points = obj.geometry.numpy_points()
                if len(points) == 0:
                    voxelized_objects.append(None)
                    continue
                local_origin, occupied = self._voxelize_points(
                    points, self.config.resolution_meters, self.projection_params["padding_meters"]
                )
                voxelized_objects.append((local_origin, occupied))
                min_xyz = np.minimum(min_xyz, local_origin)

            if np.any(np.isinf(min_xyz)):
                return None
            min_xyz -= self.config.map_padding_meters

            voxel_grid = self._build_3d_priority_map(
                min_xyz, voxelized_objects, obj_relevancy_scores, standard_deviations
            )
            log_sparse_voxel_grid(voxel_grid, "/world/exploration_priority_map_3d")
            return voxel_grid
        else:
            min_xy = np.array([np.inf, np.inf])
            max_xy = np.array([-np.inf, -np.inf])
            for obj in objects:
                occ_grid = obj.get_ground_projection("exp", **self.projection_params)
                if occ_grid is not None:
                    min_xy = np.minimum(min_xy, occ_grid.origin)
                    max_xy = np.maximum(
                        max_xy,
                        occ_grid.origin + np.array(occ_grid.grid.shape) * occ_grid.resolution,
                    )

            if np.any(np.isinf(min_xy)) or np.any(np.isinf(max_xy)):
                return None
            min_xy -= self.config.map_padding_meters
            max_xy += self.config.map_padding_meters

            occ_grid = self._build_2d_priority_map(min_xy, max_xy, objects, obj_relevancy_scores, standard_deviations)
            log_occupancy_grid(occ_grid, "/world/exploration_priority_map", colormap="raw")
            return occ_grid
