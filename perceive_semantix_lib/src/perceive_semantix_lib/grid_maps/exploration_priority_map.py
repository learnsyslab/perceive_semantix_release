from dataclasses import dataclass
from enum import Enum
from logging import getLogger
from typing import Iterable, Optional

import numpy as np
from jaxtyping import Float
from scipy.ndimage import gaussian_filter
from scipy.stats import beta

from perceive_semantix_lib.core.geometry import GeometryType
from perceive_semantix_lib.core.occupancy_grid import OccupancyGrid
from perceive_semantix_lib.core.scene_object import SceneObject
from perceive_semantix_lib.core.utils.optional_rerun_wrapper import log_occupancy_grid
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

    def get(self, objects: Iterable[SceneObject[GeometryType]]) -> Optional[OccupancyGrid]:
        """Compute the exploration priority map.

        Args:
            objects (Iterable[SceneObject[GeometryType]]): Iterable of SceneObject to consider for the exploration priority map.

        Returns:
            exploration_priority_map (Optional[OccupancyGrid]): Exploration priority map with values between -128 (lowest priority) and 127 (highest), or None if it cannot be computed.

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
        min_xy -= self.config.map_padding_meters
        max_xy += self.config.map_padding_meters

        if np.any(min_xy == np.array([np.inf, np.inf])) or np.any(max_xy == np.array([-np.inf, -np.inf])):
            return None

        confidences = np.array([obj.stationarity_confidence for obj in objects])
        standard_deviations = (1 / confidences - 1) / (1 / self.config.search_radius_stationarity - 1) * (
            self.config.search_radius - self.config.measurement_std
        ) + self.config.measurement_std

        origin = np.floor(min_xy / self.config.resolution_meters) * self.config.resolution_meters
        grid_size = np.ceil((max_xy - origin) / self.config.resolution_meters).astype(int)
        grid = np.full(grid_size, 0.0)

        relevancy_threshold = self.config.minimum_relevancy
        if np.all(obj_relevancy_scores < self.config.minimum_relevancy):
            relevancy_threshold /= 2.0

        for relevancy, obj, std in zip(obj_relevancy_scores, objects, standard_deviations):
            if relevancy < relevancy_threshold:
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
        occ_grid = OccupancyGrid(origin=origin, resolution=self.config.resolution_meters, grid=grid)
        log_occupancy_grid(occ_grid, "/world/exploration_priority_map", colormap="raw")
        return occ_grid
