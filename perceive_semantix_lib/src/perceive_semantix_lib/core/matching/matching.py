import bisect
from collections.abc import Sequence
from enum import Enum
from logging import getLogger
from typing import Iterable, Iterator, Optional, Union
from uuid import UUID

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from supervision.detection.core import Detections

from perceive_semantix_lib.core.config import ObjectMatchingConfig
from perceive_semantix_lib.core.geometry import GeometryType
from perceive_semantix_lib.core.scene_object import SceneObject
from perceive_semantix_lib.core.utils.similarity import (
    compute_invalid_measurement_2d_similarity,
    compute_overlap_matrix_general,
)

logger = getLogger(__name__)


class SimilarityMatrix:
    def __init__(self, object_ids: list[UUID], similarity: Float[torch.Tensor, "objects detections"]) -> None:
        if len(object_ids) != similarity.shape[0]:
            raise ValueError("Number of object IDs must match the first dimension of the similarity matrix.")
        self.data = similarity
        self.ids = {id: i for i, id in enumerate(object_ids)}

    def get_object_similarity(self, object_id: UUID) -> Float[torch.Tensor, "detections"]:
        if object_id not in self.ids:
            raise ValueError(f"Object ID {object_id} not found in similarity matrix.")
        return self.data[self.ids[object_id], :]

    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore


class AssociationType(Enum):
    MERGE = 1
    TRANSLATE = 2


class ObjectDetectionAdjacency:
    """Adjacency structure to represent associations between detections-objects and objects-objects.

    Internally a adjacency matrix in the form

        into (target)   detection_0 detection_1  ...  object_id_0  object_id_1  ...
    from (source)

    detection_0             0           0   ...       0           0
    detection_1             0           0   ...       X           0
       ...                 ...         ...   ...      ...         ...
    object_id_0             0           0   ...       0           0
    object_id_1             0           0   ...       0           0
       ...                 ...         ...   ...      ...         ...

    where if X == 1, detection_1 is merged into object_id_0. The values can be one of
    0 (no association), 1 (MERGE), or 2 (TRANSLATE) according to ``AssociationType``.
    """

    def __init__(self, detections: Detections, object_ids: list[UUID]):
        self.number_detections = len(detections)
        self.nodes = [i for i in range(len(detections))] + object_ids
        self.node_indices: dict[Union[int, UUID], int] = {node: i for i, node in enumerate(self.nodes)}
        self.adj_matrix = torch.zeros((len(self.nodes), len(self.nodes)), dtype=torch.uint8)
        self.transformations: dict[tuple[int, int], Float[torch.Tensor, "4 4"]] = {}

    def __len__(self) -> int:
        """Get the number of nodes in the graph (detections + objects)."""
        return len(self.nodes)

    def clear_detection_edges(self) -> None:
        self.adj_matrix[: self.number_detections, :] = 0
        self.adj_matrix[:, : self.number_detections] = 0

    def _add_edge(
        self, merge: int, into: int, type: AssociationType, transformation: Optional[Float[torch.Tensor, "4 4"]] = None
    ) -> None:
        self.adj_matrix[merge, into] = type.value
        if transformation is not None:
            self.transformations[(merge, into)] = transformation

    def add_edge(
        self,
        merge: Union[int, UUID],
        into: Union[int, UUID],
        type: AssociationType,
        transformation: Optional[Float[torch.Tensor, "4 4"]] = None,
    ) -> None:
        if self.is_merged_source(merge) and self.get_merge_target(merge)[0] != into:
            logger.error(f"Node {merge} is already a source of a merge.")
            raise ValueError(f"Node {merge} is already a source of a merge.")
        if isinstance(into, int) and into < self.number_detections:
            logger.error(f"Cannot merge into a detection node {into}.")
            raise ValueError(f"Cannot merge into a detection node {into}.")
        self._add_edge(self.node_indices[merge], self.node_indices[into], type, transformation)

    def is_merged_source(self, obj: Union[int, UUID], include_detections: bool = True) -> bool:
        idx = self.node_indices[obj]
        start_index = 0 if include_detections else self.number_detections
        return bool(torch.sum(self.adj_matrix[idx, start_index:]) > 0)

    def is_merged_target(self, obj: Union[int, UUID], include_detections: bool = True) -> bool:
        idx = self.node_indices[obj]
        start_index = 0 if include_detections else self.number_detections
        return bool(torch.sum(self.adj_matrix[start_index:, idx]) > 0)

    def get_merge_target(
        self, source: Union[int, UUID], type: Optional[AssociationType] = None, include_detections: bool = True
    ) -> tuple[Optional[UUID], Optional[AssociationType], Optional[Float[torch.Tensor, "4 4"]]]:
        if not self.is_merged_source(source, include_detections=include_detections):
            return None, None, None
        source_idx = self.node_indices[source]

        start_index = 0 if include_detections else self.number_detections
        if type is None:
            target_index = torch.nonzero(self.adj_matrix[source_idx, start_index:] > 0)[0, 0] + start_index
        else:
            target_index = torch.nonzero(self.adj_matrix[source_idx, start_index:] == type.value)[0, 0] + start_index
        if target_index < self.number_detections:
            logger.error(f"Source {source} is merged into a detection, which is not allowed.")
            raise ValueError(f"Source {source} is merged into a detection, which is not allowed.")
        return (
            self.nodes[target_index],  # type: ignore
            AssociationType(int(self.adj_matrix[source_idx, target_index])),
            self.transformations.get((source_idx, target_index), None),  # type: ignore
        )

    def get_merge_targets(
        self,
        sources: Iterator[Union[int, UUID]],
        type: Optional[AssociationType] = None,
        include_detections: bool = True,
    ) -> Iterator[tuple[Optional[UUID], Optional[AssociationType], Optional[Float[torch.Tensor, "4 4"]]]]:
        for source in sources:
            yield self.get_merge_target(source, type, include_detections=include_detections)

    def get_merge_sources(
        self, target: Union[int, UUID], type: Optional[AssociationType] = None, include_detections: bool = True
    ) -> list[tuple[Union[int, UUID], AssociationType, Optional[Float[torch.Tensor, "4 4"]]]]:
        if not self.is_merged_target(target, include_detections=include_detections):
            return []
        target_index = self.node_indices[target]

        start_index = 0 if include_detections else self.number_detections
        source_indices = torch.nonzero(self.adj_matrix[start_index:, target_index] > 0).squeeze(1) + start_index
        if type is not None:
            source_indices = source_indices[self.adj_matrix[source_indices, target_index] == type.value]
        source_indices = source_indices.tolist()
        return [
            (
                self.nodes[i],
                AssociationType(int(self.adj_matrix[i, target_index])),
                self.transformations.get((i, target_index), None),
            )
            for i in source_indices
        ]

    def is_empty(self) -> bool:
        return bool(torch.sum(self.adj_matrix) == 0)

    def get_nodes_edges(
        self, detection_prefix: str = ""
    ) -> tuple[list[Union[str, UUID]], list[tuple[Union[str, UUID], Union[str, UUID]]]]:
        def add_prefix(node: Union[int, UUID]) -> Union[str, UUID]:
            if isinstance(node, int):
                return f"{detection_prefix}{node}"
            return node

        edges = []
        for i in range(self.adj_matrix.shape[0]):
            for j in range(self.adj_matrix.shape[1]):
                if self.adj_matrix[i, j] > 0:
                    edges.append((add_prefix(self.nodes[i]), add_prefix(self.nodes[j])))
        nodes = [add_prefix(n) for n in self.nodes]
        return nodes, edges

    def add_object_nodes(self, new_object_ids: Iterable[UUID]) -> None:
        for obj_id in new_object_ids:
            if not isinstance(obj_id, UUID):
                raise ValueError("Only UUIDs can be added as new object nodes.")
            if obj_id in self.nodes:
                continue
            self.nodes.append(obj_id)
            self.node_indices[obj_id] = len(self.nodes) - 1
        new_size = len(self.nodes)
        new_adj_matrix = torch.zeros((new_size, new_size), dtype=torch.uint8)
        new_adj_matrix[: self.adj_matrix.shape[0], : self.adj_matrix.shape[1]] = self.adj_matrix
        self.adj_matrix = new_adj_matrix


def compute_spatial_similarities(
    objects: Sequence[SceneObject[GeometryType]],
    candidate_objects: Detections,
    objects_visibility_ratios: Optional[list[float]] = None,
    downsample_voxel_size: float = 0.01,
    compensate_invalid_depth: bool = True,
    object_projections: Optional[Bool[torch.Tensor, "objects H W"]] = None,
    invalid_depth_measurement: Optional[Bool[torch.Tensor, "H W"]] = None,
) -> Float[torch.Tensor, "objects candidates"]:
    """Compute spatial similarity a list of objects and a list of candidate objects.

    Args:
        objects (list[SceneObject[GeometryType]]): List of SceneObject instances.
        candidate_objects (Detections): Detections instance containing candidate objects.
        objects_visibility_ratios (Optional[list[float]]): List of visibility ratios for each object (not for each candidate object).
        downsample_voxel_size (float): Voxel size used to compute overlap, points within this distance are considered overlapping.
        compensate_invalid_depth (bool): If True, compute similarity of masks and projections onto camera in regions where no depth could be measured.
        object_projections (Optional[Bool[torch.Tensor, "objects H W"]]): Boolean tensor indicating the 2D projection of each object in the camera view.
        invalid_depth_measurement (Optional[Bool[torch.Tensor, "H W"]]): Boolean tensor indicating where depth measurements are invalid in the camera view.

    Returns:
        Float[torch.Tensor, "objects candidates"]: Similarity matrix of shape (len(objects), len(candidate_objects)).

    """
    if len(objects) == 0 or len(candidate_objects) == 0:
        return torch.empty((len(objects), len(candidate_objects)))

    if (
        candidate_objects.data is None
        or "point_clouds" not in candidate_objects.data
        or "bounding_boxes" not in candidate_objects.data
    ):
        logger.error("Candidate objects must have 'point_clouds' and 'bounding_boxes' in their data.")
        raise ValueError("Candidate objects must have 'point_clouds' and 'bounding_boxes' in their data.")

    spatial_sim = compute_overlap_matrix_general(
        point_clouds_a=[obj.geometry for obj in objects],
        bounding_boxes_a=[obj.bounding_box for obj in objects],
        visibility_ratios_a=objects_visibility_ratios,
        point_clouds_b=candidate_objects.data["point_clouds"],  # type: ignore
        bounding_boxes_b=candidate_objects.data["bounding_boxes"],  # type: ignore
        downsample_voxel_size=downsample_voxel_size,
    )

    # If provided, compute similarity of masks and projections onto camera in regions where no depth could be measured
    if compensate_invalid_depth and object_projections is not None and invalid_depth_measurement is not None:
        projection_similarity = compute_invalid_measurement_2d_similarity(
            A=object_projections,
            B=torch.as_tensor(candidate_objects.mask).to(object_projections.device),
            C=invalid_depth_measurement.to(object_projections.device),
        )
        spatial_sim += projection_similarity.to(spatial_sim.device)
        spatial_sim = spatial_sim.clamp(min=0.0, max=1.0)

    return spatial_sim


def compute_visual_similarities(
    objects: Sequence[SceneObject[GeometryType]],
    candidate_objects_clip_features: Float[torch.Tensor, "candidates clip"],
) -> SimilarityMatrix:
    """Compute visual similarity between a list of objects and candidate objects using their CLIP features.

    Args:
        objects (list[SceneObject[GeometryType]]): List of N SceneObject instances.
        candidate_objects_clip_features (Float[torch.Tensor, "candidates clip"]): CLIP features of M candidate objects.

    Returns:
        Float[torch.Tensor, "objects candidates"]: Similarity matrix of shape (N, M).

    """
    if len(objects) == 0 or candidate_objects_clip_features.shape[0] == 0:
        return SimilarityMatrix(
            [obj.id for obj in objects], torch.empty((len(objects), candidate_objects_clip_features.shape[0]))
        )
    objects_clip_features = torch.stack([o.visual_feature for o in objects], dim=0).float()

    objects_clip_features = objects_clip_features.unsqueeze(-1)  # (objects, D, 1)
    candidate_objects_clip_features = candidate_objects_clip_features.T.unsqueeze(0)  # (1, D, candidates)

    sim = F.cosine_similarity(objects_clip_features, candidate_objects_clip_features, dim=1)
    return SimilarityMatrix([obj.id for obj in objects], sim)


def match_detections_to_objects(
    candidates: Detections,
    spatial_similarity: SimilarityMatrix,
    visual_similarity: SimilarityMatrix,
    adjacency: "ObjectDetectionAdjacency",
    spatial_threshold: float = 0.5,
    visual_threshold: float = 0.5,
    prioritize_semantic_sim: bool = False,
    matching_method: str = "sep_thresh",
) -> None:
    """Match detections to existing objects based on spatial and visual similarity.

    Args:
        candidates (Detections): Detections instance containing candidate detections.
        spatial_similarity (SimilarityMatrix): Similarity matrix containing spatial similarity scores between objects and candidates.
        visual_similarity (SimilarityMatrix): Similarity matrix containing visual similarity scores between objects and candidates.
        adjacency (ObjectDetectionAdjacency): Adjacency structure to record matches.
        spatial_threshold (float): Minimum spatial similarity to consider a match.
        visual_threshold (float): Minimum visual similarity to consider a match.
        prioritize_semantic_sim (bool): If True, prioritize visual similarity when selecting the best match.
        matching_method (str): Matching method to use. Currently only "sep_thresh" is supported.

    Returns:
        None: The function modifies the 'adjacency' in place, adding edges for matched detections.

    """
    if matching_method != "sep_thresh":
        logger.error(f"Unknown matching method: {matching_method}")
        raise ValueError(f"Unknown matching method: {matching_method}")
    if spatial_similarity.shape()[1] != len(candidates):
        logger.error("spatial_similarity second dimension must match number of candidates")
        raise ValueError("spatial_similarity second dimension must match number of candidates")
    if visual_similarity.shape()[1] != len(candidates):
        logger.error("visual_similarity second dimension must match number of candidates")
        raise ValueError("visual_similarity second dimension must match number of candidates")
    matches: list[Optional[UUID]] = [None] * len(candidates)
    combined_match_similarity: list[float] = [0] * len(candidates)
    if visual_similarity.shape()[0] == 0:
        return

    if prioritize_semantic_sim:
        best_indices = torch.argmax(visual_similarity.data, dim=0)
    else:
        best_indices = torch.argmax(spatial_similarity.data, dim=0)

    for i, best_idx in enumerate(best_indices):
        spatial_sim = spatial_similarity.data[best_idx, i]
        visual_sim = visual_similarity.data[best_idx, i]
        if spatial_sim >= spatial_threshold and visual_sim >= visual_threshold:
            potential_match = list(spatial_similarity.ids.keys())[best_idx]
            potential_combined_match_similarity = float(spatial_sim + visual_sim)

            # If this object was already matched, only keep the match with the highest combined similarity
            if potential_match in matches:
                idx = matches.index(potential_match)
                if potential_combined_match_similarity > combined_match_similarity[idx]:
                    matches[i] = potential_match
                    matches[idx] = None
                    combined_match_similarity[i] = potential_combined_match_similarity
                    combined_match_similarity[idx] = 0
            else:
                matches[i] = potential_match
                combined_match_similarity[i] = potential_combined_match_similarity
    for i, match_id in enumerate(matches):
        if match_id is not None:
            adjacency.add_edge(i, match_id, AssociationType.MERGE)


def match_detections_to_objects_semantically_greedy_icp(
    objects: Iterable[SceneObject[GeometryType]],
    candidates: Detections,
    adjacency: ObjectDetectionAdjacency,
    visual_similarity: SimilarityMatrix,
    config: ObjectMatchingConfig = ObjectMatchingConfig(),
) -> None:
    """Match detections to existing objects based on spatial and visual similarity.

    Modifes the visual_similarity matrix in place to avoid matching the same detection multiple times.

    Args:
        objects (list[SceneObject[GeometryType]]): List of existing SceneObject instances.
        candidates (Detections): Detections instance containing candidate detections.
        adjacency (ObjectDetectionAdjacency): Adjacency structure to record matches.
        visual_similarity (SimilarityMatrix): Similarity matrix containing visual similarity scores between objects and candidates.
        config (ObjectMatchingConfig): Configuration parameters for matching. Using the configuration parameters from `config`:
            - visual_similarity_threshold
            - icp_threshold
            - icp_max_iteration
            - icp_inlier_rmse_threshold

    Returns:
        None: The function modifies the 'adjacency' in place, adding edges for matched detections.

    """
    if visual_similarity.shape()[1] != len(candidates):
        logger.error("visual_similarity second dimension must match number of candidates")
        raise ValueError("visual_similarity second dimension must match number of candidates")
    if candidates.data is None or "point_clouds" not in candidates.data:
        logger.error("Candidate objects must have 'point_clouds' in their data.")
        raise ValueError("Candidate objects must have 'point_clouds' in their data.")

    matched_detections_mask = [adjacency.is_merged_source(i) for i in range(len(candidates))]
    visual_similarity.data[:, matched_detections_mask] = -1.0

    for obj in objects:
        potential_matches_mask = visual_similarity.get_object_similarity(obj.id) > config.visual_similarity_threshold
        if not torch.any(potential_matches_mask):
            continue

        max_ind = int(visual_similarity.get_object_similarity(obj.id).argmax().item())
        registration_results = obj.geometry.register_geometries(
            candidates.data["point_clouds"][max_ind], np.eye(4), config.icp_threshold, config.icp_max_iteration
        )
        if (
            registration_results is not None
            and registration_results.fitness > 0
            and registration_results.inlier_rmse < config.icp_inlier_rmse_threshold
        ):
            adjacency.add_edge(max_ind, obj.id, AssociationType.TRANSLATE)
            logger.debug(
                f"Matched detection {max_ind} to object {obj.id} with ICP inlier RMSE {registration_results.inlier_rmse:.4f} and visual similarity {float(visual_similarity.get_object_similarity(obj.id)[max_ind]):.4f}"
            )
            logger.debug(f"ICP results:\n{registration_results}")
            visual_similarity.data[:, max_ind] = -1.0


def match_objects_to_objects(
    objects_to_reidentify: Sequence[SceneObject[GeometryType]],
    objects_to_check_against: Sequence[SceneObject[GeometryType]],
    adjacency: ObjectDetectionAdjacency,
    last_frame_time_sec: Optional[float] = None,
    config: ObjectMatchingConfig = ObjectMatchingConfig(),
) -> None:
    """Match objects with those that appeared within a specific time window around their disappearance time.

    Args:
        objects_to_reidentify (Iterable[SceneObject[GeometryType]]): Objects that disappeared and need to be reidentified.
        objects_to_check_against (Sequence[SceneObject[GeometryType]]): Objects that are currently present and can be matched against.
        adjacency (ObjectDetectionAdjacency): Adjacency structure to record matches.
        last_frame_time_sec (Optional[float]): Timestamp of the last processed frame. If not None, sips any object that is too old to have potential new matches.
        config (ObjectMatchingConfig): Configuration parameters for matching. Using the configuration parameters from `config`:
            - reidentification_lookback_seconds
            - reidentification_lookforward_seconds
            - visual_similarity_threshold_reidentification
            - icp_threshold
            - icp_max_iteration
            - icp_inlier_rmse_threshold

    Returns:
        None: The function modifies the 'adjacency' in place, adding edges for matched objects.

    """
    time_object_pairs = [
        (obj.observation_times[0], obj) for obj in objects_to_check_against if obj not in objects_to_reidentify
    ]
    if len(time_object_pairs) == 0:
        return
    time_object_pairs.sort(key=lambda x: x[0])
    sorted_times: list[float]
    sorted_objects: list[SceneObject[GeometryType]]
    sorted_times, sorted_objects = map(list, zip(*time_object_pairs))

    for obj in objects_to_reidentify:
        # Ignore objects that disappeared too long ago (these were already checked in previous frames)
        if (
            last_frame_time_sec is not None
            and obj.time_of_disappearance + config.reidentification_lookforward_seconds < last_frame_time_sec
        ):
            continue
        timeframe_start_index = bisect.bisect_left(
            sorted_times, obj.time_of_disappearance - config.reidentification_lookback_seconds
        )
        objects_within_timeframe = sorted_objects[
            timeframe_start_index : bisect.bisect_right(
                sorted_times, obj.time_of_disappearance + config.reidentification_lookforward_seconds
            )
        ]

        if len(objects_within_timeframe) == 0:
            continue

        visual_sim = compute_visual_similarities(objects_within_timeframe, obj.visual_feature.unsqueeze(0))
        visually_most_similar_idx = int(torch.argmax(visual_sim.data).item())
        if visual_sim.data[visually_most_similar_idx, 0] < config.visual_similarity_threshold_reidentification:
            continue

        most_similar_obj = objects_within_timeframe[visually_most_similar_idx]

        init_tf = np.eye(4)
        init_tf[:3, 3] = most_similar_obj.geometry.get_center() - obj.geometry.get_center()

        registration_results = obj.geometry.register_geometries(
            most_similar_obj.geometry,
            init_tf,
            config.icp_threshold,
            config.icp_max_iteration,
        )
        logger.debug(f"trying to reidentify object {most_similar_obj.id_str} as {obj.id_str}")
        logger.debug(
            f"with ICP inlier RMSE {registration_results.inlier_rmse:.4f} and visual similarity {float(visual_sim.data[visually_most_similar_idx, 0]):.4f}"
        )
        logger.debug(f"ICP results:\n{registration_results}")
        if (
            registration_results is not None
            and registration_results.fitness > 0
            and registration_results.inlier_rmse < config.icp_inlier_rmse_threshold
        ):
            if (
                np.linalg.norm(registration_results.transformation[:3, 3])
                < config.reidentification_merge_vs_translate_threshold_meters
            ):
                adjacency.add_edge(most_similar_obj.id, obj.id, AssociationType.MERGE)
                logger.debug(f"Reidentified object {most_similar_obj.id_str} as {obj.id_str} with MERGE")
            else:
                adjacency.add_edge(most_similar_obj.id, obj.id, AssociationType.TRANSLATE)
                logger.debug(f"Reidentified object {most_similar_obj.id_str} as {obj.id_str} with TRANSLATE")
            sorted_times.pop(timeframe_start_index + visually_most_similar_idx)
            sorted_objects.pop(timeframe_start_index + visually_most_similar_idx)
