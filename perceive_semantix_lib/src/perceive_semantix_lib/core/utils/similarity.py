from logging import getLogger
from typing import Optional

import numpy as np
import torch
from jaxtyping import Bool, Float
from open3d.geometry import AxisAlignedBoundingBox  # pyright: ignore[reportMissingImports]
from pytorch3d.ops import box3d_overlap

from perceive_semantix_lib.core.geometry import GeometryType

logger = getLogger(__name__)


def expand_3d_box(bbox: Float[torch.Tensor, "N 8 D"], eps: float = 0.02) -> Float[torch.Tensor, "N 8 D"]:
    """Expand the side of 3D boxes such that each side has at least eps length.

    Assumes the bbox corner order in open3d convention.

    Args:
        bbox (N, 8, D): N boxes with 8 corners in D dimensions
        eps (float): minimum length for each side

    Returns:
        new_bbox (N, 8, D): expanded boxes

    """
    center = bbox.mean(dim=1)  # shape: (N, D)

    va = bbox[:, 1, :] - bbox[:, 0, :]  # shape: (N, D)
    vb = bbox[:, 2, :] - bbox[:, 0, :]  # shape: (N, D)
    vc = bbox[:, 3, :] - bbox[:, 0, :]  # shape: (N, D)

    a = torch.linalg.vector_norm(va, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    b = torch.linalg.vector_norm(vb, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    c = torch.linalg.vector_norm(vc, ord=2, dim=1, keepdim=True)  # shape: (N, 1)

    va = torch.where(a < eps, va / a * eps, va)  # shape: (N, D)
    vb = torch.where(b < eps, vb / b * eps, vb)  # shape: (N, D)
    vc = torch.where(c < eps, vc / c * eps, vc)  # shape: (N, D)

    new_bbox = torch.stack(
        [
            center - va / 2.0 - vb / 2.0 - vc / 2.0,
            center + va / 2.0 - vb / 2.0 - vc / 2.0,
            center - va / 2.0 + vb / 2.0 - vc / 2.0,
            center - va / 2.0 - vb / 2.0 + vc / 2.0,
            center + va / 2.0 + vb / 2.0 + vc / 2.0,
            center - va / 2.0 + vb / 2.0 + vc / 2.0,
            center + va / 2.0 - vb / 2.0 + vc / 2.0,
            center + va / 2.0 + vb / 2.0 - vc / 2.0,
        ],
        dim=1,
    )  # shape: (N, 8, D)

    new_bbox = new_bbox.to(bbox.device)
    new_bbox = new_bbox.type(bbox.dtype)

    return new_bbox


def compute_3d_iou_accurate_batch(
    bbox1: Float[torch.Tensor, "M 8 3"], bbox2: Float[torch.Tensor, "N 8 3"]
) -> Float[torch.Tensor, "M N"]:
    """Compute IoU between two sets of oriented (or axis-aligned) 3D bounding boxes.

    bbox1: (M, 8, D), e.g. (M, 8, 3)
    bbox2: (N, 8, D), e.g. (N, 8, 3)

    returns: (M, N)
    """
    # Must expend the box beforehand, otherwise it may results overestimated results
    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)

    bbox1 = bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    bbox2 = bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]]

    _, iou = box3d_overlap(bbox1.float(), bbox2.float())

    return iou


def compute_overlap_matrix_general(
    point_clouds_a: list[GeometryType],
    bounding_boxes_a: list[AxisAlignedBoundingBox],
    visibility_ratios_a: Optional[list[float]] = None,
    point_clouds_b: Optional[list[GeometryType]] = None,
    bounding_boxes_b: Optional[list[AxisAlignedBoundingBox]] = None,
    downsample_voxel_size: float = 0.01,
) -> Float[torch.Tensor, "A B"]:
    r"""Compute overlap matrix between two sets of point clouds and bounding boxes.

    The overlap between ``point_cloud_a`` and ``point_cloud_b`` is defined as the ratio of points in ``point_cloud_b`` that are within a maximum distance ``downsample_voxel_size`` to any point in ``point_cloud_a`` to the maximum number of matchable points.
    The maximum number of matchable points is defined as the minimum of the number of points in ``point_cloud_b`` and the number of points in ``point_cloud_a``.
    Optionally, by providing ``visibility_ratios_a``, it can be accounted for the fact that only a subset of points in ``point_cloud_a`` are expected to be visible.

    Given point clouds ``pcd_A``, and ``pcd_B``, their number of points ``len(pcd_A)`` and ``len(pcd_B)``, visibility ratio ``v_A`` in [0, 1] for ``pcd_A``, and the intersection of ``pcd_A`` and ``pcd_B`` is denoted as ``pcd_A ∩ pcd_B``, the overlap is computed as:
        overlap(pcd_A, pcd_B) = |pcd_A ∩ pcd_B| / min(len(pcd_B), len(pcd_A) * v_A)

    Args:
        point_clouds_a (list[GeometryType]): List of point clouds A.
        bounding_boxes_a (list[geometry.AxisAlignedBoundingBox]): List of bounding boxes A.
        visibility_ratios_a (list[float], optional): List of visibility ratios for point clouds A.
        point_clouds_b (list[GeometryType], optional): List of point clouds B.
        bounding_boxes_b (list[geometry.AxisAlignedBoundingBox], optional): List of bounding boxes B.
        downsample_voxel_size (float, optional): Maximum distance threshold to consider a point as overlapping.

    Returns:
        overlap (Float[torch.Tensor, "A B"]): Overlap matrix of shape (len(point_clouds_a), len(point_clouds_b))

    """
    if len(point_clouds_a) != len(bounding_boxes_a):
        logger.error("Length of point_clouds_a and bounding_boxes_a must be the same.")
        raise ValueError("Length of point_clouds_a and bounding_boxes_a must be the same.")
    if visibility_ratios_a is not None and len(visibility_ratios_a) != len(point_clouds_a):
        logger.error("Length of visibility_ratios_a must be the same as point_clouds_a if provided.")
        raise ValueError("Length of visibility_ratios_a must be the same as point_clouds_a if provided.")
    if (point_clouds_b is None) != (bounding_boxes_b is None):
        logger.error("Both point_clouds_b and bounding_boxes_b should be provided or neither.")
        raise ValueError("Both point_clouds_b and bounding_boxes_b should be provided or neither.")
    if point_clouds_b is not None and bounding_boxes_b is not None and len(point_clouds_b) != len(bounding_boxes_b):
        logger.error("Length of point_clouds_b and bounding_boxes_b must be the same.")
        raise ValueError("Length of point_clouds_b and bounding_boxes_b must be the same.")

    # check for self comparision
    if point_clouds_b is None:
        point_clouds_b = point_clouds_a
        bounding_boxes_b = bounding_boxes_a
        self_comparision = True
    else:
        self_comparision = False

    if len(point_clouds_a) == 0 or len(point_clouds_b) == 0:
        return torch.empty((len(point_clouds_a), len(point_clouds_b)))

    # compute bounding box overlap
    bbox_a = torch.stack(
        [torch.from_numpy(np.asarray(bb.get_box_points())) for bb in bounding_boxes_a], dim=0
    ).float()  # (A, 8, 3)
    bbox_b = torch.stack([torch.from_numpy(np.asarray(bb.get_box_points())) for bb in bounding_boxes_b], dim=0).float()  # type: ignore # (B, 8, 3)
    ious = compute_3d_iou_accurate_batch(bbox_a, bbox_b)

    # Compute the pairwise overlaps
    overlap_matrix = torch.zeros((len(point_clouds_a), len(point_clouds_b)), dtype=torch.float32)
    if visibility_ratios_a is None:
        visibility_ratios_a_list = [1.0] * len(point_clouds_a)
    else:
        visibility_ratios_a_list = visibility_ratios_a
    for idx_a, (pcd_a, visibility_ratio_a) in enumerate(zip(point_clouds_a, visibility_ratios_a_list)):
        for idx_b, pcd_b in enumerate(point_clouds_b):
            # skip same object comparisons
            if self_comparision and idx_a <= idx_b:
                continue

            # skip if the boxes do not overlap at all
            if ious[idx_a, idx_b] < 1e-6:
                continue

            # get the distance of the nearest neighbor of each point in pcd_b to the pcd_a
            overlap_matrix[idx_a, idx_b] = pcd_a.compute_overlap(
                pcd_b, downsample_voxel_size, visibility_ratio=visibility_ratio_a
            )
            if self_comparision:
                overlap_matrix[idx_b, idx_a] = overlap_matrix[idx_a, idx_b]

    return overlap_matrix


def compute_invalid_measurement_2d_similarity(
    A: Bool[torch.Tensor, "N H W"],
    B: Bool[torch.Tensor, "M H W"],
    C: Bool[torch.Tensor, "H W"],
    A_str: Optional[list[str]] = None,
    B_str: Optional[list[str]] = None,
    visualize: bool = False,
) -> Float[torch.Tensor, "N M"]:
    """Compute similarity between masks A and B in masked region defined by C.

    result[n, m] = sum(A[n] * B[m] * C) / max(sum(A[n]), sum(B[m]))

    Args:
        A (Tensor): Tensor of shape (N, H, W)
        B (Tensor): Tensor of shape (M, H, W)
        C (Tensor): Tensor of shape (H, W)
        A_str (list[str], optional): Optional list of strings for A samples for visualization.
        B_str (list[str], optional): Optional list of strings for B samples for visualization.
        visualize (bool, optional): Whether to visualize the similarity grid for debugging.

    Returns:
        Tensor: Tensor of shape (N, M)

    """
    if A.ndim != 3:
        raise ValueError(f"A must be 3D (N, H, W), got {A.shape}")
    if B.ndim != 3:
        raise ValueError(f"B must be 3D (M, H, W), got {B.shape}")
    if C.ndim != 2:
        raise ValueError(f"C must be 2D (H, W), got {C.shape}")
    if A.shape[1:] != B.shape[1:] or A.shape[1:] != C.shape:
        raise ValueError("Spatial dimensions of A, B, and C must match")
    # Ensure all tensors are on the same device
    if not (A.device == B.device == C.device):
        raise ValueError(f"A, B, and C must be on the same device. Got A: {A.device}, B: {B.device}, C: {C.device}")

    N, H, W = A.shape
    M = B.shape[0]

    if N == 0 or M == 0:
        return torch.empty((N, M), device=A.device)

    A_exp = A.unsqueeze(1)  # (N, 1, H, W)
    B_exp = B.unsqueeze(0)  # (1, M, H, W)
    C_exp = C.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    numerator = (A_exp * B_exp * C_exp).sum(dim=(2, 3))  # (N, M)

    A_sum = A.view(N, -1).sum(dim=1).unsqueeze(1)  # (N, 1)
    B_sum = B.view(M, -1).sum(dim=1).unsqueeze(0)  # (1, M)
    denominator = torch.max(A_sum, B_sum).clamp(min=1e-8)  # (N, M)

    result = numerator / denominator

    # Visualization for debugging: visualize A, B, C as images (first few samples)
    if visualize:
        import matplotlib.pyplot as plt

        def visualize_similarity_grid(
            A: torch.Tensor,  # (N, H, W)
            B: torch.Tensor,  # (M, H, W)
            C: torch.Tensor,  # (H, W)
            result: torch.Tensor,  # (N, M)
            A_str: Optional[list[str]] = None,
            B_str: Optional[list[str]] = None,
            max_elements: int = 4,
            fig_num: int = 1,
        ) -> None:
            """Visualize A and B masks along with similarity results in a grid.

            First row: A samples
            First column: B samples
            Grid[i, j] = result[j-1, i-1] (similarity of A[i-1] and B[j-1])
            """
            N, M = A.shape[0], B.shape[0]
            num_A = min(max_elements, N)
            num_B = min(max_elements, M)

            fig = plt.figure(num=fig_num)
            plt.clf()  # Clear the figure if it already exists
            fig, axs = plt.subplots(num_A + 1, num_B + 1, figsize=((num_B + 1) * 3, (num_A + 1) * 3), num=fig_num)
            fig.suptitle("Similarity Grid")

            # Top-left cell: show C
            axs[0, 0].imshow(C.detach().cpu().numpy())
            axs[0, 0].set_title("Invalid Depth Measurement")
            axs[0, 0].axis("off")

            # First column: A samples
            for i in range(num_A):
                axs[i + 1, 0].imshow(A[i].detach().cpu().numpy())
                title = A_str[i] if A_str and i < len(A_str) else f"A[{i}]"
                axs[i + 1, 0].set_title(title)
                axs[i + 1, 0].axis("off")

            # First row: B samples
            for j in range(num_B):
                axs[0, j + 1].imshow(B[j].detach().cpu().numpy())
                title = B_str[j] if B_str and j < len(B_str) else f"B[{j}]"
                axs[0, j + 1].set_title(title)
                axs[0, j + 1].axis("off")

            # Fill in similarity result matrix
            for i in range(num_A):
                for j in range(num_B):
                    sim_value = result[i, j].item()
                    axs[i + 1, j + 1].text(0.5, 0.5, f"{sim_value:.3f}", ha="center", va="center", fontsize=12)
                    axs[i + 1, j + 1].set_xticks([])
                    axs[i + 1, j + 1].set_yticks([])
                    axs[i + 1, j + 1].set_frame_on(False)

            plt.tight_layout()

        visualize_similarity_grid(A=A, B=B, C=C, result=result, A_str=A_str, B_str=B_str, max_elements=4, fig_num=4)
        plt.pause(0.02)

    return result
