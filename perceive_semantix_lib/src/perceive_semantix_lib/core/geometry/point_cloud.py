from typing import Optional, Self

import numpy as np
import torch
from jaxtyping import Float, UInt8
from open3d import geometry, utility
from open3d.pipelines import registration  # type: ignore

from perceive_semantix_lib.core.geometry.geometry_base import GeometryBase
from perceive_semantix_lib.core.occupancy_grid import OccupancyGrid


class PointCloud(GeometryBase):
    def __init__(self):
        self.pcd = geometry.PointCloud()

    def get_bounding_box(self) -> geometry.AxisAlignedBoundingBox:
        return self.pcd.get_axis_aligned_bounding_box()

    def merge_with(self, geom2: Self) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        self.pcd += geom2.pcd
        return self

    def to_occupancy_grid(
        self, resolution: float = 0.1, occupied_height_bounds: tuple[float, float] = (0.0, 10.0)
    ) -> Optional[OccupancyGrid]:
        points_np = np.asarray(self.pcd.points)
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

    @classmethod
    def batch_project_to_camera(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        geometries: list[Self],
        camera_pose: Float[torch.Tensor, "4 4"],
        camera_intrinsics_torch: Float[torch.Tensor, "3 3"],
        img_height: int,
        img_width: int,
        min_depth: float,
        max_depth: float,
        device: str = "cuda",
        preallocated_object_mask_count: int = 10,
        expected_visibility_threshold: float = 0.1,
    ) -> tuple[list[int], list[float], UInt8[torch.Tensor, "N H W"]]:
        object_projections = torch.zeros(
            (preallocated_object_mask_count, img_height, img_width), dtype=torch.uint8, device=device
        )
        number_expected_objects = 0
        expected_object_indices = []
        object_visibility_ratios = []

        camera_position = np.array(camera_pose[:3, -1])
        camera_pose_inv = np.linalg.inv(np.array(camera_pose))
        camera_intrinsics = np.array(camera_intrinsics_torch)
        for idx, obj in enumerate(geometries):
            # skip objects which are too far away
            if np.linalg.norm(obj.pcd.get_center() - camera_position) > max_depth * 2:
                continue

            points = np.asarray(obj.pcd.points)

            # transform points to camera frame
            points = camera_pose_inv @ np.vstack([points.T, np.ones(points.shape[0])])

            # remove points behind the camera
            uv = points[:, points[2, :] > 0]
            if uv.shape[1] == 0:
                continue

            # project to camera plane
            uv[0, :] /= uv[2, :]
            uv[1, :] /= uv[2, :]
            original_z = np.array(uv[2, :])
            uv[2, :] = 1
            uv = (camera_intrinsics @ uv[:3, :]).T

            total_points = uv.shape[0]
            valid_indices = (
                (original_z > 0)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < img_width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < img_height)
                & (original_z >= min_depth)
                & (original_z <= max_depth)
            )
            uv = uv[valid_indices, :]
            expected_points = uv.shape[0]

            uv = uv.astype(int)
            visibility_ratio = expected_points / total_points if total_points > 0 else 0.0
            if visibility_ratio < expected_visibility_threshold:
                continue

            if number_expected_objects >= len(object_projections):
                object_projections = torch.cat(
                    [object_projections, torch.zeros_like(object_projections, device=device)], dim=0
                )
            object_projections[number_expected_objects, uv[:, 1], uv[:, 0]] = 1
            expected_object_indices.append(idx)
            object_visibility_ratios.append(visibility_ratio)
            number_expected_objects += 1

        object_projections = object_projections[:number_expected_objects, :, :]

        return expected_object_indices, object_visibility_ratios, object_projections

    def to_serializable(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self.pcd.points), np.asarray(self.pcd.colors))

    @classmethod
    def from_serializable(cls, data: tuple[np.ndarray, Optional[np.ndarray]]) -> Self:
        pc = cls()
        pc.pcd = geometry.PointCloud()
        pc.pcd.points = utility.Vector3dVector(data[0])
        if data[1] is not None:
            pc.pcd.colors = utility.Vector3dVector(data[1])
        return pc

    @classmethod
    def from_point_tensor(
        cls,
        points: Float[torch.Tensor, "H W 3"],
        colors: UInt8[torch.Tensor, "H W 3"],
        min_points_threshold: Optional[int] = None,
        camera_pose: Optional[Float[torch.Tensor, "4 4"]] = None,
        obj_pcd_max_points: int = -1,
    ) -> Optional[Self]:
        valid_points_mask = points[:, :, 2] > 0
        if min_points_threshold is not None and torch.sum(valid_points_mask) < min_points_threshold:
            return None

        points = points[valid_points_mask]
        colors = colors[valid_points_mask]
        downsampled_points, downsampled_colors = cls._dynamic_downsample(
            points, colors=colors, target=obj_pcd_max_points
        )

        pcd = geometry.PointCloud()
        pcd.points = utility.Vector3dVector(downsampled_points.cpu().numpy())
        if downsampled_colors is not None:
            pcd.colors = utility.Vector3dVector((downsampled_colors.to(torch.float32) / 255.0).cpu().numpy())
        if camera_pose is not None:
            pcd.transform(camera_pose)
        obj = cls()
        obj.pcd = pcd
        return obj

    def postprocess_geometry(self, voxel_downsample_size: float) -> None:
        self.pcd = self.pcd.voxel_down_sample(voxel_downsample_size)

    def compute_overlap(self, other: Self, downsample_voxel_size: float, visibility_ratio: float = 1.0) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Compute overlap between this PointCloud and another PointCloud.

        The overlap between ``point_cloud_a`` and ``point_cloud_b`` is defined as the ratio of points in ``point_cloud_b`` that are within a maximum distance ``downsample_voxel_size`` to any point in ``point_cloud_a`` to the maximum number of matchable points.
        The maximum number of matchable points is defined as the minimum of the number of points in ``point_cloud_b`` and the number of points in ``point_cloud_a``.
        Optionally, by providing ``visibility_ratios_a``, it can be accounted for the fact that only a subset of points in ``point_cloud_a`` are expected to be visible.

        Given point clouds ``pcd_A``, and ``pcd_B``, their number of points ``|pcd_A|`` and ``|pcd_B|``, visibility ratio ``v_A`` in [0, 1] for ``pcd_A``, and the intersection of ``pcd_A`` and ``pcd_B`` is denoted as ``pcd_A ∩ pcd_B``, the overlap is computed as:
            overlap(pcd_A, pcd_B) = |pcd_A ∩ pcd_B| / min(|pcd_B|, |pcd_A| * v_A)

        Args:
            other (PointCloud): The other pointcloud.
            downsample_voxel_size (float): Two points are considered overlapping if they within this distance (in m).
            visibility_ratio (float): Visibility ratio to use for the point cloud ``self``.

        Returns:
            overlap (float): Overlap metric (0-1) of ``self`` and ``other``.

        """
        # get the distance of the nearest neighbor of each point in pcd_b to the pcd_a
        d = other.pcd.compute_point_cloud_distance(self.pcd)
        overlap = (np.asarray(d) < downsample_voxel_size).sum()

        # Calculate the expected number of points in pcd_a
        expected_points_a = int(len(self.pcd.points) * visibility_ratio)

        # Calculate the ratio of points within the threshold
        return overlap / min(expected_points_a, len(other.pcd.points))

    def register_geometries(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, target_geom: Self, init: Float[np.ndarray, "4 4"], max_correspondence_distance: float, max_iterations: int
    ) -> registration.RegistrationResult:
        return registration.registration_icp(
            self.pcd,
            target_geom.pcd,
            max_correspondence_distance,
            init,
            registration.TransformationEstimationPointToPoint(),
            registration.ICPConvergenceCriteria(max_iteration=max_iterations),
        )

    @staticmethod
    def _dynamic_downsample(
        points: Float[torch.Tensor, "N 3"],
        colors: Optional[UInt8[torch.Tensor, "N 3"]] = None,
        target: int = 5000,
    ) -> tuple[Float[torch.Tensor, "M 3"], Optional[UInt8[torch.Tensor, "M 3"]]]:
        """Downsampling function that dynamically adjusts the downsampling rate based on the number of input points.

        If a target of -1 is provided, downsampling is bypassed, returning the original points and colors.

        Args:
            points (torch.Tensor): Tensor of shape (N, 3) for N points.
            target (int): Target number of points to aim for in the downsampled output,
                        or -1 to bypass downsampling.
            colors (torch.Tensor, optional): Corresponding colors tensor of shape (N, 3).
                                            Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Downsampled points and optionally
                                                        downsampled colors, or the original
                                                        points and colors if target is -1.

        """
        # Check if downsampling is bypassed
        if target == -1:
            return points, colors

        num_points = points.size(0)

        # If the number of points is less than or equal to the target, return the original points and colors
        if num_points <= target:
            return points, colors

        # Calculate downsampling factor to aim for the target number of points
        downsample_factor = max(1, num_points // target)

        # Select points based on the calculated downsampling factor
        downsampled_points = points[::downsample_factor]

        # If colors are provided, downsample them with the same factor
        downsampled_colors = colors[::downsample_factor] if colors is not None else None

        return downsampled_points, downsampled_colors

    def __len__(self) -> int:
        """Return the number of points in the point cloud."""
        return len(self.pcd.points)

    def get_center(self, device="cpu") -> Float[torch.Tensor, "3"]:
        if device == "cpu":
            return torch.from_numpy(self.pcd.get_center())
        return torch.tensor(self.pcd.get_center(), device=device)

    def numpy_points(self) -> Float[np.ndarray, "N 3"]:
        return np.asarray(self.pcd.points)

    def numpy_colors(self) -> Float[np.ndarray, "N 3"]:
        if self.pcd.has_colors():
            return np.asarray(self.pcd.colors)
        return np.zeros((len(self), 3), dtype=np.float32)

    def crop(self, bounding_box: geometry.AxisAlignedBoundingBox) -> None:
        self.pcd = self.pcd.crop(bounding_box)
