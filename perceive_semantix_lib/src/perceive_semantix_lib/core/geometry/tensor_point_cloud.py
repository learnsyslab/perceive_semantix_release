from logging import getLogger
from typing import Optional, Self

import numpy as np
import open3d.core as o3c  # pyright: ignore[reportMissingImports]
import torch
from jaxtyping import Bool, Float, UInt8
from open3d import geometry as geometry_legacy  # pyright: ignore[reportMissingImports]
from open3d.pipelines import registration as registration_legacy  # type: ignore
from open3d.t import geometry  # pyright: ignore[reportMissingImports]
from open3d.t.pipelines import registration  # type: ignore
from torch.utils import dlpack as torch_dlpack

from perceive_semantix_lib.core.geometry.geometry_base import GeometryBase
from perceive_semantix_lib.core.occupancy_grid import OccupancyGrid

logger = getLogger(__name__)


class TensorPointCloud(GeometryBase):
    device: o3c.Device = o3c.Device("CPU:0")

    def __init__(self) -> None:
        self.pcd = geometry.PointCloud().to(device=TensorPointCloud.device)

    def get_bounding_box(self) -> geometry_legacy.AxisAlignedBoundingBox:
        return self.pcd.get_axis_aligned_bounding_box().to_legacy()

    def merge_with(self, geom2: Self) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        if geom2.pcd.device != self.pcd.device:
            logger.warning(f"Merging point clouds on different devices ({self.pcd.device} vs {geom2.pcd.device}).")
        if geom2.pcd.is_empty():
            return self
        if self.pcd.is_empty():
            self.pcd = geom2.pcd
        else:
            self.pcd += geom2.pcd
        return self

    def to_occupancy_grid(
        self, resolution: float = 0.1, occupied_height_bounds: tuple[float, float] = (0.0, 10.0)
    ) -> Optional[OccupancyGrid]:
        if self.pcd.is_empty():
            return None

        points: torch.Tensor = torch_dlpack.from_dlpack(self.pcd.point.positions.to_dlpack())
        min_xy = torch.amin(points[:, :2], dim=0)
        max_xy = torch.amax(points[:, :2], dim=0)
        origin = torch.floor(min_xy / resolution) * resolution
        grid_size = torch.ceil((max_xy - origin) / resolution).to(torch.int) + 1

        grid = torch.full(grid_size.tolist(), -1, dtype=torch.int8)

        occupied_mask = (points[:, 2] >= occupied_height_bounds[0]) & (points[:, 2] <= occupied_height_bounds[1])
        indices = ((points[:, :2] - origin) / resolution).to(torch.int)
        grid[indices[~occupied_mask, 0], indices[~occupied_mask, 1]] = 0
        grid[indices[occupied_mask, 0], indices[occupied_mask, 1]] = 1
        return OccupancyGrid(origin=origin.cpu().numpy(), resolution=resolution, grid=grid.cpu().numpy())

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
        occlusion_margin: float,
        depth_image: Optional[Float[torch.Tensor, "1 H W"]] = None,
        device: str = "cuda",
        preallocated_object_mask_count: int = 10,
        expected_visibility_threshold: float = 0.1,
    ) -> tuple[list[int], list[float], UInt8[torch.Tensor, "N H W"]]:
        number_expected_objects = 0
        expected_object_indices: list[int] = []
        object_visibility_ratios: list[float] = []
        if len(geometries) == 0:
            return (
                expected_object_indices,
                object_visibility_ratios,
                torch.zeros((0, img_height, img_width), dtype=torch.uint8, device=device),
            )

        object_projections = torch.zeros(
            (preallocated_object_mask_count, img_height, img_width), dtype=torch.uint8, device=device
        )
        if ("cuda" in device) != (TensorPointCloud.device.get_type() == o3c.Device.DeviceType.CUDA):
            logger.warning(
                "Projecting TensorPointClouds to camera on CUDA device while TensorPointClouds are not on CUDA device."
            )

        camera_position = torch.tensor(camera_pose[:3, -1], device=device)
        camera_intrinsics = o3c.Tensor(camera_intrinsics_torch.cpu().numpy())
        camera_extrinsics = o3c.Tensor(camera_pose.inverse().cpu().numpy())
        for idx, obj in enumerate(geometries):
            # skip objects which are too far away
            if len(obj) == 0:
                continue

            if torch.norm(obj.get_center(device=device) - camera_position) > max_depth * 2:
                continue

            depth_projection: o3c.Tensor = obj.pcd.project_to_depth_image(
                img_width,
                img_height,
                camera_intrinsics,
                camera_extrinsics,
                depth_max=max_depth,
            ).as_tensor()
            depth_projection_torch: torch.Tensor = torch_dlpack.from_dlpack(depth_projection.to_dlpack())

            valid_depth_mask = depth_projection_torch > 0
            if depth_image is None:
                visible_depth_mask = valid_depth_mask
            else:
                depth_projection_torch /= 1000.0  # convert to meters
                depth_image = depth_image.to(depth_projection_torch.device)
                visible_depth_mask = cls._erase_occluded_region(
                    depth_image=depth_image,
                    depth_projection_torch=depth_projection_torch,
                    valid_depth_mask=valid_depth_mask,
                    occlusion_margin=occlusion_margin,
                )
            num_expected_points = visible_depth_mask.count_nonzero().item()

            visibility_ratio = float(num_expected_points) / len(obj) if len(obj) > 0 else 0.0
            if visibility_ratio < expected_visibility_threshold:
                continue

            if number_expected_objects >= len(object_projections):
                object_projections = torch.cat(
                    [object_projections, torch.zeros_like(object_projections, device=device)], dim=0
                )
            object_projections[number_expected_objects, ...] = visible_depth_mask.to(torch.uint8).squeeze()
            expected_object_indices.append(idx)
            object_visibility_ratios.append(visibility_ratio)
            number_expected_objects += 1

        object_projections = object_projections[:number_expected_objects, ...]

        return expected_object_indices, object_visibility_ratios, object_projections

    def to_serializable(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self.pcd.is_empty():
            return (np.empty((0, 3), dtype=np.float32), None)
        color_arr = self.pcd.point.colors.numpy() if "colors" in self.pcd.point else None
        return (self.pcd.point.positions.numpy(), color_arr)

    @classmethod
    def from_serializable(cls, data: tuple[np.ndarray, Optional[np.ndarray]]) -> Self:
        pc = cls()
        if data[1] is None:
            pc.pcd = geometry.PointCloud({"positions": data[0].astype(np.float32)}).to(device=TensorPointCloud.device)
        else:
            pc.pcd = geometry.PointCloud(
                {"positions": data[0].astype(np.float32), "colors": data[1].astype(np.float32)}
            ).to(device=TensorPointCloud.device)
        logger.debug(f"Loaded TensorPointCloud with {len(pc)} points from serializable data")
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

        pcd = geometry.PointCloud().to(device=TensorPointCloud.device)
        pcd.point.positions = o3c.Tensor(downsampled_points.cpu().numpy(), dtype=o3c.float32, device=pcd.device)
        if downsampled_colors is not None:
            pcd.point.colors = o3c.Tensor(
                (downsampled_colors.to(torch.float32) / 255.0).cpu().numpy(), dtype=o3c.float32, device=pcd.device
            )
        if camera_pose is not None:
            pcd.transform(o3c.Tensor(camera_pose.cpu().numpy()))
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
        if other.pcd.is_empty() or self.pcd.is_empty():
            return 0.0
        f_score = float(
            other.pcd.compute_metrics(
                self.pcd,
                metrics=[geometry.Metric.FScore],
                params=geometry.MetricParameters(fscore_radius=[downsample_voxel_size]),
            ).numpy()[0]
            / 100.0
        )
        num_points_a = len(self)
        num_points_b = len(other)
        num_overlapping_points = f_score * (num_points_a + num_points_b) / 2.0
        overlap = num_overlapping_points / min(num_points_b, num_points_a * visibility_ratio)
        return overlap

    def register_geometries(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, target_geom: Self, init: Float[np.ndarray, "4 4"], max_correspondence_distance: float, max_iterations: int
    ) -> registration_legacy.RegistrationResult:
        results = registration.icp(
            self.pcd,
            target_geom.pcd,
            max_correspondence_distance,
            init_source_to_target=o3c.Tensor(init),
            estimation_method=registration.TransformationEstimationPointToPoint(),
            criteria=registration.ICPConvergenceCriteria(max_iteration=max_iterations),
        )
        legacy_results = registration_legacy.RegistrationResult()
        # legacy_results.correspondence_set = results.correspondence_set.numpy()
        legacy_results.fitness = results.fitness
        legacy_results.inlier_rmse = results.inlier_rmse
        legacy_results.transformation = results.transformation.numpy()
        return legacy_results

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
        if self.pcd.is_empty():
            return 0
        else:
            return len(self.pcd.point.positions)

    def get_center(self, device="cpu") -> Float[torch.Tensor, "3"]:
        if self.pcd.is_empty():
            return torch.tensor([torch.nan, torch.nan, torch.nan], dtype=torch.float32, device=device)
        return torch_dlpack.from_dlpack(self.pcd.get_center().to_dlpack()).to(device=device)

    def numpy_points(self) -> Float[np.ndarray, "N 3"]:
        """Return the points of the geometry as a (N, 3) numpy array."""
        if self.pcd.is_empty():
            return np.empty((0, 3), dtype=np.float32)
        return self.pcd.point.positions.cpu().numpy()

    def numpy_colors(self) -> Float[np.ndarray, "N 3"]:
        """Return the colors of the geometry as a (N, 3) numpy array."""
        if self.pcd.is_empty():
            return np.empty((0, 3), dtype=np.float32)
        if "colors" in self.pcd.point:
            return self.pcd.point.colors.cpu().numpy()
        else:
            return np.zeros((len(self), 3), dtype=np.float32)

    def crop(self, bounding_box: geometry_legacy.AxisAlignedBoundingBox) -> None:
        self.pcd = self.pcd.crop(geometry.AxisAlignedBoundingBox.from_legacy(bounding_box, device=self.pcd.device))

    def _erase_occluded_region(
        depth_image: Float[torch.Tensor, "1 H W"],
        depth_projection_torch: Float[torch.Tensor, "H W 1"],
        valid_depth_mask: Bool[torch.Tensor, "H W 1"],
        occlusion_margin: float,
    ) -> UInt8[torch.Tensor, "H W 1"]:
        """Erase occluded regions in a objectdepth projection.

        Args:
            depth_image (Float[torch.Tensor, "1 H W"]): The depth image as a (1, H, W) tensor.
            depth_projection_torch (Float[torch.Tensor, "H W 1"]): The depth projection as a torch tensor.
            valid_depth_mask (Bool[torch.Tensor, "H W 1"]): A boolean mask indicating which pixels in the depth projection are valid (i.e., have a depth value greater than 0).
            occlusion_margin (float): Margin in meters to consider for occlusion.

        Returns:
            torch.Tensor: The occlusion-aware mask as a (H, W, 1) tensor.

        """
        device = depth_image.device
        img_height, img_width = depth_projection_torch.shape[:2]

        # depth_image_aligned: (1, 640, 360), depth_projection_torch: (640, 360, 1)
        depth_image = depth_image.reshape((img_height, img_width, 1)).to(device)

        # A pixel is considered visible if the depth projection is less than the depth image (with margin) and the depth projection is valid
        visible_depth_mask = torch.where(
            torch.logical_or(depth_image == 0, depth_projection_torch < depth_image + occlusion_margin),
            valid_depth_mask,
            torch.zeros_like(depth_image),
        )

        return visible_depth_mask.to(torch.uint8)
