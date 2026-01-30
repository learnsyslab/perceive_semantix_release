from typing import Any, Optional, Self

import numpy as np
import torch
from jaxtyping import Float, UInt8
from open3d.geometry import AxisAlignedBoundingBox  # pyright: ignore[reportMissingImports]
from open3d.pipelines import registration  # type: ignore

from perceive_semantix_lib.core.occupancy_grid import OccupancyGrid


class GeometryBase:
    def __init__(self):
        pass

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get axis aligned bounding box of this geometry."""
        raise NotImplementedError

    def merge_with(self, geom2: Self) -> Self:
        """Merge another geometry of the same type into this object (inplace as well as return the result).

        Args:
            geom2 (Self): The geometry to merge with.

        """
        raise NotImplementedError

    def __iadd__(self, other: Self) -> Self:
        """Merge ``self`` with other geometry, modifying ``self`` in place."""
        self.merge_with(other)
        return self

    def to_occupancy_grid(
        self, resolution: float, occupied_height_bounds: tuple[float, float]
    ) -> Optional[OccupancyGrid]:
        """Generate occupancy grid from the geometry.

        Args:
            resolution (float): The resolution of the resulting grid (in m).
            occupied_height_bounds (tuple[float, float]): The height bounds for occupied space (min, max).

        Returns:
            grid (Optional[OccupancyGrid]): Is ``None`` if the object contains no geometry (e.g. no points).

        """
        raise NotImplementedError

    @classmethod
    def batch_project_to_camera(
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
        """Project a batch of geometries of the same type onto a camera plane.

        Args:
            geometries (list[GeometryBase, N]): List GeometryBase (or sub classes) instances.
            camera_pose (Float[torch.Tensor, "4 4"]): Camera pose matrix.
            camera_intrinsics_torch (Float[torch.Tensor, "3 3"]): Camera intrinsic matrix.
            img_height (int): Image height to project on (in pixels).
            img_width (int): Image width to project on (in pixels).
            min_depth (float): Minimum depth to consider (in m).
            max_depth (float): Maximum depth to consider (in m).
            device (str): Device to use for computation.
            preallocated_object_mask_count (int): How many object mask arrays should be preallocated. This should be slighly larger than the average number of expected objects in each frame.
            expected_visibility_threshold (float): Minimum visibility ratio for an object to be considered visible.

        Returns:
            expected_object_indices (list[int]): Indices of the M geometries which are expected to be visible in the camera view.
            object_visibility_ratio (list[float, M]): List visibility ratios for each expected object.
                The visibility ratio is the portion of the object's geometry which is visible on the camera plane.
            object_projections (Float[torch.Tensor, "M H W"]): Binary masks of shape (M, H, W) for each expected object.

        """
        raise NotImplementedError

    def to_serializable(self) -> Any:
        """Convert geometry to a type which is picklable."""
        raise NotImplementedError

    @classmethod
    def from_serializable(cls, data: Any) -> Self:
        """Create geometry object from data."""
        raise NotImplementedError

    @classmethod
    def from_point_tensor(
        cls,
        points: Float[torch.Tensor, "H W 3"],
        colors: UInt8[torch.Tensor, "H W 3"],
        min_points_threshold: Optional[int] = None,
        camera_pose: Optional[Float[torch.Tensor, "4 4"]] = None,
        obj_pcd_max_points: int = -1,
    ) -> Optional[Self]:
        """Convert a point tensor to an 3D Geometry.

        Only points with z > 0 are considered valid.

        Args:
            points (Float[torch.Tensor, "H W 3"]): Tensor of 3D points. The last dimension contains each point as (x, y, z).
            colors (UInt8[torch.Tensor, "H W 3"]): Tensor of colors corresponding to the points. The last dimension contains each color as (B, G, R).
            min_points_threshold (Optional[int], optional): Minimum number of valid points required to create a point cloud. Only points with z > 0 are considered valid.
                                                            If None, no threshold is applied. Defaults to None.
            camera_pose (Optional[Float[torch.Tensor, "4 4"]], optional): Camera pose to transform the point cloud. Defaults to None.
            obj_pcd_max_points (int, optional): Maximum number of points in the output point cloud. Larger point clouds are downsampled.
                                            If -1, no downsampling is applied. Defaults to -1.

        Returns:
            geometry (Optional[GeometryBase]): The resulting 3D Geometry, or None if the number of valid points is below the threshold.

        """
        raise NotImplementedError

    def postprocess_geometry(self, voxel_downsample_size: float) -> None:
        """Apply common postprocessing like downsampling.

        Args:
            voxel_downsample_size (float): Minimum distance between points after downsampling.

        """
        raise NotImplementedError

    def compute_overlap(self, other: Self, downsample_voxel_size: float, visibility_ratio: float = 1) -> float:
        """Compute overlap between two geometries of the same type.

        See sub-class documentation for details.

        Args:
            other (GeometryBase): The geometry to compute the overlap to.
            downsample_voxel_size (float): The voxel size to use for downsampling.
            visibility_ratio (float): The visibility ratio to use for ``self``. This describes how much (in %) of the object is in view of the camera.

        Returns:
            overlap (float): Overlap metric between 0 (no overlap) and 1 (fully overlapping).

        """
        raise NotImplementedError

    def register_geometries(
        self, target_geom: Self, init: Float[np.ndarray, "4 4"], max_correspondence_distance: float, max_iterations: int
    ) -> registration.RegistrationResult:
        """Register (align) ``self`` with ``target_geom``.

        Args:
            target_geom (GeometryBase): Geometry to align with.
            init (Float[np.ndarray, "4 4"]): Transform with which to initialize the registration process.
            max_correspondence_distance (float): Maximum correspondence distance (see open3d icp implementation).
            max_iterations (int): Maximum number of iterations.

        Returns:
            results (open3d.pipelines.registration.RegistrationResult): Result of the registration.

        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of points/voxels/... in the geometry."""
        raise NotImplementedError

    def get_center(self, device: str = "cpu") -> Float[torch.Tensor, "3"]:
        """Return the center of the geometry as a (3,) tensor."""
        raise NotImplementedError

    def numpy_points(self) -> Float[np.ndarray, "N 3"]:
        """Return the points of the geometry as a (N, 3) numpy array."""
        raise NotImplementedError

    def numpy_colors(self) -> Float[np.ndarray, "N 3"]:
        """Return the colors of the geometry as a (N, 3) numpy array."""
        raise NotImplementedError

    def crop(self, bounding_box: AxisAlignedBoundingBox) -> None:
        """Crop the geometry to the given axis aligned bounding box. Modifies the geometry in place.

        Args:
            bounding_box (AxisAlignedBoundingBox): The bounding box to crop to.

        """
        raise NotImplementedError
