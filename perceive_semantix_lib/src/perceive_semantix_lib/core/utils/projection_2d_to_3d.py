from logging import getLogger
from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, UInt8, jaxtyped
from supervision.detection.core import Detections
from typeguard import typechecked as typechecker

from perceive_semantix_lib.core.geometry import GeometryType

logger = getLogger(__name__)


@jaxtyped(typechecker=typechecker)
def detections_to_3d_pointcloud_and_bounding_boxes(
    geometry_type: type[GeometryType],
    detections: Detections,
    depth_image: Float[torch.Tensor, "1 1 H W"],
    camera_pose: Float[torch.Tensor, "4 4"],
    camera_intrinsics: Float[torch.Tensor, "3 3"],
    rgb_image: Optional[UInt8[torch.Tensor, "1 3 H W"]] = None,
    device: str = "cuda",
    min_points_threshold: int = 16,
    obj_pcd_max_points: int = 5000,
    min_bbox_volume: float = 1e-6,
    min_max_depth: tuple[float, float] = (0.1, 5.0),
    background_mask_erosion_radius: int = 3,
) -> tuple[Detections, GeometryType]:
    """Convert 2D detections to 3D point clouds and bounding boxes.

    Args:
        geometry_type (type[GeometryType]): The geometry type to use.
        detections (Detections): 2D detections with masks.
        depth_image (Float[torch.Tensor, "1 1 H W"]): Depth image tensor.
        camera_pose (Float[torch.Tensor, "4 4"]): Camera pose matrix.
        camera_intrinsics (Float[torch.Tensor, "3 3"]): Camera intrinsic matrix.
        rgb_image (Optional[UInt8[torch.Tensor, "1 3 H W"]], optional): RGB image tensor. Defaults to None.
        device (str, optional): Device for computation. Defaults to 'cuda'.
        min_points_threshold (int, optional): Minimum points for valid object. Defaults to 16.
        obj_pcd_max_points (int, optional): Max points per object point cloud. Larger objects are downsampled. Defaults to 5000.
        min_bbox_volume (float, optional): Minimum bounding box volume. Defaults to 1e-6.
        min_max_depth (tuple[float, float], optional): Minimum and maximum depth range to consider points. Defaults to (0.1, 5.0).
        background_mask_erosion_radius (int, optional): Radius for background mask erosion. Defaults to 3.

    Returns:
        Detections: Updated detections with .data['point_clouds'] and .data['bounding_boxes'].
        o3d.geometry.PointCloud: Background point cloud.

    """
    if len(detections) == 0 or detections.mask is None:
        masks_tensor = torch.zeros(
            (0, depth_image.shape[2], depth_image.shape[3]),
            dtype=torch.bool,
            device=device,
        )
    else:
        masks_tensor = torch.from_numpy(detections.mask).to(device)
    background_mask = F.max_pool2d(
        masks_tensor.sum(dim=0).clamp(0, 1).unsqueeze(0).to(torch.float32),
        kernel_size=background_mask_erosion_radius * 2 + 1,
        stride=1,
        padding=background_mask_erosion_radius,
    ).logical_not()

    depth_image = depth_image.to(device)
    cam_K_tensor = camera_intrinsics.to(device)

    if rgb_image is not None:
        image_rgb_tensor = rgb_image.permute(0, 2, 3, 1).to(device)
    else:
        image_rgb_tensor = None

    detection_points, detections_colors, bg_points, bg_colors = batch_mask_depth_to_points_colors(
        depth_image.squeeze(0),
        masks_tensor,
        background_mask,
        cam_K_tensor,
        min_max_depth=min_max_depth,
        image_rgb_tensor=image_rgb_tensor,
    )

    point_clouds = []
    bounding_boxes = []
    valid_detection_indices = []
    for i, (points, colors) in enumerate(zip(detection_points, detections_colors)):
        pcd = geometry_type.from_point_tensor(
            points,
            colors,
            min_points_threshold=min_points_threshold,
            camera_pose=camera_pose,
            obj_pcd_max_points=obj_pcd_max_points,
        )
        if pcd is None:
            continue
        bbox = pcd.get_bounding_box()
        if bbox.volume() < min_bbox_volume:
            continue

        point_clouds.append(pcd)
        bounding_boxes.append(bbox)
        valid_detection_indices.append(i)

    detections = detections[valid_detection_indices]  # type: ignore
    detections.data["point_clouds"] = point_clouds
    detections.data["bounding_boxes"] = bounding_boxes
    background_pcd = geometry_type.from_point_tensor(
        bg_points,
        bg_colors,
        min_points_threshold=-1,
        camera_pose=camera_pose,
        obj_pcd_max_points=obj_pcd_max_points,
    )
    if background_pcd is None:
        background_pcd = geometry_type()

    return detections, background_pcd


@jaxtyped(typechecker=typechecker)
def batch_mask_depth_to_points_colors(
    depth_tensor: Float[torch.Tensor, "1 H W"],
    masks_tensor: Bool[torch.Tensor, "N H W"],
    background_mask: Bool[torch.Tensor, "1 H W"],
    cam_K: Float[torch.Tensor, "3 3"],
    min_max_depth: tuple[float, float],
    image_rgb_tensor: Optional[UInt8[torch.Tensor, "1 H W 3"]] = None,
    device: str = "cuda",
) -> tuple[
    Float[torch.Tensor, "N H W 3"],
    UInt8[torch.Tensor, "N H W 3"],
    Float[torch.Tensor, "H W 3"],
    UInt8[torch.Tensor, "H W 3"],
]:
    """Convert a batch of masked depth image to 3D points and corresponding colors.

    Args:
        depth_tensor (torch.Tensor): A tensor of shape (1, H, W) representing the depth image.
        masks_tensor (torch.Tensor): A tensor of shape (N, H, W) representing the masks for each depth image.
        background_mask (torch.Tensor): A tensor of shape (1, H, W) representing the background mask.
        cam_K (torch.Tensor): A tensor of shape (3, 3) representing the camera intrinsic matrix.
        min_max_depth (tuple): A tuple (min_depth, max_depth) specifying the valid depth range.
        image_rgb_tensor (torch.Tensor, optional): A tensor of shape (1, H, W, 3) representing the RGB image. Defaults to None.
        device (str, optional): The device to perform the computation on. Defaults to 'cuda'.

    Returns:
        tuple: A tuple containing:
            - points (torch.Tensor): A tensor of shape (N, H, W, 3) representing the 3D points for each mask.
            - colors (torch.Tensor): A tensor of shape (N, H, W, 3) representing the colors for each mask.
            - bg_points (torch.Tensor): A tensor of shape (H, W, 3) representing the background 3D points.
            - bg_colors (torch.Tensor): A tensor of shape (H, W, 3) representing the background colors.

    """
    # Insert background mask at front of batch
    masks_tensor = torch.cat([background_mask, masks_tensor], dim=0).to(torch.float32)
    N, H, W = masks_tensor.shape

    # Generate grid of pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(0, H, device=device),
        torch.arange(0, W, device=device),
        indexing="ij",
    )
    z = depth_tensor.repeat(N, 1, 1) * masks_tensor  # Apply masks to depth

    # Points which are (1) contained in the mask and (2) have non-zero depth
    valid_and_masked = (z != 0).float()
    fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    points = torch.stack((x, y, z), dim=-1) * valid_and_masked.unsqueeze(-1)  # Shape: (N, H, W, 3)

    in_range_mask = (depth_tensor >= min_max_depth[0]) & (depth_tensor <= min_max_depth[1])
    # Non-background points must be in valid depth range
    points[1:, :, :, :] *= in_range_mask.unsqueeze(-1)

    # Build tensor of point colors (N, H, W, 3)
    valid = (points[:, :, :, 2] != 0).unsqueeze(-1)
    if image_rgb_tensor is not None:
        repeated_rgb = image_rgb_tensor.repeat(N, 1, 1, 1) * masks_tensor.unsqueeze(-1)
        colors = repeated_rgb * valid  # Shape: (N, H, W, 3)
    else:
        logger.warning("No RGB image provided, assigning random colors to objects")
        random_colors = torch.randint(0, 256, (N, 3), device=device, dtype=torch.uint8)
        colors = random_colors.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1) * valid
    colors = colors.to(torch.uint8)

    # Separate background from foreground objects
    return points[1:, ...], colors[1:, ...], points[0, ...], colors[0, ...]
