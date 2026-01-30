from logging import getLogger
from pathlib import Path
from typing import Callable, Generic, Optional

import cv2
import numpy as np
import open_clip
import torch
import torchvision
from jaxtyping import Bool, Float, UInt8, jaxtyped
from PIL import Image as PILImage
from PIL.Image import Image
from supervision.detection.core import Detections
from typeguard import typechecked as typechecker
from ultralytics.models import SAM, YOLO

from perceive_semantix_lib.core.config import DetectionConfig
from perceive_semantix_lib.core.geometry import GeometryType
from perceive_semantix_lib.core.utils.object_classes import get_global_object_classes
from perceive_semantix_lib.core.utils.projection_2d_to_3d import detections_to_3d_pointcloud_and_bounding_boxes

logger = getLogger(__name__)


class ObjectDetector(Generic[GeometryType]):
    def __init__(
        self, config: DetectionConfig, device: str, geometry_type: type[GeometryType], weight_dir: Optional[Path] = None
    ) -> None:
        self.config = config
        self.device = device
        self.geometry_type = geometry_type
        if "cuda" in self.device and not torch.cuda.is_available():
            logger.error("CUDA requested but not available.")
        if self.config.object_class_names_file is None:
            logger.error("Object class names file must be provided in the config.")
            raise RuntimeError("Object class names file must be provided in the config.")

        if weight_dir is not None:
            weight_dir.mkdir(parents=True, exist_ok=True)

        self.detection_model: YOLO = ObjectDetector.load_ultralytics_model(config.yolo_model_name, weight_dir, YOLO)  # type: ignore
        self.segmentation_model: SAM = ObjectDetector.load_ultralytics_model(config.sam_model_name, weight_dir, SAM)  # type: ignore
        self.clip_model, self.clip_preprocess, self.clip_tokenizer = ObjectDetector.load_clip(
            config.clip_model_name, config.clip_pretrained, weight_dir
        )
        self.detection_model.to(self.device)
        self.segmentation_model.to(self.device)
        self.clip_model.to(self.device)

        self.object_classes = get_global_object_classes()
        self.detection_model.set_classes(self.object_classes.get_classes_arr())  # type: ignore

    @staticmethod
    def load_ultralytics_model(model_name: str, cache_dir: Optional[Path], model_constructor: Callable) -> YOLO | SAM:
        """Load a Ultralytics model (YOLO or SAM) with weights cached in a custom directory."""
        if cache_dir is None:
            model = model_constructor(model_name)
            logger.info(f"Loaded Ultralytics model {model_name}")
            return model

        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / model_name
        model_path = model_path.with_name(f"{model_path.name}.pt")
        model = model_constructor(model_path)
        logger.info(f"Loaded Ultralytics model {model_name} with weights from {model_path.resolve()}")
        return model

    @staticmethod
    def load_clip(
        model_name: str, pretrained: str, cache_dir: Optional[Path]
    ) -> tuple[torch.nn.Module, torchvision.transforms.Compose, Callable]:
        """Load a CLIP model and tokenizer with weights cached in a custom directory."""
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            cache_dir=str(cache_dir.resolve()) if cache_dir is not None else None,
            weights_only=True,
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        logger.info(
            f"Loaded CLIP model {model_name}/{pretrained} with weights from {cache_dir.resolve() if cache_dir is not None else '<default location>'}"
        )
        return model, preprocess, tokenizer  # type: ignore

    def post_process_3d_detections(self, detections: Detections, voxel_downsample_size: float) -> Detections:
        """Downsample the point clouds in the detections."""
        if detections.data is not None and "point_clouds" in detections.data:
            for pcd in detections.data["point_clouds"]:
                pcd.postprocess_geometry(voxel_downsample_size)
        return detections

    @jaxtyped(typechecker=typechecker)
    def detect_3d(
        self,
        rgb_image: UInt8[torch.Tensor, "1 3 H W"],
        depth_image: Float[torch.Tensor, "1 1 H W"],
        camera_pose: Float[torch.Tensor, "4 4"],
        camera_intrinsics: Float[torch.Tensor, "3 3"],
    ) -> tuple[Detections, Float[torch.Tensor, "N clip"], GeometryType]:
        """Detect objects in 3D from RGB-D images.

        Args:
            rgb_image (UInt8[torch.Tensor, "1 3 H W"]): Input color image.
            depth_image (Float[torch.Tensor, "1 1 H W"]): Input depth image.
            camera_pose (Float[torch.Tensor, "4 4"]): Camera pose matrix (homogenous transform matrix).
            camera_intrinsics (Float[torch.Tensor, "3 3"]): Camera intrinsics matrix K (fx 0 cx; 0 fy cy; 0 0 1).

        Returns:
            Detections: A detections objects (containing bounding boxes, masks, class IDs).
            Float[torch.Tensor, "N clip"]: CLIP features for each detected object. The clip features are separated because they reside on the GPU.
            PointCloud: The background point cloud (i.e., points not belonging to any detected object).

        """
        pil_image = PILImage.fromarray(rgb_image.squeeze(0).permute(1, 2, 0).cpu().numpy())
        detections = self._detect_2d(pil_image, mask_erosion_radius=self.config.mask_erosion_radius)
        detections.data = {"classes": [self.object_classes.get_classes_arr()[i] for i in detections.class_id]}  # type: ignore
        detections = self._postprocess_2d_detections(detections, rgb_image)

        detections, background_pc = detections_to_3d_pointcloud_and_bounding_boxes(
            self.geometry_type,
            detections,
            depth_image,
            camera_pose,
            camera_intrinsics,
            rgb_image,
            min_points_threshold=self.config.min_points_threshold,
            obj_pcd_max_points=self.config.obj_pcd_max_points,
            min_bbox_volume=self.config.min_bbox_volume,
            min_max_depth=(self.config.min_depth_m, self.config.max_depth_m),
            background_mask_erosion_radius=2
            * self.config.mask_erosion_radius,  # double to account for earlier erosion of object masks
        )
        clip_features, cropped_images = self._compute_clip_features(pil_image, detections)

        return detections, clip_features, background_pc

    def _is_too_blurry(self, pil_image: Image) -> bool:
        blur_score = cv2.Laplacian(np.array(pil_image), cv2.CV_64F).var()
        if blur_score < self.config.blur_threshold:
            return True
        return False

    def _detect_2d(self, pil_image: Image, mask_erosion_radius: int = 3) -> Detections:
        """Run 2D object detection and segmentation on the input image.

        Args:
            pil_image (Image): Input color image.
            mask_erosion_radius (int, optional): Radius for mask erosion (shrinking the masks slightly to avoid artifacts at the edges). Defaults to 3.

        Returns:
            Detections: A detections objects containing bounding boxes, masks, class IDs.

        """
        if self._is_too_blurry(pil_image):
            logger.warning("Image is too blurry, skipping detection.")
            return Detections.empty()

        logger.info("Running object detection and segmentation.")
        detection_result = self.detection_model.predict(
            pil_image, device=self.device, conf=self.config.object_min_detection_confidence, save=False, verbose=False
        )
        if len(detection_result) == 0 or len(detection_result[0]) == 0 or detection_result[0].boxes is None:
            logger.info("No objects detected.")
            return Detections.empty()
        detections_keep = self._remove_duplicate_bboxes(detection_result[0].boxes.xyxy)  # type: ignore
        if (num_kept := sum(detections_keep)) != len(detection_result[0]):
            logger.info(
                f"Removed {len(detection_result[0]) - num_kept}/{len(detection_result[0])} detections due to duplicate bounding boxes."
            )
        detection_result = [detection_result[0][detections_keep]]

        segmentation_results = self.segmentation_model.predict(
            pil_image,
            device=self.device,
            bboxes=detection_result[0].boxes.xyxy.clone().detach(),  # type: ignore
            verbose=False,  # type: ignore
        )  # attention: modifes xyxy_tensor in-place
        if len(segmentation_results) == 0 or len(segmentation_results[0]) == 0 or segmentation_results[0].masks is None:
            logger.error("Segmentation model did not return any masks.")
            return Detections.empty()

        masks_tensor: Bool[torch.Tensor, "N H W"] = segmentation_results[0].masks.data  # type: ignore
        detections_keep = self._remove_duplicate_masks(masks_tensor)
        if (num_kept := sum(detections_keep)) != len(masks_tensor):
            logger.info(
                f"Removed {len(masks_tensor) - num_kept}/{len(masks_tensor)} detections due to duplicate masks."
            )

        # Erode masks to avoid artifacts at the edges
        if sum(detections_keep) > 0:
            masks_tensor = torch.nn.functional.max_pool2d(
                -masks_tensor[detections_keep].to(torch.float32),
                kernel_size=2 * mask_erosion_radius + 1,
                stride=1,
                padding=mask_erosion_radius,
            ).to(torch.bool)
        else:
            masks_tensor = masks_tensor[detections_keep]

        detections = Detections(
            xyxy=detection_result[0].boxes.xyxy[detections_keep].cpu().numpy(),  # type: ignore
            confidence=detection_result[0].boxes.conf[detections_keep].cpu().numpy(),  # type: ignore
            class_id=detection_result[0].boxes.cls[detections_keep].cpu().numpy().astype(int),  # type: ignore
            mask=masks_tensor.cpu().numpy(),
        )

        return detections

    def _remove_duplicate_bboxes(self, xyxy: Float[torch.Tensor, "N 4"], maximum_iou: float = 0.95) -> list[bool]:
        """Remove duplicate bounding boxes based on IoU.

        Args:
            xyxy (Float[torch.Tensor, "N 4"]): Bounding boxes in (x1, y1, x2, y2) format.
            maximum_iou (float): Maximum IoU to consider a box as duplicate. Defaults to 0.9.

        Returns:
            list[bool]: A list of booleans indicating which boxes to keep.

        """
        N = xyxy.shape[0]
        if N == 0:
            return []

        intersection_top_left = torch.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # (N, N, 2)
        intersection_bottom_right = torch.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # (N, N, 2)
        intersection_sizes = torch.clamp(intersection_bottom_right - intersection_top_left, min=0)  # (N, N, 2)
        intersection_areas = intersection_sizes[:, :, 0] * intersection_sizes[:, :, 1]  # (N, N)
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])  # (N,)
        union_areas = areas[:, None] + areas[None, :] - intersection_areas  # (N, N)
        iou = intersection_areas / union_areas  # (N, N)

        to_remove = set()
        for i in range(N):
            if i in to_remove:
                continue
            for j in range(i + 1, N):
                if j in to_remove:
                    continue
                if iou[i, j] > maximum_iou:
                    to_remove.add(j)

        return [i not in to_remove for i in range(N)]

    def _remove_duplicate_masks(self, masks: Bool[torch.Tensor, "N H W"], maximum_iou: float = 0.9) -> list[bool]:
        """Remove duplicate masks based on IoU.

        Args:
            masks (Bool[torch.Tensor, "N H W"]): Binary masks.
            maximum_iou (float): Maximum IoU to consider a mask as duplicate. Defaults to 0.9.

        Returns:
            list[bool]: A list of booleans indicating which masks to keep.

        """
        N = masks.shape[0]
        if N == 0:
            return []

        # Compute mask IoU
        masks_flat = masks.view(N, -1).float()  # (N, H*W)
        intersection_masks = torch.matmul(masks_flat, masks_flat.t())  # (N, N)
        areas_masks = masks_flat.sum(dim=1)  # (N,)
        union_masks = areas_masks[:, None] + areas_masks[None, :] - intersection_masks  # (N, N)
        iou_masks = intersection_masks / union_masks  # (N, N)

        to_remove = set()
        for i in range(N):
            if i in to_remove:
                continue
            for j in range(i + 1, N):
                if j in to_remove:
                    continue
                if iou_masks[i, j] > maximum_iou:
                    to_remove.add(j)

        return [i not in to_remove for i in range(N)]

    def _compute_clip_features(
        self, image: Image, detections: Detections, padding: int = 20
    ) -> tuple[Float[torch.Tensor, "N clip"], list[Image]]:
        """Compute CLIP feature for each detected object: Crop the input image around each detection bounding box, with some padding, and compute the CLIP feature for each cropped image."""
        if len(detections) == 0:
            return torch.empty((0, 0)), []
        image_crops = []
        preprocessed_images = []

        # Prepare data for batch processing
        # TODO: Do this without a loop
        for _, (x_min, y_min, x_max, y_max) in enumerate(detections.xyxy):
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image.width - x_max)
            bottom_padding = min(padding, image.height - y_max)

            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding

            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            preprocessed_image = self.clip_preprocess(cropped_image).unsqueeze(0)  # type: ignore
            preprocessed_images.append(preprocessed_image)
            image_crops.append(cropped_image)

        # Convert lists to batches
        preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(preprocessed_images_batch)  # type: ignore
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features, image_crops

    def _postprocess_2d_detections(self, detections: Detections, image: UInt8[torch.Tensor, "1 3 H W"]) -> Detections:
        """Post-process 2D detections to remove unwanted classes, etc."""
        detections = ObjectDetector._resize_masks_bounding_boxes(detections, image)
        detections = ObjectDetector._filter_2d_detections(
            detections,
            image,
            self.object_classes.skip_bg,
            self.object_classes.bg_classes,
            self.config.mask_area_threshold,
            self.config.max_bbox_area_ratio,
            self.config.object_min_detection_confidence,
        )
        if detections.mask is not None and detections.xyxy is not None:
            detections.mask = ObjectDetector._masks_subtract_contained(detections.xyxy, detections.mask)
        return detections

    @staticmethod
    def _resize_masks_bounding_boxes(detections: Detections, image: UInt8[torch.Tensor, "1 3 H W"]) -> Detections:
        """Make sure masks and have the the same width and height as the input image.

        If necessary, resize the masks and adjust the bounding boxes accordingly.
        """
        if detections.mask is None or detections.xyxy is None:
            return detections

        if detections.mask.shape[1:] == image.shape[2:]:
            return detections

        logger.error("Mask resizing not implemented yet.")
        raise NotImplementedError("Mask resizing not implemented yet.")

    @staticmethod
    def _filter_2d_detections(
        detections: Detections,
        image: UInt8[torch.Tensor, "1 3 H W"],
        skip_bg: bool = False,
        bg_classes: list = [],
        mask_area_threshold: float = 10,
        max_bbox_area_ratio: Optional[float] = None,
        mask_conf_threshold: Optional[float] = None,
    ) -> Detections:
        """Filter out unwanted detections based on various criteria.

        Args:
            detections (Detections): Detections object containing masks, bounding boxes, class IDs, and confidence scores.
            image (UInt8[torch.Tensor, "1 3 H W"]): Input color image.
            skip_bg (bool, optional): Whether to skip background classes. Defaults to False.
            bg_classes (list, optional): List of background class names. Defaults to [].
            mask_area_threshold (int, optional): Minimum area (in pixels) for a mask to be kept. Defaults to 10. Masks with less pixels are ignored.
            max_bbox_area_ratio (float, optional): Maximum bounding box area ratio to image area. Defaults to None.
            mask_conf_threshold (float, optional): Minimum mask confidence threshold. Defaults to None.

        Returns:
            Detections: Filtered detections.

        """
        # If no detection at all
        if len(detections) == 0:
            return detections

        if (
            detections.mask is None
            or detections.xyxy is None
            or detections.class_id is None
            or detections.data is None
            or "classes" not in detections.data
        ):
            logger.error("Detections must have masks, bounding boxes, and class IDs.")
            return detections

        # Filter out the objects based on various criteria
        idx_to_keep = []
        for idx, _ in enumerate(detections):
            # local_class_id = detections.class_id[idx]
            class_name = detections.data["classes"][idx]

            # Skip masks that are too small
            mask_area = detections.mask[idx].sum()
            if mask_area < max(mask_area_threshold, 10):
                logger.debug(f"Skipped due to small mask area ({mask_area} pixels) - Class: {class_name}")
                continue

            # Skip the BG classes
            if skip_bg and class_name in bg_classes:
                logger.debug(f"Skipped background class: {class_name}")
                continue

            # Skip the non-background boxes that are too large
            if class_name not in bg_classes:
                x1, y1, x2, y2 = detections.xyxy[idx]
                bbox_area = (x2 - x1) * (y2 - y1)
                image_area = image.shape[-2] * image.shape[-1]
                if max_bbox_area_ratio is not None and bbox_area > max_bbox_area_ratio * image_area:
                    logger.debug(
                        f"Skipped due to large bounding box area ratio - Class: {class_name}, Area Ratio: {bbox_area / image_area:.4f}"
                    )
                    continue

            # Skip masks with low confidence
            if mask_conf_threshold is not None and detections.confidence is not None:
                if detections.confidence[idx] < mask_conf_threshold:
                    logger.debug(f"Skipped due to low confidence ({detections.confidence[idx]}) - Class: {class_name}")
                    continue

            idx_to_keep.append(idx)

        logger.debug(f"Kept {len(idx_to_keep)}/{len(detections)} detections after filtering.")
        return detections[idx_to_keep]  # type: ignore

    @staticmethod
    def _masks_subtract_contained(
        xyxy: Float[np.ndarray, "N 4"], mask: Bool[np.ndarray, "N H W"], th1: float = 0.8, th2: float = 0.7
    ) -> Bool[np.ndarray, "N H W"]:
        """Compute the containing relationship between all pair of bounding boxes. For each mask, subtract the mask of bounding boxes that are contained by it.

        Args:
            xyxy (Float[np.ndarray, "N 4"]): Bounding boxes in (x1, y1, x2, y2) format
            mask (Bool[np.ndarray, "N H W"]): Binary masks
            th1 (float): Threshold for computing intersection over box1
            th2 (float): Threshold for computing intersection over box2

        Returns:
            mask_sub (Bool[np.ndarray, "N H W"]): Binary mask

        """
        # Get areas of each xyxy
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])  # (N,)

        # Compute intersection boxes
        # left-top points (N, N, 2)
        lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])
        # right-bottom points (N, N, 2)
        rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])

        # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)
        inter = (rb - lt).clip(min=0)

        # Compute areas of intersection boxes
        inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)

        inter_over_box1 = inter_areas / areas[:, None]  # (N, N)
        # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
        inter_over_box2 = inter_over_box1.T  # (N, N)

        # if the intersection area is smaller than th2 of the area of box1,
        # and the intersection area is larger than th1 of the area of box2,
        # then box2 is considered contained by box1
        contained = (inter_over_box1 < th2) & (inter_over_box2 > th1)  # (N, N)
        contained_idx = contained.nonzero()  # (num_contained, 2)

        mask_sub = mask.copy()  # (N, H, W)
        # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
        for i in range(len(contained_idx[0])):
            mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])

        return mask_sub
