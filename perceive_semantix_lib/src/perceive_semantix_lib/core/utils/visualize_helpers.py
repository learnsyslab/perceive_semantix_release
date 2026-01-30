import dataclasses
import logging

import numpy as np
from jaxtyping import UInt8
from supervision.annotators.core import BoxAnnotator, LabelAnnotator, MaskAnnotator
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette

logger = logging.getLogger(__name__)


def vis_result_fast(
    image: UInt8[np.ndarray, "H W 3"],
    detections: Detections,
    classes: list[str],
    color: Color | ColorPalette = ColorPalette.DEFAULT,
    instance_random_color: bool = False,
    draw_bbox: bool = True,
) -> UInt8[np.ndarray, "H W 3"]:
    """Annotate the image with the detection results.

    This is fast but of the same resolution of the input image, thus can be blurry.
    """
    # annotate image with detections
    bounding_box_annotator = BoxAnnotator(color=color)
    label_annotator = LabelAnnotator(color=color, text_scale=0.3, text_thickness=1, text_padding=2)
    mask_annotator = MaskAnnotator(color=color)

    if hasattr(detections, "confidence") and hasattr(detections, "class_id"):
        confidences = detections.confidence
        class_ids = detections.class_id
        if confidences is not None:
            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for confidence, class_id in zip(confidences, class_ids)  # type: ignore
            ]
            labels = [label + f" {i}" for i, label in enumerate(labels)]  # make labels unique
        else:
            labels = [f"{classes[class_id]}" for class_id in class_ids]  # type: ignore
    else:
        logger.error("Detections object does not have 'confidence' or 'class_id' attributes or one of them is missing.")
        return image

    if instance_random_color:
        # generate random colors for each segmentation
        # First create a shallow copy of the input detections
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

    if draw_bbox:
        annotated_image = bounding_box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image
