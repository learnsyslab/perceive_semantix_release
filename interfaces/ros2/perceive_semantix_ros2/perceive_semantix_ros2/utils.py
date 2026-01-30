import warnings
from typing import Union

import numpy as np
from jaxtyping import Float


def scale_intrinsics(
    intrinsics: Union[Float[np.ndarray, "N 3 3"], Float[np.ndarray, "N 4 4"]],
    h_ratio: Union[float, int],
    w_ratio: Union[float, int],
) -> Float[np.ndarray, "N M M"]:
    r"""Scale the intrinsics appropriately for resized frames where :math:`h_\text{ratio} = h_\text{new} / h_\text{old}` and :math:`w_\text{ratio} = w_\text{new} / w_\text{old}`.

    Args:
        intrinsics (Union[Float[np.ndarray, "N 3 3"], Float[np.ndarray, "N 4 4"]]): Intrinsics matrix of original frame
        h_ratio (float or int): Ratio of new frame's height to old frame's height
            :math:`h_\text{ratio} = h_\text{new} / h_\text{old}`
        w_ratio (float or int): Ratio of new frame's width to old frame's width
            :math:`w_\text{ratio} = w_\text{new} / w_\text{old}`

    Returns:
        numpy.ndarray: Intrinsics matrix scaled approprately for new frame size

    """
    if isinstance(intrinsics, np.ndarray):
        scaled_intrinsics = intrinsics.astype(np.float32).copy()
    else:
        raise TypeError("Unsupported input intrinsics type {}".format(type(intrinsics)))
    if not (intrinsics.shape[-2:] == (3, 3) or intrinsics.shape[-2:] == (4, 4)):
        raise ValueError(
            "intrinsics must have shape (*, 3, 3) or (*, 4, 4), but had shape {} instead".format(intrinsics.shape)
        )
    if (intrinsics[..., -1, -1] != 1).any() or (intrinsics[..., 2, 2] != 1).any():
        warnings.warn("Incorrect intrinsics: intrinsics[..., -1, -1] and intrinsics[..., 2, 2] should be 1.")

    scaled_intrinsics[..., 0, 0] *= w_ratio  # fx
    scaled_intrinsics[..., 1, 1] *= h_ratio  # fy
    scaled_intrinsics[..., 0, 2] *= w_ratio  # cx
    scaled_intrinsics[..., 1, 2] *= h_ratio  # cy
    return scaled_intrinsics
