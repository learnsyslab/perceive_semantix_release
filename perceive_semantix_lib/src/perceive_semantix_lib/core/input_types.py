import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from jaxtyping import Float, UInt8, jaxtyped
from typeguard import typechecked as typechecker

logger = logging.getLogger(__name__)


@jaxtyped(typechecker=typechecker)
@dataclass
class InputData:
    camera_intrinsics: Float[np.ndarray, "3 3"]
    color: UInt8[np.ndarray, "N M 3"]
    depth: Float[np.ndarray, "N M 1"]
    pose: Float[np.ndarray, "4 4"]

    def verify(self) -> bool:
        if self.color.max() > 255 or self.color.min() < 0:
            logger.warning(
                f"Color image values are expected to be in the range [0, 255]. Got {self.color.min()} to {self.color.max()}, dtype= {self.color.dtype}."
            )
            return False
        if not np.all(np.isfinite(self.depth)):
            logger.warning("There is a non-finite value in the depth measuremente. Invalid values should equal 0.0")
            return False
        return True


@dataclass
class InputDataStamped:
    time_sec: float
    data: Optional[InputData] = None


@jaxtyped(typechecker=typechecker)
@dataclass(frozen=True)
class TorchInputData:
    camera_intrinsics: Float[torch.Tensor, "3 3"]
    color: UInt8[torch.Tensor, "1 3 H W"]
    depth: Float[torch.Tensor, "1 1 H W"]
    pose: Float[torch.Tensor, "4 4"]

    @classmethod
    def from_input_data(cls, input_data: InputData) -> "TorchInputData":
        return cls(
            camera_intrinsics=torch.tensor(input_data.camera_intrinsics, dtype=torch.float32),
            color=torch.tensor(input_data.color, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0),
            depth=torch.tensor(input_data.depth, dtype=torch.float32).squeeze(-1).unsqueeze(0).unsqueeze(0),
            pose=torch.tensor(input_data.pose, dtype=torch.float32),
        )
