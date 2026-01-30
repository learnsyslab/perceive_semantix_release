from dataclasses import dataclass
from enum import Enum


class POCDObjectTypes(Enum):
    DYNAMIC = 0
    STATIC = 1
    DISSAPEARED = 2


@dataclass
class StandardDistributedValue:
    mean: float
    std: float


@dataclass
class BetaDistributedValue:
    alpha: float
    beta: float

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
