from enum import Enum
from itertools import product

from .interfaces import NeighborhoodOffsetsReceive


def moore_neighborhood(shape: tuple[int, ...]) -> list:
    dimensions = len(shape)
    offsets = list(product([-1, 0, 1], repeat=dimensions))

    offsets = [offset for offset in offsets if any(element != 0 for element in offset)]

    return offsets


def von_neumann_neighborhood(shape: tuple[int, ...]) -> list:
    offsets = moore_neighborhood(shape)
    offsets = [
        offset for offset in offsets if sum(abs(element) for element in offset) == 1
    ]
    return offsets


class NeighborhoodType(str, Enum):
    MOORE = "moore"
    VON_NEUMANN = "von_neumann"


NEIGHBORHOOD_TO_FUNC: dict[NeighborhoodType, NeighborhoodOffsetsReceive] = {
    NeighborhoodType.MOORE: moore_neighborhood,
    NeighborhoodType.VON_NEUMANN: von_neumann_neighborhood,
}
