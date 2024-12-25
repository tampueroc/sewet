from itertools import repeat
import collections.abc
from typing import Callable, Tuple, Union, Iterable, TypeVar

T = TypeVar('T')  # Generic type variable


def _ntuple(n: int) -> Callable[[Union[T, Iterable[T]]], Tuple[T, ...]]:
    def parse(x: Union[T, Iterable[T]]) -> Tuple[T, ...]:
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)  # Convert iterable to a tuple
        return tuple(repeat(x, n))  # Repeat the value `n` times and convert to tuple
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
