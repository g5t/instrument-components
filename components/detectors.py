from dataclasses import dataclass
from typing import Tuple


@dataclass
class Wire:
    at: Tuple[float, float, float]
    to: Tuple[float, float, float]

    @property
    def center_of_mass(self) -> Tuple[float, float, float]:
        return (self.at[0] + self.to[0])/2., (self.at[1] + self.to[1])/2., (self.at[2] + self.to[2])/2.

    def charge_position(self, a, b) -> Tuple[float, ...]:
        if a+b == 0:
            raise RuntimeError("Sum of a and b must not zero")
        return self.__charge_position(a/(a+b), b/(a+b))

    def __charge_position(self, fa, fb) -> Tuple[float, float, float]:
        if fb < 0 or fb > 1 or fa < 0 or fa > 1:
            raise RuntimeError("Both a and b should have the same sign")
        return fa * self.at[0] + fb * self.to[0], fa * self.at[1] + fb * self.to[1], fa *self.at[2] + fb * self.to[2]


@dataclass
class DiscreteWire(Wire):
    elements: int

    def __post_init__(self):
        if self.elements == 0:
            raise RuntimeError("A discrete detector should have a finite number of elements")
        if self.elements < 0:
            raise RuntimeError("A discrete detector must have a positive number of elements")

    def charge_index(self, a, b) -> int:
        from math import floor
        if a * b < 0:
            raise RuntimeError("Both a and b should have the same sign")
        if a+b == 0:
            raise RuntimeError("Sum of a and b must not be zero")
        # Pick the *bin* index (0,N-1) -- but a/(a+b) is on the interval (0,1)
        n = floor(self.elements * (a / (a + b))) if b else self.elements - 1
        if 0 > n or n > self.elements:
            raise RuntimeError("Error calculating index")
        return n

    def index_position(self, index) -> Tuple[float, float, float]:
        # take the bin center as its position -- this is (n+1/2)/N along the whole length
        fa = (index + 0.5) / self.elements
        return self.__charge_position(fa, 1 - fa)

    def charge_position(self, a, b) -> Tuple[float, float, float]:
        return self.index_position(self.charge_index(a, b))


@dataclass
class DiscreteTube(DiscreteWire):
    radius: float


@dataclass
class He3Tube(DiscreteTube):
    pressure: float
