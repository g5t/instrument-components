from dataclasses import dataclass
from typing import Tuple
from numpy import ndarray


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
        return fa * self.at[0] + fb * self.to[0], fa * self.at[1] + fb * self.to[1], fa * self.at[2] + fb * self.to[2]

    @staticmethod
    def _serialize_types():
        from numpy import dtype
        return dtype([(n, 'f4') for n in ('at0', 'at1', 'at2', 'to0', 'to1', 'to2')])

    @property
    def _serialize_data(self):
        from numpy import hstack
        return hstack(self.at, self.to)

    def serialize(self):
        from numpy.lib.recfunctions import unstructured_to_structured as u2s
        return u2s(self._serialize_data, self._serialize_types())

    @staticmethod
    def deserialize(structured: ndarray):
        dt = Wire._serialize_types()
        if structured.dtype != dt:
            raise RuntimeError(f"Expected types {dt} but provided with {structured.dtype}")
        if structured.size > 1:
            ats = [(s['at0'], s['at1'], s['at2']) for s in structured]
            tos = [(s['to0'], s['to1'], s['to2']) for s in structured]
            return [Wire(at, to) for at, to in zip(ats, tos)]
        at = structured['at0'], structured['at1'], structured['at2']
        to = structured['to0'], structured['to1'], structured['to2']
        return Wire(at, to)


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

    @staticmethod
    def _serialize_types():
        from numpy import dtype
        t = super()._serialize_types().descr
        t.append(('elements', '<i4'))  # little-endian 32-bit signed integer
        return dtype(t)

    @property
    def _serialize_data(self):
        from numpy import hstack
        return hstack((super(self)._serialize_data, (self.elements,)))

    @staticmethod
    def deserialize(structured: ndarray):
        dt = DiscreteWire._serialize_types()
        if structured.dtype != dt:
            raise RuntimeError(f"Expected types {dt} but provided with {structured.dtype}")
        if structured.size > 1:
            ats = [(s['at0'], s['at1'], s['at2']) for s in structured]
            tos = [(s['to0'], s['to1'], s['to2']) for s in structured]
            elements = [s['elements'] for s in structured]
            return [DiscreteWire(at, to, element) for at, to, element in zip(ats, tos, elements)]
        at = structured['at0'], structured['at1'], structured['at2']
        to = structured['to0'], structured['to1'], structured['to2']
        return DiscreteWire(at, to, structured['elements'])


@dataclass
class DiscreteTube(DiscreteWire):
    radius: float

    @property
    def _serialize_data(self):
        from numpy import hstack
        return hstack((super(self)._serialize_data, (self.radius, )))

    @staticmethod
    def _serialize_types():
        from numpy import dtype
        t = super()._serialize_types().descr
        t.append(('radius', 'f4'))
        return dtype(t)

    @staticmethod
    def deserialize(structured: ndarray):
        dt = DiscreteTube._serialize_types()
        if structured.dtype != dt:
            raise RuntimeError(f"Expected types {dt} but provided with {structured.dtype}")
        if structured.size > 1:
            ats = [(s['at0'], s['at1'], s['at2']) for s in structured]
            tos = [(s['to0'], s['to1'], s['to2']) for s in structured]
            elements = [s['elements'] for s in structured]
            radii = [s['radius'] for s in structured]
            return [DiscreteTube(at, to, element, radius) for at, to, element, radius in zip(ats, tos, elements, radii)]
        at = structured['at0'], structured['at1'], structured['at2']
        to = structured['to0'], structured['to1'], structured['to2']
        return DiscreteTube(at, to, structured['elements'], structured['radius'])


@dataclass
class He3Tube(DiscreteTube):
    pressure: float

    @property
    def _serialize_data(self):
        from numpy import hstack
        return hstack((super(self)._serialize_data, (self.pressure,)))

    @staticmethod
    def _serialize_types():
        from numpy import dtype
        t = super()._serialize_types().descr
        t.append(('pressure', 'f4'))
        return dtype(t)

    @staticmethod
    def deserialize(structured: ndarray):
        dt = He3Tube._serialize_types()
        if structured.dtype != dt:
            raise RuntimeError(f"Expected types {dt} but provided with {structured.dtype}")
        if structured.size > 1:
            ats = [(s['at0'], s['at1'], s['at2']) for s in structured]
            tos = [(s['to0'], s['to1'], s['to2']) for s in structured]
            elements = [s['elements'] for s in structured]
            radii = [s['radius'] for s in structured]
            pressures = [s['pressure'] for s in structured]
            return [He3Tube(*pack) for pack in zip(ats, tos, elements, radii, pressures)]
        at = structured['at0'], structured['at1'], structured['at2']
        to = structured['to0'], structured['to1'], structured['to2']
        return He3Tube(at, to, structured['elements'], structured['radius'], structured['pressure'])
