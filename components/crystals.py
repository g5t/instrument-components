from dataclasses import dataclass
from typing import Tuple
from numpy import ndarray
from scipp import Variable
from scipp.DType import vector3
from .serialize import vector_deserialize, vector_serialize_types


@dataclass
class IdealCrystal:
    position: Variable
    tau: Variable
    
    def __post_init__(self):
        from scipp.DType import vector3
        if position.dtype != vector3:
            raise RuntimeError("position must be of type scipp.DType('vector3')")
        if tau.dtype != vector3:
            raise RuntimeError("tau must be of type scipp.DType('vector3')")

    @property
    def momentum(self) -> Variable:
        from scipp import sqrt, dot
        return sqrt(dot(self.tau, self.tau))

    @property
    def momentum_vector(self) -> Variable:
        return -self.tau

    @property
    def plane_spacing(self) -> Variable
        from math import pi
        return 2 * pi / self.momentum

    def scattering_angle(self, *, wavelength=None, wavenumber=None, k=None) -> Variable:
        from math import pi
        from scipp import asin
        if wavenumber is not None and k is not None and wavenumber != k:
            raise RuntimeError("k == wavenumber; Do not provide both")
        if wavenumber is not None and k is None:
            k = wavenumber
        if k is not None and wavelength is not None and wavelength != 2 * pi / k:
            raise RuntimeError("k = 2 * pi / wavelength; Do not provide both")
        if k is None and wavelength is not None:
            k = 2 * pi / wavelength
        if k is None:
            raise RuntimeError("wavelength or wavenumber must be provided")

        return 2 * asin(self.momentum / (2 * k.to(unit=self.tau.unit)))

    def reflectivity(self, *a, **k) -> float:
        return 1.

    def transmission(self, *a, **k) -> float:
        return 1.

    @staticmethod
    def _serialize_types():
        pairs = list(vector_serialize_types(self.position, name='position', dtype='f4'))
        pairs.extend(list(vector_serialize_types(self.tau, name='tau', dtype='f4'))
        return pairs

    @property
    def _serialize_data(self):
        from numpy import hstack
        return hstack((self.position.values, self.tau.values))

    def serialize(self):
        from numpy.lib.recfunctions import unstructured_to_structured as u2s
        from numpy import dtype
        return u2s(self._serialize_data, dtype(self._serialize_types()))

    @staticmethod
    def deserialize(structured: ndarray):
        from numpy import dtype
        dt = IdealCrystal._serialize_types()
        if structured.dtype != dtype(dt):
            raise RuntimeError(f"Expected types {dt} but provided with {structured.dtype}")
        dim = 'crystals'
        pos = vector_deserialize(structured, 'position', dim=dim)
        tau = vector_deserialize(structured, 'tau', dim=dim)
        out = [IdealCrystal(*pack) for pack in zip(pos, tau)]
        return out[0] if len(out) == 1 else out


@dataclass
class Crystal(IdealCrystal):
    shape: Variable # lengths: (in-scattering-plane perpendicular to Q, perpendicular to plane, along Q)

    def __post_init__(self):
        super().__post_init__()
        if shape.dtype != vector3:
            raise RuntimeError("shape must be of type scipp.DType('vector3')")

    @staticmethod
    def _serialize_types():
        types = super(Crystal, Crystal)._serialize_types()
        types.extend(vector_serialize_types(self.shape, 'shape', dtype='f4'))
        return types

    @property
    def _serialize_data(self):
        from numpy import hstack
        return hstack((super()._serialize_data, self.shape.values))

    @staticmethod
    def deserialize(structured: ndarray):
        from numpy import dtype
        dt = Crystal._serialize_types()
        if structured.dtype != dtype(dt):
            raise RuntimeError(f"Expected types {dt} but provided with {structured.dtype}")
        dim = 'crystals'
        pos = vector_deserialize(structured, 'position', dim=dim)
        tau = vector_deserialize(structured, 'tau', dim=dim)
        shape = vector_deserialize(structured, 'shape', dim=dim)
        out = [Crystal(*pack) for pack in zip(pos, tau, shape)]
        return out[0] if len(out) == 1 else out



