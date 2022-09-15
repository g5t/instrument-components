from dataclasses import dataclass
from numpy import ndarray
from scipp import Variable
from .serialize import vector_deserialize, vector_serialize_types


@dataclass
class IdealCrystal:
    position: Variable
    tau: Variable
    
    def __post_init__(self):
        from scipp import DType
        if self.position.dtype != DType.vector3:
            raise RuntimeError("position must be of type scipp.DType('vector3')")
        if self.tau.dtype != DType.vector3:
            raise RuntimeError("tau must be of type scipp.DType('vector3')")

    @property
    def momentum(self) -> Variable:
        from scipp import sqrt, dot
        return sqrt(dot(self.tau, self.tau))

    @property
    def momentum_vector(self) -> Variable:
        return -self.tau

    @property
    def plane_spacing(self) -> Variable:
        from math import pi
        return 2 * pi / self.momentum

    def scattering_angle(self, **kwargs) -> Variable:
        from math import pi, inf
        from scipp import asin, scalar, isinf, isnan, abs
        if len(kwargs) != 1:
            raise RuntimeError("A single keyword argument (k, wavenumber, wavelength) is required")
        k = kwargs.get('k', kwargs.get('wavenumber', 2 * pi / kwargs.get('wavelength', scalar(inf, unit='angstrom'))))
        if k.value == 0 or isinf(k) or isnan(k):
            raise RuntimeError("The provided keyword must produce a finite wavenumber")
        t = self.momentum.to(unit=k.unit)
        if t > 2 * abs(k):
            raise RuntimeError(f"Bragg scattering from |Q|={t:c} planes is not possible for k={k:c}")
        return 2 * asin(t / (2 * k))

    def reflectivity(self, *a, **k) -> float:
        return 1.

    def transmission(self, *a, **k) -> float:
        return 1.

    @property
    def _serialize_types(self):
        pairs = list(vector_serialize_types(self.position, name='position', dtype='f4'))
        pairs.extend(vector_serialize_types(self.tau, name='tau', dtype='f4'))
        return pairs

    @property
    def _serialize_data(self):
        from numpy import hstack
        return hstack((self.position.values, self.tau.values))

    def serialize(self):
        from numpy.lib.recfunctions import unstructured_to_structured as u2s
        from numpy import dtype
        return u2s(self._serialize_data, dtype(self._serialize_types))

    @staticmethod
    def deserialize(structured: ndarray):
        dim = 'crystals'
        pos = vector_deserialize(structured, 'position', dim=dim)
        tau = vector_deserialize(structured, 'tau', dim=dim)
        out = [IdealCrystal(*pack) for pack in zip(pos, tau)]
        return out[0] if len(out) == 1 else out


@dataclass
class Crystal(IdealCrystal):
    shape: Variable # lengths: (in-scattering-plane perpendicular to Q, perpendicular to plane, along Q)

    def __post_init__(self):
        from scipp import DType
        super().__post_init__()
        if self.shape.dtype != DType.vector3:
            raise RuntimeError("shape must be of type scipp.DType('vector3')")

    def _serialize_types(self):
        types = super()._serialize_types
        types.extend(vector_serialize_types(self.shape, 'shape', dtype='f4'))
        return types

    @property
    def _serialize_data(self):
        from numpy import hstack
        return hstack((super()._serialize_data, self.shape.values))

    @staticmethod
    def deserialize(structured: ndarray):
        dim = 'crystals'
        pos = vector_deserialize(structured, 'position', dim=dim)
        tau = vector_deserialize(structured, 'tau', dim=dim)
        shape = vector_deserialize(structured, 'shape', dim=dim)
        out = [Crystal(*pack) for pack in zip(pos, tau, shape)]
        return out[0] if len(out) == 1 else out



