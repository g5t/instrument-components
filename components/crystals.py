from dataclasses import dataclass
from typing import Tuple
from numpy import ndarray


@dataclass
class IdealCrystal:
    position: Tuple[float, float, float]
    tau: Tuple[float, float, float]

    @property
    def momentum(self) -> float:
        from math import sqrt
        return sqrt(sum([t*t for t in self.tau]))

    @property
    def momentum_vector(self) -> Tuple[float, float, float]:
        return -self.tau[0], -self.tau[1], -self.tau[2]

    @property
    def plane_spacing(self) -> float:
        from math import pi
        return 2 * pi / self.momentum

    def scattering_angle_rad(self, *, wavelength=None, wavenumber=None, k=None) -> float:
        from math import pi, asin
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
        return 2 * asin(self.momentum / (2 * k))

    def scattering_angle_deg(self, **kwargs):
        from math import pi
        return self.scattering_angle_rad(**kwargs) * 180 / pi

    def reflectivity(self, *a, **k) -> float:
        return 1.

    def transmission(self, *a, **k) -> float:
        return 1.

    @staticmethod
    def _serialize_types():
        from numpy import dtype
        return dtype([(n, 'f4') for n in ('pos0', 'pos1', 'pos2', 'tau0', 'tau1', 'tau2')])

    @property
    def _serialize_data(self):
        from numpy import hstack
        return hstack(self.position, self.tau)

    def serialize(self):
        from numpy.lib.recfunctions import unstructured_to_structured as u2s
        return u2s(self._serialize_data, self._serialize_types())

    @staticmethod
    def deserialize(structured: ndarray):
        dt = IdealCrystal._serialize_types()
        if structured.dtype != dt:
            raise RuntimeError(f"Expected types {dt} but provided with {structured.dtype}")
        if structured.size > 1:
            poss = [(s['pos0'], s['pos1'], s['pos2']) for s in structured]
            taus = [(s['tau0'], s['tau1'], s['tau2']) for s in structured]
            return [IdealCrystal(pos, tau) for pos, tau in zip(poss, taus)]
        pos = structured['pos0'], structured['pos1'], structured['pos2']
        tau = structured['tau0'], structured['tau1'], structured['tau2']
        return IdealCrystal(pos, tau)


@dataclass
class Crystal(IdealCrystal):
    width: float   # length in the scattering plane, perpendicular to Q
    height: float  # length perpendicular to the scattering plane
    depth: float   # length along Q

    @staticmethod
    def _serialize_types():
        from numpy import dtype
        t = super()._serialize_types().descr
        t.extend([(x, 'f4') for x in ('width', 'height', 'depth')])
        return dtype(t)

    @property
    def _serialize_data(self):
        from numpy import hstack
        return hstack((super(self)._serialize_data, (self.width, self.height, self.depth)))

    @staticmethod
    def deserialize(structured: ndarray):
        dt = Crystal._serialize_types()
        if structured.dtype != dt:
            raise RuntimeError(f"Expected types {dt} but provided with {structured.dtype}")
        if structured.size > 1:
            poss = [(s['pos0'], s['pos1'], s['pos2']) for s in structured]
            taus = [(s['tau0'], s['tau1'], s['tau2']) for s in structured]
            widths = [s['width'] for s in structured]
            heights = [s['height'] for s in structured]
            depths = [s['depth'] for s in structured]
            return [Crystal(*pack) for *pack in zip(poss, taus, widths, heights, depths)]
        pos = structured['pos0'], structured['pos1'], structured['pos2']
        tau = structured['tau0'], structured['tau1'], structured['tau2']
        return Crystal(pos, tau, structured['width'], structured['height'], structured['depth'])


