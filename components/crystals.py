from dataclasses import dataclass
from typing import Tuple


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


@dataclass
class Crystal:
    width: float   # length in the scattering plane, perpendicular to Q
    height: float  # length perpendicular to the scattering plane
    depth: float   # length along Q
