from dataclasses import dataclass
from numpy import ndarray
from scipp import Variable
from .serialize import vector_deserialize, vector_serialize_types


@dataclass
class IdealCrystal:
    position: Variable
    tau: Variable

    def threejs_children(self, material=None, unit=None):
        from scipp import sqrt, dot, vector
        from .spatial import pythreejs_vector_to_vector_quaternion
        from pythreejs import Mesh, SphereGeometry, ConeGeometry, CylinderGeometry
        if material is None:
            from pythreejs import MeshLambertMaterial
            material = MeshLambertMaterial(color='red')
        if unit is None:
            unit = 'm'
        com = Mesh(geometry=SphereGeometry(radius=2, widthSegments=32, heightSegments=32),
                   material=material, position=self.position.to(unit=unit).value)

        length = sqrt(dot(self.tau, self.tau))
        Q = pythreejs_vector_to_vector_quaternion(vector([0, 0, 1.]), self.tau)

        shaft_geom = CylinderGeometry(radiusTop=1, radiusBottom=1, height=length.value,
                                 radialSegments=32, heightSegments=1, openEnded=True)
        head_geom = ConeGeometry(radialSegments=32, radius=2, height=0.1*length.value)


    def triangulate(self, unit=None):
        from scipp import sqrt, dot, vector, arange, concat
        from scipp.spatial import rotations_from_rotvecs
        if unit is None:
            unit = self.position.unit
        lt = sqrt(dot(self.tau, self.tau))
        # *a* vector perpendicular to tau
        p = cross(self.tau, vector([1., 0, 0]) if isclose(self.tau.fields.z, ll) else vector([0, 0, 1.]))
        p = (p/sqrt(dot(p, p)) / lt).to(unit=unit)
        a = arange(start=0, stop=360, step=10, dim='vertices', unit='degree')
        r = rotations_from_rotvecs(a*self.tau/lt)
        vertices = concat((self.position, r*p + self.position), dim='vertices')
        lv = len(r)
        triangles = [[0, i + 1, (i + 1)%lv + 1] for i in range(lv)]

        return vertices, triangles


    def __eq__(self, other):
        if not isinstance(other, IdealCrystal):
            return False
        return self.position == other.position and self.tau == other.tau

    def approx(self, other):
        from scipp import allclose
        if not isinstance(other, IdealCrystal):
            return False
        return allclose(self.position, other.position) and allclose(self.tau, other.tau)
    
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

    def wavenumber(self, scattering_angle: Variable):
        from scipp import sin
        return self.momentum / (2 * sin(0.5 * scattering_angle.to(unit='radian')))

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
        t = self._serialize_types()  # why does this need to be called? it is supposed to be a property
        return u2s(self._serialize_data, dtype(t))

    @staticmethod
    def deserialize(structured: ndarray):
        dim = 'crystals'
        pos = vector_deserialize(structured, 'position', dim=dim)
        tau = vector_deserialize(structured, 'tau', dim=dim)
        out = [IdealCrystal(*pack) for pack in zip(pos, tau)]
        return out[0] if len(out) == 1 else out


@dataclass
class Crystal(IdealCrystal):
    shape: Variable  # lengths: (in-scattering-plane perpendicular to Q, perpendicular to plane, along Q)

    def triangulate(self, unit=None):
        from .spatial import vector_to_vector_quaternion
        from scipp import vectors, vector
        if unit is None:
            unit = self.position.unit
        r = vector_to_vector_quaternion(vector([0,0,1.]), self.tau)
        x, y, z = self.shape.value/2
        vertices = vectors(unit=self.shape.unit, dims=['vertices'],
                          values=[[-x, -y, -z], [+x, -y, -z], [+x, +y, -z], [-x, +y, -z],
                                  [-x, -y, +z], [+x, -y, +z], [+x, +y, +z], [-x, +y, +z]])
        vertices = r * vertices
        faces = [[0, 2, 1], [2, 0, 3], [1, 2, 6], [1, 6, 5], [0, 1, 5], [0, 5, 4],
                 [3, 0, 4], [3, 4, 7], [2, 3, 7], [2, 7, 6], [4, 5, 6], [4, 6, 7]]
        return vertices.to(unit=unit) + self.position.to(unit=unit), faces


    def __eq__(self, other):
        if not isinstance(other, Crystal):
            return False
        return self.shape == other.shape and super().__eq__(other)

    def approx(self, other):
        from scipp import allclose
        if not isinstance(other, Crystal):
            return False
        return allclose(self.shape, other.shape) and super().approx(other)

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



