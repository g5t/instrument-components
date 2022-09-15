from dataclasses import dataclass, field
from typing import List, Tuple, Union
from h5py import File, Group
from scipp import Variable, vector

from .detectors import DiscreteTube
from .crystals import IdealCrystal
from .serialize import vector_serialize_types, vector_deserialize, vector_serialize


def serialize_to(g, name, what):
    from numpy import vstack
    ser = vstack([x.serialize() for x in what])
    g.create_dataset(name, data=ser)
    g[name].attrs['py_class'] = type(what[0]).__name__


@dataclass
class DirectSecondary:
    detectors: List[DiscreteTube]
    sample_at: Variable = field(default_factory=lambda: vector([0., 0., 0.], unit='m'))

    def __eq__(self, other):
        return all(sd == od for sd, od in zip(self.detectors, other.detectors)) and self.sample_at == other.sample_at

    def approx(self, other):
        from scipp import allclose
        if not isinstance(other, DirectSecondary):
            return False
        return all(sd.approx(od) for sd, od in zip(self.detectors, other.detectors)) \
               and allclose(self.sample_at, other.sample_at)

    def __post_init__(self):
        from scipp import DType
        if any([not isinstance(x, DiscreteTube) for x in self.detectors]):
            raise RuntimeError("DiscreteTube detectors expected")
        if self.sample_at.dtype != DType.vector3:
            raise RuntimeError("sample_at should be a scipp.DType('vector3')")

    def final_vector(self, detector: int, element: int) -> Variable:
        p = self.detectors[detector].index_position(element)
        return p - self.sample_at.to(unit=p.unit)

    def final_distance(self, detector, element) -> Variable:
        from scipp import sqrt, dot
        v = (self.final_vector(detector, element))
        return sqrt(dot(v, v))

    def final_direction(self, detector, element) -> Variable:
        from scipp import sqrt, dot
        v = self.final_vector(detector, element)
        return v / sqrt(dot(v, v))

    def add_to_hdf(self, obj: Union[File, Group]):
        group = obj.create_group('DirectSecondary')
        group.attrs['py_class'] = 'DirectSecondary'
        group.attrs['py_module'] = 'nicess'
        group.create_dataset('sample', data=vector_serialize(self.sample_at, 'sample', dtype='f8'))
        serialize_to(group, 'detectors', self.detectors)

    @staticmethod
    def from_hdf(obj: Group):
        if 'py_class' not in obj.attrs:
            raise RuntimeError("Expected group to have an attribute named 'py_class'")
        if obj.attrs['py_class'] != 'DirectSecondary':
            raise RuntimeError(f"Expected attribute 'py_class' to be 'DirectSecondary' but got {obj.attrs['py_class']}")
        if 'detectors' not in obj:
            raise RuntimeError("Expected 'detectors' group in provided HDF5 group")
        if 'py_class' not in obj['detectors'].attrs:
            raise RuntimeError("Expected detectors group to have an attribute named 'py_class'")
        if obj['detectors'].attrs['py_class'] != 'DiscreteTube':
            raise RuntimeError(f"Expected detectors to be 'DiscreteTube' but got {obj['detectors'].attrs['py_class']}")

        if 'sample' not in obj:
            raise RuntimeError("Expected 'sample_at' group in provided HDF5 group")

        detectors = DiscreteTube.deserialize(obj['detectors'])
        sample_at = vector_deserialize(obj['sample'], 'sample').squeeze()

        return DirectSecondary(detectors, sample_at)


@dataclass
class IndirectSecondary:
    detectors: List[DiscreteTube]
    analyzers: List[IdealCrystal]
    analyzer_per_detector: List[int]
    sample_at: Variable = field(default_factory=lambda: vector([0., 0., 0.], unit='m'))

    def __post_init__(self):
        from scipp import DType
        n_det = len(self.detectors)
        n_ana = len(self.analyzers)
        if len(self.analyzer_per_detector) != n_det:
            raise RuntimeError("The detector-to-analyzer map must have one entry per detector")
        if any((x < 0 or x > n_ana for x in self.analyzer_per_detector)):
            raise RuntimeError("The analyzer index for each detector must be valid")
        if any([not isinstance(x, DiscreteTube) for x in self.detectors]):
            raise RuntimeError("DiscreteTube detectors expected")
        if any([not isinstance(x, IdealCrystal) for x in self.analyzers]):
            raise RuntimeError("IdealCrystal analyzers expected")
        if self.sample_at.dtype != DType.vector3:
            raise RuntimeError("sample_at should be a scipp.DType('vector3')")

    def scattering_plane_normal(self, analyzer) -> Variable:
        from scipp import sqrt, dot, cross
        tau = self.analyzers[analyzer].tau  # points into the scattering plane
        mod_q = sqrt(dot(tau, tau))

        a = self._analyzer_center(analyzer)
        mod_a = sqrt(dot(a, a))

        # the cross product: tau x a
        n = cross(tau / mod_q, a / mod_a)
        return n

    def _analyzer_center(self, analyzer) -> Variable:
        a = self.analyzers[analyzer].position
        return a - self.sample_at.to(unit=a.unit)

    def _detector_partial_vectors(self, detector, element) \
            -> Tuple[Variable, Variable, Variable, Variable, Variable, Variable, Variable, Variable]:
        from scipp import dot, sqrt
        # TODO CONTINUE!
        d = self.detector_vector(detector, element)

        analyzer = self.analyzer_per_detector[detector]
        a = self._analyzer_center(analyzer)
        n = self.scattering_plane_normal(analyzer)

        d_dot_n = dot(d, n)
        d_n = d_dot_n * n

        # the vector from the analyzer center to in-plane detector position is the scattered wavevector direction
        f = d - d_n - a

        mod_a = sqrt(dot(a, a))
        mod_d_n = sqrt(dot(d_n, d_n))
        mod_f = sqrt(dot(f, f))

        mod_a_n = (mod_a / (mod_a + mod_f)) * mod_d_n
        a_n = mod_a_n * n

        return a, f, d_n, a_n, mod_a, mod_f, mod_d_n, mod_a_n

    def detector_vector(self, detector, element) -> Variable:
        p = self.detectors[detector].index_position(element)
        return p - self.sample_at.to(unit=p.unit)

    def analyzer_vector(self, detector, element) -> Variable:
        a, f, d_n, a_n, _, _, _, _ = self._detector_partial_vectors(detector, element)
        return a + a_n

    def final_direction(self, detector, element) -> Variable:
        from scipp import sqrt, dot
        av = self.analyzer_vector(detector, element)
        a = sqrt(dot(av, av))
        return av / a

    def final_distance(self, detector, element) -> Variable:
        from scipp import sqrt
        _, _, _, _, la, lf, ld_n, _ = self._detector_partial_vectors(detector, element)
        return sqrt((la + lf) * (la + lf) + ld_n * ld_n)

    def add_to_hdf(self, obj: Union[File, Group]):
        group = obj.create_group('IndirectSecondary')
        group.attrs['py_class'] = 'IndirectSecondary'
        group.attrs['py_module'] = 'nicess'
        group.create_dataset('sample', data=vector_serialize(self.sample_at, 'sample', dtype='f8'))
        serialize_to(group, 'detectors', self.detectors)
        serialize_to(group, 'analyzers', self.analyzers)
        group.create_dataset('analyzer_per_detector', data=self.analyzer_per_detector, dtype='<i4')

    @staticmethod
    def from_hdf(obj: Group):
        if 'py_class' not in obj.attrs:
            raise RuntimeError("Expected group to have an attributed named 'py_class'")
        if obj.attrs['py_class'] != 'IndirectSecondary':
            raise RuntimeError(
                f"Expected attribute 'py_class' to be 'IndirectSecondary' but got {obj.attrs['py_class']}")

        if 'detectors' not in obj:
            raise RuntimeError("Expected 'detectors' group in provided HDF5 group")
        if 'py_class' not in obj['detectors'].attrs:
            raise RuntimeError("Expected detectors group to have an attribute named 'py_class'")
        if obj['detectors'].attrs['py_class'] != 'DiscreteTube':
            raise RuntimeError(f"Expected detectors to be 'DiscreteTube' but got {obj['detectors'].attrs['py_class']}")

        if 'analyzers' not in obj:
            raise RuntimeError("Expected 'analyzers' group in provided HDF5 group")
        if 'py_class' not in obj['analyzers'].attrs:
            raise RuntimeError("Expected analyzers group to have an attribute named 'py_class'")
        if obj['analyzers'].attrs['py_class'] != 'IdealCrystal':
            raise RuntimeError(f"Expected analyzers to be 'IdealCrystal' but got {obj['analyzers'].attrs['py_class']}")

        if 'sample' not in obj:
            raise RuntimeError("Expected 'sample_at' group in provided HDF5 group")

        detectors = DiscreteTube.deserialize(obj['detectors'])
        analyzers = IdealCrystal.deserialize(obj['analyzers'])
        analyzer_per_detector = list(obj['analyzer_per_detector'])
        sample_at = vector_deserialize(obj['sample'], 'sample').squeeze()

        return IndirectSecondary(detectors, analyzers, analyzer_per_detector, sample_at)
