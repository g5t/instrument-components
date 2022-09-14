from dataclasses import dataclass, field
from typing import List, Tuple, Union
from h5py import File, Group

from .detectors import DiscreteTube
from .crystals import IdealCrystal

def vector_length(vector: Tuple[float, float, float]) -> float:
    from math import sqrt
    return sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


def serialize_to(g, name, what):
    from numpy import vstack
    ser = vstack([x.serialize() for x in what])
    g.create_dataset(name, data=ser)
    g[name].attrs['py_class'] = type(what[0]).__name__

@dataclass
class DirectSecondary:
    detectors: List[DiscreteTube]
    sample_at: Tuple[float, float, float] = field(default_factory=lambda: (0., 0., 0.))

    def __post_init__(self):
        if any([not isinstance(x, DiscreteTube) for x in self.detectors]):
            raise RuntimeError("DiscreteTube detectors expected")

    def final_vector(self, detector: int, element: int) -> Tuple[float, float, float]:
        p = self.detectors[detector].index_position(element)
        return p[0] - self.sample_at[0], p[1] - self.sample_at[1], p[2] - self.sample_at[2]

    def final_distance(self, detector, element) -> float:
        return vector_length(self.final_vector(detector, element))

    def final_direction(self, detector, element) -> Tuple[float, float, float]:
        vector = self.final_vector(detector, element)
        v = vector_length(vector)
        return vector[0]/v, vector[1]/v, vector[2]/v

    def add_to_hdf(self, obj: Union[File, Group]):
        group = obj.create_group('DirectSecondary')
        group.attrs['py_class'] = 'DirectSecondary'
        group.attrs['py_module'] = 'components'
        group.attrs['sample_at'] = self.sample_at
        serialize_to(group, 'detectors', self.detectors)

    @staticmethod
    def from_hdf(obj: Group):
        if not 'py_class' in obj.attrs:
            raise RuntimeError("Expected group to have an attribute named 'py_class'")
        if obj.attrs['py_class'] != 'DirectSecondary':
            raise RuntimeError(f"Expected attribute 'py_class' to be 'DirectSecondary' but got {obj.attrs['py_class']}")
        if 'detectors' not in obj:
            raise RuntimeError("Expected 'detectors' group in provided HDF5 group")
        if 'py_class' not in obj['detectors'].attrs:
            raise RuntimeError("Expected detectors group to have an attribute named 'py_class'")
        if obj['detectors'].attrs['py_class'] != 'DiscreteTube':
            raise RuntimeError(f"Expected detectors to be 'DiscreteTube' but got {obj['detectors'].attrs['py_class']}")

        detectors = DiscreteTube.deserialize(obj['detectors'])
        sample_at = obj.attrs['sample_at']

        return DirectSecondary(detectors, sample_at)


@dataclass
class IndirectSecondary:
    detectors: List[DiscreteTube]
    analyzers: List[IdealCrystal]
    analyzer_per_detector: List[int]
    sample_at: Tuple[float, float, float] = field(default_factory=lambda: (0., 0., 0.))

    def __post_init__(self):
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

    def scattering_plane_normal(self, analyzer) -> Tuple[float, float, float]:
        tau = self.analyzers[analyzer].tau  # points into the scattering plane
        mod_q = vector_length(tau)

        a = self._analyzer_center(analyzer)
        mod_a = vector_length(a)

        # the cross product: tau x a
        n = tau[1] * a[2] - tau[2] * a[1], tau[2] * a[0] - tau[0] * a[2], tau[0] * a[1] - tau[1] * a[0]

        return n[0] / mod_a / mod_q, n[1] / mod_a / mod_q, n[2] / mod_a / mod_q

    def _analyzer_center(self, analyzer) -> Tuple[float, float, float]:
        a = self.analyzers[analyzer].position
        a = a[0] - self.sample_at[0], a[1] - self.sample_at[1], a[2] - self.sample_at[2]
        return a

    def _detector_partial_vectors(self, detector, element) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], float, float, float, float]:
        d = self.detector_vector(detector, element)

        analyzer = self.analyzer_per_detector[detector]
        a = self._analyzer_center(analyzer)
        n = self.scattering_plane_normal(analyzer)

        d_dot_n = sum([x * y for x, y in zip(d, n)])
        d_n = d_dot_n * n[0], d_dot_n * n[1], d_dot_n * n[2]

        # the vector from the analyzer center to in-plane detector position is the scattered wavevector direction
        f = d[0] - d_n[0] - a[0], d[1] - d_n[1] - a[1], d[2] - d_n[2] - a[2]

        mod_a = vector_length(a)
        mod_d_n = vector_length(d_n)
        mod_f = vector_length(f)
        mod_a_n = (mod_a / (mod_a + mod_f)) * mod_d_n
        a_n = mod_a_n * n[0], mod_a_n * n[1], mod_a_n * n[2]

        return a, f, d_n, a_n, mod_a, mod_f, mod_d_n, mod_a_n

    def detector_vector(self, detector, element) -> Tuple[float, float, float]:
        p = self.detectors[detector].index_position(element)
        return p[0] - self.sample_at[0], p[1] - self.sample_at[1], p[2] - self.sample_at[2]

    def analyzer_vector(self, detector, element) -> Tuple[float, float, float]:
        a, f, d_n, a_n, _, _, _, _ = self._detector_partial_vectors(detector, element)
        return a[0] + a_n[0], a[1] + a_n[1], a[2] + a_n[2]

    def final_direction(self, detector, element) -> Tuple[float, float, float]:
        av = self.analyzer_vector(detector, element)
        a = vector_length(av)
        return av[0]/a, av[1]/a, av[2]/a

    def final_distance(self, detector, element) -> float:
        from math import sqrt
        _, _, _, _, la, lf, ld_n, _ = self._detector_partial_vectors(detector, element)
        return sqrt((la + lf) * (la + lf) + ld_n * ld_n)

    def add_to_hdf(self, obj: Union[File, Group]):
        group = obj.create_group('IndirectSecondary')
        group.attrs['py_class'] = 'IndirectSecondary'
        group.attrs['py_module'] = 'components'
        group.attrs['sample_at'] = self.sample_at
        serialize_to(group, 'detectors', self.detectors)
        serialize_to(group, 'analyzers', self.analyzers)
        group.create_dataset('analyzer_per_detector', data=self.analyzer_per_detector, dtype='<i4')

    @staticmethod
    def from_hdf(obj: Group):
        if not 'py_class' in obj.attrs:
            raise RuntimeError("Expected group to have an attributed named 'py_class'")
        if obj.attrs['py_class'] != 'IndirectSecondary':
            raise RuntimeError(f"Expected attribute 'py_class' to be 'IndirectSecondary' but got {obj.attrs['py_class']}")

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

        detectors = DiscreteTube.deserialize(obj['detectors'])
        analyzers = IdealCrystal.deserialize(obj['analyzers'])
        analyzer_per_detector = list(obj['analyzer_per_detector'])
        sample_at = obj.attrs['sample_at']

        return IndirectSecondary(detectors, analyzers, analyzer_per_detector, sample_at)
