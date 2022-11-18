from dataclasses import dataclass, field
from typing import List, Tuple, Union
from h5py import File, Group
from scipp import Variable, vector, zeros

from .detectors import DiscreteTube
from .crystals import IdealCrystal
from .serialize import vector_serialize_types, vector_deserialize, vector_serialize, deserialize_valid_class


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
        if not deserialize_valid_class(obj['detectors'], DiscreteTube):
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
    analyzer_map: Variable = field(default_factory=lambda: zeros(shape=[0, 0], dims=['channel', 'pair']))
    detector_map: Variable = field(default_factory=lambda: zeros(shape=[0, 0, 0], dims=['channel', 'pair', 'tube']))

    def __eq__(self, other):
        if not isinstance(other, IndirectSecondary):
            return False
        if self.sample_at != other.sample_at:
            return False
        if not all(sd == od for sd, od in zip(self.detectors, other.detectors)):
            return False
        if not all(sa == oa for sa, oa in zip(self.analyzers, other.analyzers)):
            return False
        return all(sad == oad for sad, oad in zip(self.analyzer_per_detector, other.analyzer_per_detector))

    def approx(self, other):
        from scipp import allclose
        if not isinstance(other, IndirectSecondary):
            return False
        if not allclose(self.sample_at, other.sample_at):
            return False
        if not all(sd.approx(od) for sd, od in zip(self.detectors, other.detectors)):
            return False
        if not all(sa.approx(oa) for sa, oa in zip(self.analyzers, other.analyzers)):
            return False
        return all(sad == oad for sad, oad in zip(self.analyzer_per_detector, other.analyzer_per_detector))

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
        return n / sqrt(dot(n, n))

    def _analyzer_center(self, analyzer) -> Variable:
        a = self.analyzers[analyzer].position
        return a - self.sample_at.to(unit=a.unit)

    def _detector_partial_vectors(self, detector: int, d: Variable) \
            -> Tuple[Variable, Variable, Variable, Variable, Variable, Variable, Variable, Variable]:
        from scipp import dot, sqrt
        # TODO CONTINUE!

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

    def continuous_detector_vector(self, detector, ratio) -> Variable:
        p = self.detectors[detector].continuous_position(ratio)
        return p - self.sample_at.to(unit=p.unit)

    def _analyzer_vector(self, detector, d) -> Variable:
        a, _, _, a_n, _, _, _, _ = self._detector_partial_vectors(detector, d)
        return a + a_n

    def analyzer_vector(self, detector, element) -> Variable:
        return self._analyzer_vector(detector, self.detector_vector(detector, element))

    def continuous_analyzer_vector(self, detector, ratio) -> Variable:
        return self._analyzer_vector(detector, self.continuous_detector_vector(detector, ratio))

    def final_direction(self, detector, element) -> Variable:
        from scipp import sqrt, dot
        av = self.analyzer_vector(detector, element)
        a = sqrt(dot(av, av))
        return av / a

    def continuous_final_direction(self, detector, ratio) -> Variable:
        from scipp import sqrt, dot
        av = self.continuous_analyzer_vector(detector, ratio)
        return av / sqrt(dot(av, av))

    def _final_distance(self, detector, d: Variable) -> Variable:
        from scipp import sqrt
        _, _, _, _, la, lf, ld_n, _ = self._detector_partial_vectors(detector, d)
        return sqrt((la + lf) * (la + lf) + ld_n * ld_n)

    def final_distance(self, detector, element) -> Variable:
        return self._final_distance(detector, self.detector_vector(detector, element))

    def continuous_final_distance(self, detector, ratio) -> Variable:
        return self._final_distance(detector, self.continuous_detector_vector(detector, ratio))

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
        if not deserialize_valid_class(obj['detectors'], DiscreteTube):
            raise RuntimeError(f"Expected detectors to be 'DiscreteTube' but got {obj['detectors'].attrs['py_class']}")

        if 'analyzers' not in obj:
            raise RuntimeError("Expected 'analyzers' group in provided HDF5 group")
        if 'py_class' not in obj['analyzers'].attrs:
            raise RuntimeError("Expected analyzers group to have an attribute named 'py_class'")
        if not deserialize_valid_class(obj['analyzers'], IdealCrystal):
            raise RuntimeError(f"Expected analyzers to be 'IdealCrystal' but got {obj['analyzers'].attrs['py_class']}")

        if 'sample' not in obj:
            raise RuntimeError("Expected 'sample_at' group in provided HDF5 group")

        detectors = DiscreteTube.deserialize(obj['detectors'])
        analyzers = IdealCrystal.deserialize(obj['analyzers'])
        analyzer_per_detector = list(obj['analyzer_per_detector'])
        sample_at = vector_deserialize(obj['sample'], 'sample').squeeze()

        return IndirectSecondary(detectors, analyzers, analyzer_per_detector, sample_at)

    def _broadcast_continuous_per_tube(self, detector_index: Variable):
        from scipp import concat, scalar, dot, sqrt, acos
        dim = detector_index.dims[0]

        detectors = [self.detectors[i] for i in detector_index.values]

        at = concat([d.at for d in detectors], dim=dim)
        to = concat([d.to for d in detectors], dim=dim)

        sd = (at + to) / 2.0 - self.sample_at

        analyzers = [self.analyzer_per_detector[i] for i in detector_index.values]
        sac = concat([self._analyzer_center(analyzer) for analyzer in analyzers], dim=dim)
        n = concat([self.scattering_plane_normal(analyzer) for analyzer in analyzers], dim=dim)

        # sample-to-detection-point vector component which is *out of the nominal scattering plane*
        # # This *SHOULD BE* zero for the tube-centre, unless if the tube is off centre
        d_dot_n = dot(sd, n)

        # the vector from the analyzer center to in-plane detector position is the scattered wave-vector direction
        f = sd - d_dot_n * n - sac

        mod_a = sqrt(dot(sac, sac))
        mod_f = sqrt(dot(f, f))

        mod_a_n = (mod_a / (mod_a + mod_f)) * d_dot_n
        a_n = mod_a_n * n

        # sample to analyzer interaction-point vector
        sa = sac + a_n
        # analyzer interaction-point to detection-point vector
        ad = sd - sa
        return {'sample_analyzer': sa, 'analyzer_detector': ad, 'sample_analyzer_centre': sac,
                'analyzer_detector_centre': f,
                'length_sample_analyzer_centre': mod_a, 'length_analyzer_detector_centre': mod_f,
                'signed_length_detector_position': d_dot_n}

    def _broadcast_continuous_common(self, detector_index: Variable, ratio: Variable):
        from scipp import concat, scalar, dot, sqrt, acos
        dim = detector_index.dims[0]

        detectors = [self.detectors[i] for i in detector_index.values]

        at = concat([d.at for d in detectors], dim=dim)
        to = concat([d.to for d in detectors], dim=dim)

        sd = (scalar(1) - ratio) * at + ratio * to - self.sample_at

        analyzers = [self.analyzer_per_detector[i] for i in detector_index.values]
        sac = concat([self._analyzer_center(analyzer) for analyzer in analyzers], dim=dim)
        n = concat([self.scattering_plane_normal(analyzer) for analyzer in analyzers], dim=dim)

        # sample-to-detection-point vector component which is *out of the nominal scattering plane*
        d_dot_n = dot(sd, n)

        # the vector from the analyzer center to in-plane detector position is the scattered wave-vector direction
        f = sd - d_dot_n * n - sac

        mod_a = sqrt(dot(sac, sac))
        mod_f = sqrt(dot(f, f))

        mod_a_n = (mod_a / (mod_a + mod_f)) * d_dot_n
        a_n = mod_a_n * n

        # sample to analyzer interaction-point vector
        sa = sac + a_n
        # analyzer interaction-point to detection-point vector
        ad = sd - sa
        return {'sample_analyzer': sa, 'analyzer_detector': ad, 'sample_analyzer_centre': sac,
                'analyzer_detector_centre': f,
                'length_sample_analyzer_centre': mod_a, 'length_analyzer_detector_centre': mod_f,
                'signed_length_detector_position': d_dot_n}

    def broadcast_continuous_theta(self, detector_index: Variable, ratio: Variable):
        from scipp import scalar, dot, sqrt, acos
        # have an angle between them which is 2*theta
        values = self._broadcast_continuous_common(detector_index, ratio)
        # sample to analyzer interaction-point vector
        sa = values['sample_analyzer']
        # analyzer interaction-point to detection-point vector
        ad = values['analyzer_detector']
        two_theta = acos(dot(sa, ad) / sqrt(dot(sa, sa)) / sqrt(dot(ad, ad)))
        return two_theta / scalar(2)

    def broadcast_continuous_delta_a4(self, detector_index: Variable, ratio: Variable):
        from scipp import dot, sqrt, acos, cross, vector
        values = self._broadcast_continuous_common(detector_index, ratio)
        sac, sa = values['sample_analyzer_centre'], values['sample_analyzer']
        c_hat = sac / sqrt(dot(sac, sac))
        a_hat = sa / sqrt(dot(sa, sa))
        z = vector([0, 0, 1.])
        delta_a4 = acos(dot(c_hat, a_hat))
        cca = cross(c_hat, a_hat)
        sign = dot(cca / sqrt(dot(cca, cca)), vector([0, 0, 1]))
        return delta_a4 * sign

    def broadcast_continuous_final_distance(self, detector_index: Variable, ratio: Variable):
        from scipp import sqrt
        values = self._broadcast_continuous_common(detector_index, ratio)
        a = values['length_sample_analyzer_centre']
        f = values['length_analyzer_detector_centre']
        d = values['signed_length_detector_position']
        return sqrt((a + f) * (a + f) + d * d)

    def broadcast_continuous_analyzer_distance(self, detector_index: Variable):
        values = self._broadcast_continuous_per_tube(detector_index)
        return values['length_sample_analyzer_centre']

    def broadcast_continuous_detector_distance(self, detector_index: Variable):
        values = self._broadcast_continuous_per_tube(detector_index)
        return values['length_analyzer_detector_centre']

    def broadcast_continuous_a6(self, detector_index: Variable):
        from scipp import sqrt, dot, acos
        values = self._broadcast_continuous_per_tube(detector_index)
        sa = values['sample_analyzer_centre']
        ad = values['analyzer_detector_centre']
        return acos(dot(sa, ad) / sqrt(dot(sa, sa)) / sqrt(dot(ad, ad)))

    def broadcast_continuous_plane_spacing(self, detector_index: Variable):
        from scipp import concat, scalar, dot, sqrt, acos
        dim = detector_index.dims[0]
        analyzers = [self.analyzer_per_detector[i] for i in detector_index.values]
        d = concat([self.analyzers[analyzer].plane_spacing for analyzer in analyzers], dim=dim)
        return d
