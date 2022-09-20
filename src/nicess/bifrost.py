from __future__ import annotations
from dataclasses import dataclass

from scipp import Variable, vector, scalar, array, vectors

from .secondary import IndirectSecondary
from .detectors import He3Tube
from .crystals import Crystal


def __is_type__(x, t, name):
    if not isinstance(x, t):
        raise RuntimeError(f"{name} must be a {t}")


@dataclass
class BIFROST:
    # primary: BandwidthPrimary   # A minimal form of the primary spectrometer necessary to transform events
    secondary: IndirectSecondary  # A minimal form of the secondary spectrometer necessary to transform events

    @staticmethod
    def from_calibration(**params):
        # primary = ...
        params['sample'] = params.get('sample', vector([0, 0, 0.], unit='m'))
        tank = Tank.from_calibration(**params)

        secondary = tank.to_secondary(sample=params['sample'])
        return BIFROST(secondary)


@dataclass
class Triplet:
    tubes: tuple[He3Tube, He3Tube, He3Tube]

    @staticmethod
    def from_calibration(position: Variable, length: Variable, **params):
        """Take (fitting) calibration data and construct the object used to convert events to (Q,E)"""
        # The current crop of inputs is not sufficient to capture all degrees of freedom, but is a start.
        from scipp import sqrt, dot, vector
        from scipp.spatial import rotations
        from .spatial import is_scipp_vector
        map(lambda x: is_scipp_vector(*x), ((position, 'position'),))
        if position.ndim != 1 or 'tube' not in position.dims or position.sizes['tube'] != 3:
            raise RuntimeError("Expected positions for 3 'tube'")

        ori = params.get('orient', None)
        ori = rotations(values=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dims=['tube']) if ori is None else ori

        pressure = params.get('pressure', scalar(1., unit='atm'))
        radius = params.get('radius', scalar(10., unit='mm'))
        elements = params.get('elements', 10)
        map(lambda x: __is_type__(*x), ((pressure, Variable, 'pressure'), (length, Variable, 'length'),
                                        (radius, Variable, 'radius'), (elements, int, 'elements')))
        # pack the tube parameters
        pack = elements, radius.value, pressure.value

        # Make the oriented tube-axis vector(s)
        axis = ori * (length.to(unit=position.unit) * vector([0, 1., 0]))  # may be a 0-D or 1-D (tube) vector array
        tube_at = position - 0.5 * axis  # should now be a 1-D (tube) vector array
        tube_to = position + 0.5 * axis  # *ditto*
        tubes = (He3Tube(at, to, *pack) for at, to in zip(tube_at, tube_to))
        return Triplet(tuple(tubes))

    def triangulate(self, unit=None):
        from .spatial import combine_triangulations
        vts = [tube.triangulate(unit=unit) for tube in self.tubes]
        return combine_triangulations(vts)

    def extreme_path_corners(self, horizontal, vertical, unit=None):
        from .spatial import combine_extremes
        vs = [tube.extreme_path_corners(horizontal, vertical, unit=unit) for tube in self.tubes]
        return combine_extremes(vs, horizontal, vertical)


@dataclass
class Analyzer:
    blades: tuple[Crystal, ...]  # 7-9 blades

    @property
    def central_blade(self):
        return self.blades[len(self.blades) >> 1]

    @staticmethod
    def from_calibration(position: Variable, focus: Variable, **params):
        from math import pi
        from scipp.spatial import rotation
        from .spatial import is_scipp_vector
        from .rowland import rowland_blades
        map(lambda x: is_scipp_vector(*x), ((position, 'position'), (focus, 'focus')))
        count = params.get('count', 9)  # most analyzers have 9 blades
        shape = params.get('shape', vector([10., 200., 2.], unit='mm'))
        orient = params.get('orient', None)
        orient = rotation(value=[0, 0, 0, 1.]) if orient is None else orient
        tau = params.get('tau', 2 * pi / params.get('dspacing', scalar(3.355, unit='angstrom')))  # PG(002)
        # qin_coverage = params.get('qin_coverage', params.get('coverage', scalar(0.1, unit='1/angstrom')))
        coverage = params.get('coverage', scalar(2, unit='degree'))
        source = params.get('source', params.get('sample_position', vector([0, 0, 0], unit='m')))
        #
        # # Use the crystal lattice and scattering triangle to find k, then angular coverage from Q_in_plane-coverage
        # sa = position - source
        # sd = focus - source
        # ki_hat = sa / sqrt(dot(sa, sa))
        # kf_hat = (sd - sa) / sqrt(dot(sd - sa, sd - sa))
        # scattering_angle = scalar(180, unit='deg') - acos(dot(kf_hat, ki_hat))
        # k = tau / (2 * sin(0.5 * scattering_angle))
        # # the angular coverage is given by the triangle with base |k_i| and height |Q_in_plane|/2
        # alpha = atan2(0.5 * qin_coverage, k)  # the angular positions around the Rowland circle are not +/- alpha
        #
        alpha = 0.5 * coverage

        # Use the Rowland geometry to define each blade position & normal direction
        positions, taus = rowland_blades(source, position, focus, alpha, shape.fields.x, count)
        taus *= tau  # convert from directions to full tau vectors

        blades = [Crystal(p, t, shape, orient) for p, t in zip(positions, taus)]
        return Analyzer(tuple(blades))

    def triangulate(self, unit=None):
        from .spatial import combine_triangulations
        vts = [blade.triangulate(unit=unit) for blade in self.blades]
        return combine_triangulations(vts)

    def extreme_path_corners(self, horizontal, vertical, unit=None):
        from .spatial import combine_extremes
        vs = [blade.extreme_path_corners(horizontal, vertical, unit=unit) for blade in self.blades]
        return combine_extremes(vs, horizontal, vertical)


@dataclass
class Arm:
    analyzer: Analyzer
    detector: Triplet

    @staticmethod
    def from_calibration(a_position, d_position, d_length, **params):
        analyzer_orient = params.get('analyzer_orient', None)
        detector_orient = params.get('detector_orient', None)
        analyzer = Analyzer.from_calibration(a_position, d_position, **params, orient=analyzer_orient)
        detector = Triplet.from_calibration(d_position, d_length, orient=detector_orient)
        return Arm(analyzer, detector)

    def triangulate_detector(self, unit=None):
        return self.detector.triangulate(unit=unit)

    def triangulate_analyzer(self, unit=None):
        return self.analyzer.triangulate(unit=unit)

    def triangulate(self, unit=None):
        from .spatial import combine_triangulations
        return combine_triangulations([self.triangulate_analyzer(unit=unit), self.triangulate_detector(unit=unit)])

    def extreme_path_edges(self, sample: Variable):
        from scipp import concat
        from numpy import array as numpy_array, int as numpy_int
        from .spatial import is_scipp_vector, perpendicular_directions
        is_scipp_vector(sample, 'sample')
        a_pos = self.analyzer.central_blade.position.to(unit=sample.unit)
        d_pos = 0.5 * (self.detector.tubes[1].at + self.detector.tubes[1].to).to(unit=sample.unit)

        at_analyzer = self.analyzer.extreme_path_corners(*perpendicular_directions(a_pos - sample), unit=sample.unit)
        at_detector = self.detector.extreme_path_corners(*perpendicular_directions(d_pos - a_pos), unit=sample.unit)

        laa = len(at_analyzer)
        lad = len(at_detector)
        edges = [[0, 1 + a] for a in range(laa)]
        # ... filter out edges which intersect anywhere other than {sample} ?
        # TODO This requires identifying the Convex Hull and then line segments which are inside the polyehedron
        edges.extend([[1 + a, 1 + laa + d] for a in range(laa) for d in range(lad)])
        # ... filter out new edges which are non-divergent

        vertices = concat((sample, at_analyzer, at_detector), 'vertices')
        return vertices, numpy_array(edges, dtype=numpy_int)


@dataclass
class Channel:
    pairs: tuple[Arm, Arm, Arm, Arm, Arm]

    @staticmethod
    def from_calibration(relative_angle: Variable, **params):
        from math import pi
        from scipp import sqrt, dot, acos, sin, tan, atan, atan2, asin, min, max
        from scipp.constants import hbar, neutron_mass
        from scipp.spatial import rotation, rotations_from_rotvecs
        known = dict()
        dist_sa = {
            's': [1.100, 1.238, 1.342, 1.433, 1.544],
            'm': [1.189, 1.316, 1.420, 1.521, 1.623],
            'l': [1.276, 1.388, 1.493, 1.595, 1.697],
        }
        known['dist_sa'] = {k: array(values=v, unit='m', dims=['analyzer']) for k, v in dist_sa.items()}
        d_length_mm = {
            's': [217.9, 242.0, 260.8, 279.2, 298.8],
            'm': [226.0, 249.0, 267.9, 286.3, 304.8],
            'l': [233.9, 255.9, 274.9, 293.4, 311.9],
        }
        known['d_length'] = {k: array(values=v, unit='mm', dims=['analyzer']) for k, v in d_length_mm.items()}
        known['d_offset'] = vectors(values=[[0, 0, -20.], [0, 0, 0], [0, 0, 20]], unit='mm', dims=['tube'])
        known['d_orient'] = rotation(value=[0, 0, 0, 1])
        a_shape_mm = {
            's': [[12.0, 134.0, 2], [14.0, 147.1, 2], [11.5, 156.2, 2], [12.0, 165.2, 2], [13.5, 175.6, 2]],
            'm': [[12.5, 142.0, 2], [14.5, 154.1, 2], [11.5, 163.2, 2], [12.5, 172.3, 2], [13.5, 181.6, 2]],
            'l': [[13.5, 149.9, 2], [15.0, 161.0, 2], [12.0, 170.2, 2], [13.0, 179.3, 2], [14.0, 188.6, 2]],
        }
        known['a_shape'] = {k: vectors(values=v, unit='mm', dims=['analyzer']) for k, v in a_shape_mm.items()}
        known['blade_count'] = array(values=[9, 9, 9, 7, 7], dims=['analyzer'])
        known['d_spacing'] = scalar(3.355, unit='angstrom')  # PG(002)
        known['coverage'] = scalar(2., unit='degree')
        known['energy'] = array(values=[2.7, 3.2, 3.7, 4.4, 5.], unit='meV', dims=['analyzer'])

        variant = params.get('variant', 'm')
        blade_count = params.get('blade_count', known['blade_count'])
        tau = params.get('tau', 2 * pi / params.get('d_spacing', known['d_spacing']))
        crystal_shape = params.get('crystal_shape', known['a_shape'][variant])
        detector_length = params.get('detector_length', known['d_length'][variant])
        detector_orient = params.get('detector_orient', known['d_orient'])
        detector_offset = params.get('detector_offset', known['d_offset'])
        coverage = params.get('coverage', known['coverage'])
        energy = params.get('energy', known['energy'])
        dist_sa = params.get('sample_analyzer_distance', known['dist_sa'][variant])
        dist_ad = params.get('analyzer_detector_distance', known['dist_sa']['m'])
        sample = params.get('sample', vector([0, 0, 0], unit='m'))

        analyzer_vector = vector([1, 0, 0], unit='1') * dist_sa

        ks = (sqrt(energy * 2 * neutron_mass) / hbar).to(unit='1/angstrom')
        two_thetas = -2 * asin(0.5 * tau / ks)
        two_theta_vectors = two_thetas * vector([0, -1, 0])
        two_theta_rotation = rotations_from_rotvecs(two_theta_vectors)
        # Detector offsets are specified in a frame with x along the scattered beam, y in the plane of the analyzer
        detector_vector = two_theta_rotation * (vector([1, 0, 0], unit='1') * dist_ad + detector_offset)

        # Rotation of the whole analyzer channel around the vertical sample-table axis
        relative_rotation = rotations_from_rotvecs(relative_angle * vector([0, 0, 1], unit='1'))

        analyzer_position = sample + relative_rotation * analyzer_vector
        detector_position = sample + relative_rotation * (analyzer_vector + detector_vector)

        # The detector tube orientation rotation(s) must be modified by the channel rotation:
        detector_orient = relative_rotation * detector_orient

        # coverages = tan(min(ks) * atan(1.0*coverage) / ks)
        coverages = atan(min(ks) * tan(1.0 * coverage) / ks)

        per_det = 'analyzer' in detector_orient.dims
        pairs = []
        for idx, ap, dl, ct, cs, cc in enumerate(zip(
                analyzer_position, detector_length, blade_count, crystal_shape, coverages
        )):
            params = dict(sample=sample, blade_count=ct, shape=cs, analyzer_orient=relative_rotation, tau=tau,
                          coverage=cc, detector_orient=detector_orient['analyzer', idx] if per_det else detector_orient)
            pairs.append(Arm.from_calibration(ap, detector_position['analyzer', idx], dl, **params))

        return Channel(tuple(pairs))

    def triangulate_detectors(self, unit=None):
        from .spatial import combine_triangulations
        return combine_triangulations([arm.triangulate_detector(unit=unit) for arm in self.pairs])

    def triangulate_analyzers(self, unit=None):
        from .spatial import combine_triangulations
        return combine_triangulations([arm.triangulate_analyzer(unit=unit) for arm in self.pairs])

    def triangulate(self, unit=None):
        from .spatial import combine_triangulations
        return combine_triangulations([arm.triangulate(unit=unit) for arm in self.pairs])

    def extreme_path_edges(self, sample: Variable):
        from scipp import concat
        from numpy import vstack, hstack, cumsum
        ves = [pair.extreme_path_edges(sample) for pair in self.pairs]
        # ... deduplicate repeated sample position ...
        # offset the edge indexes to account for to-be-concatenated vertices
        offset = hstack((0, cumsum([len(v) for v, _ in ves])))[:-1]
        edges = vstack([e + o for (v, e), o in zip(ves, offset)])
        # concatenate the vertices
        vertices = concat([v for v, _ in ves], 'vertices')
        return vertices, edges


@dataclass
class Tank:
    channels: tuple[Channel, Channel, Channel, Channel, Channel, Channel, Channel, Channel, Channel]

    @staticmethod
    def from_calibration(**params):
        # by default the channels are ('l', 'sym', 's', 'l', 'sym', 's', 'l', 'sym', 's')
        channel_params = [{'variant': x} for x in ('l', 'm', 's')]
        channel_params = {i: channel_params[i % 3] for i in range(9)}
        # but this can be overridden by specifying an integer-keyed dictionary with the parameters for each channel
        channel_params = params.get('channel_params', channel_params)
        # The central a4 angle for each channel, relative to the reference tank angle
        angles = params.get('angles',
                            array(values=[-40, -30, -20, -10, 0, 10, 20, 30, 40.], unit='degree', dims=['channel']))

        channels = [Channel.from_calibration(angles[i], **channel_params[i]) for i in range(9)]
        return Tank(tuple(channels))

    def to_secondary(self, **params):
        sample_at = params.get('sample', vector([0, 0, 0.], unit='m'))

        detectors = []
        analyzers = []
        a_per_d = []
        for channel in self.channels:
            for analyzer, triplet in zip(channel.analyzers, channel.detectors):
                analyzers.append(analyzer.central_blade)
                detectors.extend(triplet.tubes)
                a_per_d.extend([len(analyzers) - 1 for _ in triplet.tubes])

        return IndirectSecondary(detectors, analyzers, a_per_d, sample_at)

    def triangulate_detectors(self, unit=None):
        from .spatial import combine_triangulations
        vts = [channel.triangulate_detectors(unit=unit) for channel in self.channels]
        return combine_triangulations(vts)

    def triangulate_analyzers(self, unit=None):
        from .spatial import combine_triangulations
        vts = [channel.triangulate_analyzers(unit=unit) for channel in self.channels]
        return combine_triangulations(vts)

    def triangulate(self, unit=None):
        from .spatial import combine_triangulations
        vts = [channel.triangulate(unit=unit) for channel in self.channels]
        return combine_triangulations(vts)

    def extreme_path_edges(self, sample: Variable):
        from scipp import concat
        from numpy import vstack, hstack, cumsum
        ves = [channel.extreme_path_edges(sample) for channel in self.channels]
        # ... deduplicate repeated sample position ...
        # offset the edge indexes to account for to-be-concatenated vertices
        offset = hstack((0, cumsum([len(v) for v, _ in ves])))[:-1]
        edges = vstack([e + o for (v, e), o in zip(ves, offset)])
        # concatenate the vertices
        vertices = concat([v for v, _ in ves], 'vertices')
        return vertices, edges

    def plot(self, unit=None):
        from meshplot import plot
        from numpy import array
        vdet, fdet = self.triangulate_detectors(unit=unit)
        p = plot(vdet.values, array(fdet))
        vana, fana = self.triangulate_analyzers(unit=unit)
        p.add_mesh(vana.values, array(fana))
