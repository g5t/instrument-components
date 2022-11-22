from __future__ import annotations
from dataclasses import dataclass

from scipp import Variable, vector, scalar, array, vectors

from .secondary import IndirectSecondary
from .detectors import He3Tube
from .crystals import Crystal


def __is_type__(x, t, name):
    if not isinstance(x, t):
        raise RuntimeError(f"{name} must be a {t}")


def variant_parameters(params: dict, default: dict):
    variant = params.get('variant', default['variant'])
    complete = {k: params.get(k, v[variant] if isinstance(v, dict) else v) for k, v in default.items()}
    return complete


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
    resistances: Variable

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
        resistivity = params.get('resistivity', scalar(140., unit='Ohm/in').to(unit='Ohm/m'))
        map(lambda x: __is_type__(*x), ((pressure, Variable, 'pressure'), (length, Variable, 'length'),
                                        (radius, Variable, 'radius'), (elements, int, 'elements'),
                                        (resistivity, Variable, 'resistivity')))
        # pack the tube parameters
        pack = elements, radius, pressure

        # ensure that there is one resistivity per tube
        from .utilities import is_scalar
        from scipp import concat
        if is_scalar(resistivity):
            resistivity = concat((resistivity, resistivity, resistivity), dim='tube')

        # Make the oriented tube-axis vector(s)
        axis = ori * (length.to(unit=position.unit) * vector([0, 1., 0]))  # may be a 0-D or 1-D (tube) vector array
        tube_at = position - 0.5 * axis  # should now be a 1-D (tube) vector array
        tube_to = position + 0.5 * axis  # *ditto*
        tubes = (He3Tube(at, to, rho, *pack) for at, to, rho in zip(tube_at, tube_to, resistivity))

        # Define the ex-Tube resistances
        resistance = params.get('resistance', scalar(2, unit='Ohm'))
        if is_scalar(resistance):
            resistance = concat((resistance, resistance), dim='tube')
        if len(resistance) < 3:
            contact = params.get('contact_resistance', scalar(0, unit='Ohm'))
            resistance = concat((contact, resistance, contact), dim='tube')
        for idx, name in enumerate(('resistance_A', 'resistance_01', 'resistance_12', 'resistance_B')):
            if name in params:
                resistance['tube', idx] = params.get(name)

        return Triplet(tuple(tubes), resistance)

    def triangulate(self, unit=None):
        from .spatial import combine_triangulations
        vts = [tube.triangulate(unit=unit) for tube in self.tubes]
        return combine_triangulations(vts)

    def extreme_path_corners(self, horizontal, vertical, unit=None):
        from .spatial import combine_extremes
        vs = [tube.extreme_path_corners(horizontal, vertical, unit=unit) for tube in self.tubes]
        return combine_extremes(vs, horizontal, vertical)

    def mcstas_parameters(self):
        from numpy import vstack
        return vstack([tube.mcstas_parameters for tube in self.tubes])

    def tube_com(self):
        from scipp import concat
        return concat([(t.at + t.to)/2 for t in self.tubes], 'tube')

    def tube_end(self):
        from scipp import concat
        return concat([(t.to - t.at)/2 for t in self.tubes], 'tube')

    def to_cadquery(self, unit=None):
        from .spatial import combine_assembly
        t = {k: tube.to_cadquery(unit=unit) for k, tube in zip(("tube-0", "tube-1", "tube-2"), self.tubes)}
        return combine_assembly(**t)

    def a_over_a_plus_b_edges(self):
        """Points to convert continuous A/(A+B) to discrete segments per tube"""
        from scipp import concat, scalar, cumsum, max
        tr = [1.0 * t.resistance for t in self.tubes]
        rs = [scalar(0., unit='Ohm'), *[x for a in zip(self.resistances, tr) for x in a], self.resistances[-1]]
        # rs is [0, left_contact, left_tube, left_resistor, center_tube, right_resistor, right_tube, right_contact]
        partial_sums = cumsum(concat(rs, dim='tube'))
        return partial_sums / max(partial_sums)

    def a_minus_b_over_a_plus_b_edges(self):
        """Points to convert continuous (A-B)/(A+B) to discrete segments per tube"""
        from scipp import scalar
        return 2 * self.a_over_a_plus_b_edges() - scalar(1)


@dataclass
class Analyzer:
    blades: tuple[Crystal, ...]  # 7-9 blades

    @property
    def central_blade(self):
        return self.blades[len(self.blades) >> 1]

    @property
    def count(self):
        return len(self.blades)

    @staticmethod
    def from_calibration(position: Variable, focus: Variable, tau: Variable, **params):
        from math import pi
        from scipp.spatial import rotation
        from .spatial import is_scipp_vector
        from .rowland import rowland_blades
        map(lambda x: is_scipp_vector(*x), ((position, 'position'), (focus, 'focus'), (tau, 'tau')))
        count = params.get('blade_count', scalar(9))  # most analyzers have 9 blades
        shape = params.get('shape', vector([10., 200., 2.], unit='mm'))
        orient = params.get('orient', None)
        orient = rotation(value=[0, 0, 0, 1.]) if orient is None else orient
        # qin_coverage = params.get('qin_coverage', params.get('coverage', scalar(0.1, unit='1/angstrom')))
        coverage = params.get('coverage', scalar(2, unit='degree'))
        source = params.get('source', params.get('sample_position', vector([0, 0, 0], unit='m')))
        gap = params.get('gap', None)
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
        # Use the Rowland geometry to define each blade position & normal direction
        positions, taus = rowland_blades(source, position, focus, coverage, shape.fields.x, count.value, tau, gap)

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

    def mcstas_parameters(self):
        from numpy import hstack
        return hstack((len(self.blades), self.central_blade.mcstas_parameters))

    def to_cadquery(self, unit=None):
        from .spatial import combine_assembly
        b = {f'blade-{i}': blade.to_cadquery(unit=unit) for i, blade in enumerate(self.blades)}
        return combine_assembly(**b)

    def coverage(self, sample: Variable):
        from scipp import sqrt, dot, cross, max, min, atan2
        # Define a pseudo McStas coordinate system (requiring y is mostly vertical)
        z = (self.central_blade.position - sample)
        sa_dist = sqrt(dot(z, z))
        z = z / sa_dist
        y = vector([0, 1.0, 0])
        y = cross(cross(z, y), z)  # in case y is not perpendicular to z
        y = y / sqrt(dot(y, y))
        x = cross(y, z)
        x = x / sqrt(dot(x, x))

        # Define the horizontal (along x) and vertical (along y) extreme points of the array
        xtr = self.extreme_path_corners(x, y)
        coverages = [atan2(y=(max(dot(xtr, w)) - min(dot(xtr, w))) / 2., x=sa_dist).to(unit='radian') for w in (x, y)]
        return tuple(coverages)

    def sample_space_angle(self, sample: Variable):
        from scipp import dot, atan2
        z = (self.central_blade.position - sample)
        sample_space_x = vector([1, 0, 0])
        sample_space_y = vector([0, 1, 0])
        return atan2(y=dot(sample_space_y, z), x=dot(sample_space_x, z)).to(unit='radian')

    def rtp_parameters(self, sample: Variable, oop: Variable):
        from scipp import concat
        p0 = self.central_blade.position
        # exploit that for x in zip returns first all the first elements, then all the second elements, etc.
        x, y, a = [concat(x, dim='blades') for x in zip(*[b.rtp_parameters(sample, p0, oop) for b in self.blades])]
        return x, y, a

@dataclass
class Arm:
    analyzer: Analyzer
    detector: Triplet

    @staticmethod
    def from_calibration(a_position, tau, d_position, d_length, **params):
        analyzer_orient = params.get('analyzer_orient', None)
        detector_orient = params.get('detector_orient', None)
        # the analyzer focuses on the center tube of the triplet
        a_focus = d_position['tube', 1] if 'tube' in d_position.dims else d_position
        analyzer = Analyzer.from_calibration(a_position, a_focus, tau, **params, orient=analyzer_orient)
        detector = Triplet.from_calibration(d_position, d_length, **params, orient=detector_orient)
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

    def mcstas_parameters(self, sample: Variable):
        from numpy import stack, hstack
        from scipp import sqrt, dot, cross, vector, acos
        from .spatial import is_scipp_vector, perpendicular_directions
        is_scipp_vector(sample, 'sample')

        # TODO find sample-analyzer and analyzer-detector distances, move positions into appropriate frames
        # analyzer_position -> [0, 0, sample-analyzer-distance]
        # detector_position -> [[dx0, dy0, dz0], [dx1, dy1, analyzer-detector-distance], [dx2, dy2, dz2]]
        # end_position -> z along analyzer-detector vector 'Arm' (in McStas local coordinate frame)

        sa = self.analyzer.central_blade.position - sample
        ad = (self.detector.tubes[1].at + self.detector.tubes[1].to)/2 - self.analyzer.central_blade.position
        distances = [sqrt(dot(x, x)).to(unit='m').value for x in (sa, ad)]
        # the coordinate system here has 'local' x along the beam, and z vertical
        # the McStas local cooridnate system always has z along the beam and y defines the local scattering plane normal
        # for BIFROST's analyzers, the two coordinate systems have parallel (or maybe antiparallel) y directions

        za = sa / sqrt(dot(sa, sa))
        zd = ad / sqrt(dot(ad, ad))
        yd = cross(za, zd)
        yd /= sqrt(dot(yd, yd))
        xd = cross(yd, zd)

        two_theta = acos(dot(za, zd))

        tube_com = self.detector.tube_com() - self.analyzer.central_blade.position
        tube_end = self.detector.tube_end()

        x, y, z = [vector(q) for q in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]

        # this could be simplified if we built the column matrix (xd, yd, zd)
        tube_com_x, tube_com_y, tube_com_z = [dot(tube_com, d) * i for d, i in zip((xd, yd, zd), (x, y, z))]
        tube_com_d = tube_com_x + tube_com_y + tube_com_z
        tube_end_x, tube_end_y, tube_end_z = [dot(tube_end, d) * i for d, i in zip((xd, yd, zd), (x, y, z))]
        tube_end_d = tube_end_x + tube_end_y + tube_end_z
        # shift the COM relative to the expected detector position
        tube_com_d.fields.z -= sqrt(dot(ad, ad))

        # this is not good. Can we verify which axis is the coordinate axis and which is the tube axis?
        d = stack((tube_com_d.to(unit='m').values, tube_end_d.to(unit='m').values), axis=1)

        hc, vc = self.analyzer.coverage(sample)
        a = hstack((self.analyzer.count, self.analyzer.central_blade.shape.to(unit='m').value, [hc.value, vc.value]))

        return {'distances': distances, 'analyzer': a, 'detector': d, 'two_theta': two_theta.value}

    def rtp_parameters(self, sample: Variable):
        from scipp import concat, cross, dot, sqrt
        sa = self.analyzer.central_blade.position - sample
        ad = (self.detector.tubes[1].at + self.detector.tubes[1].to)/2 - self.analyzer.central_blade.position

        out_of_plane = cross(ad, sa)
        x, y, angle = self.analyzer.rtp_parameters(sample, out_of_plane)
        return sqrt(dot(sa, sa)), sqrt(dot(ad, ad)), x, y, angle

    def to_cadquery(self, unit=None):
        from .spatial import combine_assembly
        if unit is None:
            unit = 'mm'
        a = self.analyzer.to_cadquery(unit=unit)
        d = self.detector.to_cadquery(unit=unit)
        # combine a and d into an Assembly?
        return combine_assembly(analyzer=a, detector=d)

    def sample_space_angle(self, sample: Variable):
        return self.analyzer.sample_space_angle(sample)


def known_channel_params():
    known = dict()
    dist_sa = {
        's': [1.100, 1.238, 1.342, 1.443, 1.557],
        'm': [1.189, 1.316, 1.420, 1.521, 1.623],
        'l': [1.276, 1.392, 1.497, 1.599, 1.701],
    }
    known['sample_analyzer_distance'] = {k: array(values=v, unit='m', dims=['analyzer']) for k, v in dist_sa.items()}
    known['analyzer_detector_distance'] = known['sample_analyzer_distance']['m']
    d_length_mm = {
        's': [217.9, 242.0, 260.8, 279.2, 298.8],
        'm': [226.0, 249.0, 267.9, 286.3, 304.8],
        'l': [233.9, 255.9, 274.9, 293.4, 311.9],
    }
    dex = scalar(10, unit='mm')  # The detector tubes were ordered with 10 mm extra length buffer
    known['detector_length'] = {k: dex + array(values=v, unit='mm', dims=['analyzer']) for k, v in d_length_mm.items()}
    known['detector_offset'] = vectors(values=[[0, 0, -14.], [0, 0, 0], [0, 0, 14]], unit='mm', dims=['tube'])
    known['detector_orient'] = vector([0, 0, 0], unit='mm')
    a_shape_mm = {
        's': [[12.0, 134, 1], [14.0, 147, 1], [11.5, 156, 1], [12.0, 165, 1], [13.5, 177, 1]],
        'm': [[12.5, 144, 1], [14.5, 156, 1], [11.5, 165, 1], [12.5, 174, 1], [13.5, 183, 1]],
        'l': [[13.5, 150, 1], [15.0, 162, 1], [12.0, 171, 1], [13.0, 180, 1], [14.0, 189, 1]],
    }
    known['crystal_shape'] = {k: vectors(values=v, unit='mm', dims=['analyzer']) for k, v in a_shape_mm.items()}
    known['blade_count'] = array(values=[7, 7, 9, 9, 9], dims=['analyzer'])  # two lowest energy analyzer have 7 blades
    known['d_spacing'] = scalar(3.355, unit='angstrom')  # PG(002)
    known['coverage'] = scalar(2., unit='degree')
    known['energy'] = array(values=[2.7, 3.2, 3.8, 4.4, 5.], unit='meV', dims=['analyzer'])
    known['sample'] = vector([0, 0, 0.], unit='m')
    known['gap'] = array(values=[2, 2, 2, 2, 2.], unit='mm', dims=['analyzer'])
    known['variant'] = 'm'

    known['resistance'] = scalar(380., unit='Ohm')
    known['contact_resistance'] = scalar(0., unit='Ohm')
    known['resistivity'] = scalar(200., unit='Ohm/in').to(unit='Ohm/m')

    return known


def tube_xz_displacement_to_quaternion(length: Variable, displacement: Variable):
    from scipp import vector, scalar, any, sqrt, allclose
    from .spatial import vector_to_vector_quaternion
    com_to_end = length * vector([0, 0.5, 0]) + displacement
    l2 = length * length
    x2 = displacement.fields.x * displacement.fields.x
    z2 = displacement.fields.z * displacement.fields.z

    com_to_end.fields.y = sqrt(0.25 * l2 - x2 - z2)

    y2 = displacement.fields.y * displacement.fields.y
    if any(y2 > scalar(0, unit=y2.unit)) and not allclose(com_to_end.fields.y, 0.5 * length - displacement.fields.y):
        raise RuntimeError("Provided tube-end displacement vector(s) contain wrong y-component value(s)")

    # The tube *should* point along y, but we were told it is displaced in x and z;
    # return the orienting Quaternion that takes (010) to the actual orientation
    quaternion = vector_to_vector_quaternion(vector([0, 1, 0]), com_to_end)
    return quaternion



@dataclass
class Channel:
    pairs: tuple[Arm, Arm, Arm, Arm, Arm]

    @staticmethod
    def from_calibration(relative_angle: Variable, **params):
        from math import pi
        from scipp import sqrt, tan, atan, asin, min
        from scipp.constants import hbar, neutron_mass
        from scipp.spatial import rotations_from_rotvecs


        vp = variant_parameters(params, known_channel_params())

        tau = params.get('tau', 2 * pi / vp['d_spacing'])

        sample = vp['sample']

        analyzer_vector = vector([1, 0, 0]) * vp['sample_analyzer_distance']

        ks = (sqrt(vp['energy'] * 2 * neutron_mass) / hbar).to(unit='1/angstrom')
        two_thetas = -2 * asin(0.5 * tau / ks)
        two_theta_vectors = two_thetas * vector([0, -1, 0])
        two_theta_rotation = rotations_from_rotvecs(two_theta_vectors)

        # Detector offsets are specified in a frame with x along the scattered beam, y in the plane of the analyzer
        add = 'analyzer_detector_distance'
        detector_vector = vector([1, 0, 0]) * vp[add] + vp['detector_offset'].to(unit=vp[add].unit)

        # Rotation of the whole analyzer channel around the vertical sample-table axis
        relative_rotation = rotations_from_rotvecs(relative_angle * vector([0, 0, 1]))

        analyzer_position = sample + relative_rotation * analyzer_vector
        detector_position = sample + relative_rotation * (analyzer_vector + two_theta_rotation * detector_vector)

        tau_vecs = relative_rotation * rotations_from_rotvecs(0.5 * two_theta_vectors) * (tau * vector([0, 0, -1]))

        # The detector orientation is given by a displacement vector of the tube-end, we want the associated quaternion
        detector_orient = tube_xz_displacement_to_quaternion(vp['detector_length'], vp['detector_orient'])

        # The detector tube orientation rotation(s) must be modified by the channel rotation:
        detector_orient = relative_rotation * detector_orient

        # coverages = tan(min(ks) * atan(1.0*coverage) / ks)
        coverages = atan(min(ks) * tan(1.0 * vp['coverage']) / ks)

        # print(f"Vertical coverage = {coverages.to(unit='degree'): c}")

        resistance = vp['resistance']
        resistivity = vp['resistivity']
        from .utilities import is_scalar
        from scipp import concat
        if is_scalar(resistance):
            contact_resistance = vp['contact_resistance']
            resistance = concat((contact_resistance, resistance, resistance, contact_resistance), dim='tube')
        if is_scalar(resistivity):
            resistivity = concat((resistivity, resistivity, resistivity), dim='tube')

        orient_per, resistance_per, resistivity_per = ['analyzer' in x.dims for x in
                                                       (detector_orient, resistance, resistivity)]
        pairs = []
        for idx, (ap, tv, dl, ct, cs, cc, gp) in enumerate(zip(
                analyzer_position, tau_vecs, vp['detector_length'], vp['blade_count'], vp['crystal_shape'], coverages,
                vp['gap']
        )):
            params = dict(sample=sample, blade_count=ct, shape=cs, analyzer_orient=relative_rotation, coverage=cc,
                          detector_orient=detector_orient['analyzer', idx] if orient_per else detector_orient,
                          resistance=resistance['analyzer', idx] if resistance_per else resistance,
                          resistivity=resistivity['analyzer', idx] if resistivity_per else resistivity,
                          gap=gp
                          )
            pairs.append(Arm.from_calibration(ap, tv, detector_position['analyzer', idx], dl, **params))

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

    def mcstas_parameters(self, sample: Variable):
        from numpy import stack
        parameters = [arm.mcstas_parameters(sample) for arm in self.pairs]
        distances = stack([p['distances'] for p in parameters], axis=0)  # (5 ,2)
        analyzers = stack([p['analyzer'] for p in parameters], axis=0)  # (5, 6)
        detectors = stack([p['detector'] for p in parameters], axis=0)  # (5, 3, 2, 3)
        two_theta = stack([p['two_theta'] for p in parameters], axis=0)  # (5, )
        return {'distances': distances, 'analyzer': analyzers, 'detector': detectors, 'two_theta': two_theta}

    def to_cadquery(self, unit=None):
        from cadquery import Assembly, Color
        d_colors = 'tan', 'tan1', 'tan2', 'tan3', 'tan4'
        assembly = Assembly()
        for index, (arm, c) in enumerate(zip(self.pairs, d_colors)):
            d = arm.to_cadquery(unit=unit)
            assembly = assembly.add(d, name=f"pair-{index}", color=Color(c))
        return assembly.toCompound()

    def sample_space_angle(self, sample: Variable):
        return self.pairs[0].sample_space_angle(sample)

    def rtp_parameters(self, sample: Variable):
        from scipp import concat, all, isclose
        sa, ad, x, y, angle = zip(*[p.rtp_parameters(sample) for p in self.pairs])
        sa = concat(sa, dim='pairs')
        ad = concat(ad, dim='pairs')
        x7, y7, a7 = [concat(q[:2], dim='pairs') for q in (x, y, angle)]
        x9, y9, a9 = [concat(q[2:], dim='pairs') for q in (x, y, angle)]

        relative_angles = [arm.sample_space_angle(sample) for arm in self.pairs]
        ra0 = relative_angles[0]
        if not all(isclose(concat(relative_angles, dim='arm'), ra0)):
            raise RuntimeError("different relative angles for same-channel analyzers?!")

        return sa, ad, x7, y7, a7, x9, y9, a9, ra0


@dataclass
class Tank:
    channels: tuple[Channel, Channel, Channel, Channel, Channel, Channel, Channel, Channel, Channel]

    @staticmethod
    def from_calibration(**params):
        channel_params = [{'variant': x} for x in ('s', 'm', 'l')]
        channel_params = {i: channel_params[i % 3] for i in range(9)}
        # but this can be overridden by specifying an integer-keyed dictionary with the parameters for each channel
        channel_params = params.get('channel_params', channel_params)
        # The central a4 angle for each channel, relative to the reference tank angle
        angles = params.get('angles',
                            array(values=[-40, -30, -20, -10, 0, 10, 20, 30, 40.], unit='degree', dims=['channel']))

        channels = [Channel.from_calibration(angles[i], **channel_params[i]) for i in range(9)]
        return Tank(tuple(channels))

    @staticmethod
    def unique_from_calibration(**params):
        channel_params = [{'variant': x} for x in ('s', 'm', 'l')]
        channel_params = {i: channel_params[i % 3] for i in range(3)}
        # but this can be overridden by specifying an integer-keyed dictionary with the parameters for each channel
        channel_params = params.get('channel_params', channel_params)
        # The central a4 angle for each channel, relative to the reference tank angle
        angles = params.get('angles',
                            array(values=[-40, -30, -20, -10, 0, 10, 20, 30, 40.], unit='degree', dims=['channel']))

        channels = [Channel.from_calibration(angles[i], **channel_params[i]) for i in range(3)]
        return Tank(tuple(channels))

    def to_secondary(self, **params):
        sample_at = params.get('sample', vector([0, 0, 0.], unit='m'))

        detectors = []
        analyzers = []
        a_per_d = []
        for channel in self.channels:
            for arm in channel.pairs:
                analyzers.append(arm.analyzer.central_blade)
                detectors.extend(arm.detector.tubes)
                a_per_d.extend([len(analyzers) - 1 for _ in arm.detector.tubes])

        from scipp import arange
        nc = len(self.channels)
        np = len(self.channels[0].pairs)
        a = arange(start=0, stop=len(analyzers), dim='n').fold('n', sizes={'channel': nc, 'pair': np})
        d = arange(start=0, stop=len(detectors), dim='n').fold('n', sizes={'channel': nc, 'pair': np, 'tube': 3})

        return IndirectSecondary(detectors, analyzers, a_per_d, sample_at, a, d)

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

    def mcstas_parameters(self, sample: Variable):
        from numpy import stack, hstack
        parameters = [channel.mcstas_parameters(sample) for channel in self.channels]
        y = stack([p['distances'] for p in parameters], axis=0)  # (9, 5, 2)
        a = stack([p['analyzer'] for p in parameters], axis=0)  # (9, 5, 6)
        d = stack([p['detector'] for p in parameters], axis=0)  # (9, 5, 3, 2, 3)
        t = stack([p['two_theta'] for p in parameters], axis=0)  # (9, 5)
        s = hstack([channel.sample_space_angle(sample).value for channel in self.channels])
        return {'distances': y, 'analyzer': a, 'detector': d, 'channel': s, 'two_theta': t}

    def to_cadquery(self, unit=None, add_sphere_at_origin=False):
        from cadquery import Assembly
        if unit is None:
            unit = 'mm'
        assembly = Assembly()
        for index, channel in enumerate(self.channels):
            assembly = assembly.add(channel.to_cadquery(unit=unit), name=f"channel-{index}")

        if add_sphere_at_origin:
            from cadquery import Workplane
            w = Workplane().sphere(radius=10)
            assembly.add(w, name='origin')

        assembly.name = 'BIFROST-secondary'

        return assembly

    def rtp_parameters(self, sample: Variable):
        from scipp import concat
        return [concat(q, dim='channel') for q in zip(*[c.rtp_parameters(sample) for c in self.channels])]
