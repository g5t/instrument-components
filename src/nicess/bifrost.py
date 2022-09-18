from __future__ import annotations
from dataclasses import dataclass

from scipp import Variable, vector, scalar, array, vectors

from .secondary import IndirectSecondary
from .detectors import He3Tube
from .crystals import Crystal


def __is__scipp_vector__(v: Variable, name: str):
    from scipp import DType
    if v.dtype != DType.vector3:
        raise RuntimeError(f"The {name} must be a scipp.DType('vector3')")


def __is_type__(x, t, name):
    if not isinstance(x, t):
        raise RuntimeError(f"{name} must be a {t}")


@dataclass
class MinimalBIFROST:
    # primary: BandwidthPrimary   # A minimal form of the primary spectrometer necessary to transform events
    secondary: IndirectSecondary  # A minimal form of the secondary spectrometer necessary to transform events


@dataclass
class Triplet:
    tubes: tuple[He3Tube, He3Tube, He3Tube]

    @staticmethod
    def from_calibration(position: Variable, length: Variable, spacing: Variable, **params):
        """Take (fitting) calibration data and construct the object used to convert events to (Q,E)"""
        # The current crop of inputs is not sufficient to capture all degrees of freedom, but is a start.
        from scipp import sqrt, dot
        map(lambda x: __is__scipp_vector__(*x), ((position, 'position'), (length, 'length'), (spacing, 'spacing')))
        pressure = params.get('pressure', scalar(1., unit='atm'))
        radius = params.get('radius', sqrt(dot(spacing, spacing)) / 2)
        elements = params.get('elements', 100)
        map(lambda x: __is_type__(*x), ((pressure, Variable, 'pressure'),
                                        (radius, Variable, 'radius'), (elements, int, 'elements')))
        # pack the tube parameters
        pack = elements, radius.value, pressure.value
        # make sure all vectors are in the same length unit:
        l = 0.5 * length.to(unit=position.unit)
        s = spacing.to(unit=position.unit)
        # tube order: (-spacing, 0, +spacing); each tube from p-l/2 to p+l/2
        tubes = [He3Tube(position - l + m * s, position + l + m * s, *pack) for m in (-1, 0, 1)]
        return Triplet(tuple(tubes))


@dataclass
class Analyzer:
    blades: tuple[Crystal, ...]  # 7-9 blades

    @property
    def central_blade(self):
        return self.blades[len(self.blades) >> 1]

    @staticmethod
    def from_calibration(position: Variable, focus: Variable, **params):
        from math import pi
        from scipp import sqrt, dot, acos, sin, atan2
        from .rowland import rowland_blades
        map(lambda x: __is__scipp_vector__(*x), ((position, 'position'), (focus, 'focus')))
        count = params.get('count', 9)  # most analyzers have 9 blades
        width = params.get('width', scalar(10., unit='mm'))
        length = params.get('length', scalar(200., unit='mm'))
        depth = params.get('depth', params.get('thickness', scalar(2., unit='mm')))
        tau = params.get('tau', 2 * pi / params.get('dspacing', scalar(3.355, unit='angstrom')))  # PG(002)
        #qin_coverage = params.get('qin_coverage', params.get('coverage', scalar(0.1, unit='1/angstrom')))
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
        positions, taus = rowland_blades(source, position, focus, alpha, width, count)
        taus *= tau  # convert from directions to full tau vectors

        edges = [vector(x, unit='1') * s for x, s in (([1., 0, 0], width), ([0., 1, 0], length), ([0., 0, 1], depth))]
        shape = edges[0] + edges[1] + edges[2]
        blades = [Crystal(p, t, shape) for p, t in zip(positions, taus)]
        return Analyzer(tuple(blades))


@dataclass
class Channel:
    analyzers: tuple[Analyzer, Analyzer, Analyzer, Analyzer, Analyzer]
    detectors: tuple[Triplet, Triplet, Triplet, Triplet, Triplet]

    @staticmethod
    def from_calibration(relative_angle: Variable, **params):
        from math import pi
        from scipp import sqrt, dot, acos, sin, tan, atan, atan2, asin, min, max
        from scipp.constants import hbar, neutron_mass
        from scipp.spatial import rotations_from_rotvecs
        known_dists_sa = {
            'short': array(values=[1.100, 1.238, 1.342, 1.433, 1.544], unit='m', dims=['analyzer']),
            'symmetric': array(values=[1.189, 1.316, 1.420, 1.521, 1.623], unit='m', dims=['analyzer']),
            'long': array(values=[1.276, 1.388, 1.493, 1.595, 1.697], unit='m', dims=['analyzer']),
        }
        known_d_lengths = {
            'short': vectors(values=[[0,217.9,0], [0,242.0,0], [0,260.8,0], [0,279.2,0], [0,298.8,0]], unit='mm', dims=['analyzer']),
            'symmetric': vectors(values=[[0,226.0,0], [0,249.0,0], [0,267.9,0], [0,286.3,0], [0,304.8,0]], unit='mm', dims=['analyzer']),
            'long': vectors(values=[[0,233.9,0], [0,255.9,0], [0,274.9,0], [0,293.4,0], [0,311.9,0]], unit='mm', dims=['analyzer']),
        }
        known_a_shapes = {
            'short': vectors(values=[[0,134.0,0], [0,147.1,0], [0,156.2,0], [0,165.2,0], [0,175.6,0]], unit='mm', dims=['analyzer']),
            'symmetric': vectors(values=[[0,142.0,0], [0,154.1,0], [0,163.2,0], [0,172.3,0], [0,181.6,0]], unit='mm', dims=['analyzer']),
            'long': vectors(values=[[0,149.9,0], [0,161.0,0], [0,170.2,0], [0,179.3,0], [0,188.6,0]], unit='mm', dims=['analyzer']),
        }
        known_a_lengths = {
            'short': array(values=[134.0, 147.1, 156.2, 165.2, 175.6], unit='mm', dims=['analyzer']),
            'symmetric': array(values=[142.0, 154.1, 163.2, 172.3, 181.6], unit='mm', dims=['analyzer']),
            'long': array(values=[149.9, 161.0, 170.2, 179.3, 188.6], unit='mm', dims=['analyzer']),
        }
        known_a_widths = {
            'short': array(values=[12.0, 14.0, 11.5, 12.0, 13.5], unit='mm', dims=['analyzer']),
            'symmetric': array(values=[12.5, 14.5, 11.5, 12.5, 13.5], unit='mm', dims=['analyzer']),
            'long': array(values=[13.5, 15.0, 12.0, 13.0, 14.0], unit='mm', dims=['analyzer']),
        }
        known_a_depths = {
            'short': array(values=[2, 2, 2, 2, 2.0], unit='mm', dims=['analyzer']),
            'symmetric': array(values=[2, 2, 2, 2, 2.0], unit='mm', dims=['analyzer']),
            'long': array(values=[2, 2, 2, 2, 2.0], unit='mm', dims=['analyzer']),
        }
        known_d_spacings = {
            'short': vectors(values=[[20.,0,0], [20.,0,0], [20.,0,0], [20.,0,0], [20.,0,0]], unit='mm', dims=['analyzer']),
            'symmetric': vectors(values=[[20.,0,0], [20.,0,0], [20.,0,0], [20.,0,0], [20.,0,0]], unit='mm', dims=['analyzer']),
            'long': vectors(values=[[20.,0,0], [20.,0,0], [20.,0,0], [20.,0,0], [20.,0,0]], unit='mm', dims=['analyzer']),
        }
        counts = params.get('counts', [9, 9, 9, 7, 7])
        tau = params.get('tau', 2 * pi / params.get('dspacing', scalar(3.355, unit='angstrom')))  # PG(002)
        crystal_lengths = params.get('crystal_lengths', known_a_lengths[params.get('variant', 'symmetric')])
        crystal_widths = params.get('crystal_widths', known_a_widths[params.get('variant', 'symmetric')])
        crystal_depths = params.get('crystal_depths', known_a_depths[params.get('variant', 'symmetric')])
        detector_lengths = params.get('detector_lengths', known_d_lengths[params.get('variant', 'symmetric')])
        detector_spacings = params.get('detector_spacings', known_d_spacings[params.get('variant', 'symmetric')])
        coverage = params.get('coverage', scalar(2., unit='degree'))
        energies = params.get('energies', array(values=[2.7, 3.2, 3.7, 4.4, 5.], unit='meV', dims=['analyzer']))
        dists_sa = params.get('sample_analyzer_distances', known_dists_sa[params.get('variant', 'symmetric')])
        dists_ad = params.get('analyzer_detector_distances', known_dists_sa['symmetric'])
        sample = params.get('sample', vector([0, 0, 0], unit='m'))

        analyzer_vectors = vector([1, 0, 0], unit='1') * dists_sa

        ks = (sqrt(energies * 2 * neutron_mass) / hbar).to(unit='1/angstrom')
        two_thetas = -2 * asin(0.5 * tau / ks)
        rotations = rotations_from_rotvecs(rotation_vectors=two_thetas * vector([0, 0, 1], unit='1'))
        detector_vectors = rotations * (vector([1, 0, 0], unit='1') * dists_ad)

        relative_rotation = rotations_from_rotvecs(rotation_vectors=relative_angle * vector([0, 1, 0], unit='1'))

        analyzer_positions = sample + relative_rotation * analyzer_vectors
        detector_positions = sample + relative_rotation * (analyzer_vectors + detector_vectors)

        # coverages = tan(min(ks) * atan(1.0*coverage) / ks)
        coverages = atan(min(ks) * tan(1.0*coverage) / ks)

        analyzers = []
        detectors = []
        for ap, dp, dl, ds, ct, cl, cw, cd, cc in zip(analyzer_positions, detector_positions, detector_lengths, detector_spacings, counts, crystal_lengths, crystal_widths, crystal_depths, coverages):
            params = dict(sample=sample, count=ct, width=cw, length=cl, depth=cd, tau=tau, coverage=cc)
            analyzers.append(Analyzer.from_calibration(ap, dp, **params))
            detectors.append(Triplet.from_calibration(dp, dl, ds))

        return Channel(analyzers, detectors)


@dataclass
class Tank:
    channels: tuple[Channel, Channel, Channel, Channel, Channel, Channel, Channel, Channel, Channel]

    @staticmethod
    def from_calibration(**params):
        # by default the channels are ('long', 'sym', 'short', 'long', 'sym', 'short', 'long', 'sym', 'short')
        channel_params = [{'variant': x} for x in ('long', 'symmetric', 'short')]
        channel_params = {i: channel_params[i%3] for i in range(9)}
        # but this can be overridden by specifying an integer-keyed dictionary with the parameters for each channel
        channel_params = params.get('channel_params', channel_params)
        # The central a4 angle for each channel, relative to the reference tank angle
        angles = params.get('angles', array(values=[-40,-30,-20,-10,  0, 10, 20, 30, 40.], unit='degree', dims=['channel']))

        channels = [Channel(angles[i], **channel_parmas[i]) for i in range(9)]
        return Tank(tuple(channels))


