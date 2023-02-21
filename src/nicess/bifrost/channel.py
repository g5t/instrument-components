from dataclasses import dataclass


def variant_parameters(params: dict, default: dict):
    variant = params.get('variant', default['variant'])
    complete = {k: params.get(k, v[variant] if isinstance(v, dict) else v) for k, v in default.items()}
    return complete


@dataclass
class Channel:
    from mcstasscript.interface.instr import McStas_instr as ScriptInstrument
    from mcstasscript.helper.mcstas_objects import Component as ScriptComponent
    from scipp import Variable
    from .arm import Arm

    pairs: tuple[Arm, Arm, Arm, Arm, Arm]

    @staticmethod
    def from_calibration(relative_angle: Variable, **params):
        from math import pi
        from scipp import sqrt, tan, atan, asin, min, vector
        from scipp.constants import hbar, neutron_mass
        from scipp.spatial import rotations_from_rotvecs
        from .parameters import known_channel_params, tube_xz_displacement_to_quaternion
        from .arm import Arm

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
        from ..utilities import is_scalar
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

        return Channel((pairs[0], pairs[1], pairs[2], pairs[3], pairs[4]))

    def triangulate_detectors(self, unit=None):
        from ..spatial import combine_triangulations
        return combine_triangulations([arm.triangulate_detector(unit=unit) for arm in self.pairs])

    def triangulate_analyzers(self, unit=None):
        from ..spatial import combine_triangulations
        return combine_triangulations([arm.triangulate_analyzer(unit=unit) for arm in self.pairs])

    def triangulate(self, unit=None):
        from ..spatial import combine_triangulations
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

    def coverage(self, sample: Variable):
        from scipp import concat, max
        cov_xy = [x.coverage(sample) for x in self.pairs]
        cov_x = max(concat([x for x, _ in cov_xy], dim='pairs'))
        cov_y = max(concat([y for _, y in cov_xy], dim='pairs'))
        return cov_x, cov_y

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

    def to_mcstasscript(self, inst: ScriptInstrument, relative: ScriptComponent,
                        name: str = None, when: str = None, settings: dict = None):
        from scipp import concat, all, isclose, vector
        from ..mcstasscript import ensure_user_var
        # For each channel we need to define the local coordinate system, relative to the provided sample
        origin = vector([0, 0, 0], unit='m')
        ra0 = self.sample_space_angle(origin).to(unit='degree').value
        cassette = inst.component("Arm", name=f"{name}_arm", RELATIVE=relative, ROTATED=[0, ra0, 0], WHEN=when)

        ensure_user_var(inst, 'int', 'secondary_scattered', 'Flag indicates if Bragg scattering has occurred')
        ensure_user_var(inst, 'int', 'analyzer', 'Flag indicates were Bragg scattering has occurred')
        ensure_user_var(inst, 'int', 'flag', 'Flag indicates detection in a monitor')

        for arm_index, arm in enumerate(self.pairs):
            arm_name = f"{name}_{1 + arm_index}"
            arm_when = f"0 == secondary_scattered && {when}"
            extend = f"secondary_scattered = (SCATTERED) ? 1 : 0;\nanalyzer = (SCATTERED) ? {1 + arm_index} : 0;"
            detector_when = f"{when} && {1 + arm_index}==analyzer"
            detector_extend = f"flag = (SCATTERED) ? 1 : 0;"
            arm.to_mcstasscript(inst, cassette, name=arm_name, analyzer_when=arm_when, analyzer_extend=extend,
                                settings=settings,
                                detector_when=detector_when, detector_extend=detector_extend)
