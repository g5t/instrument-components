from dataclasses import dataclass

@dataclass
class Arm:
    from mcstasscript.interface.instr import McStas_instr as ScriptInstrument
    from mcstasscript.helper.mcstas_objects import Component as ScriptComponent
    from .analyzer import Analyzer
    from .triplet import Triplet
    from scipp import Variable

    analyzer: Analyzer
    detector: Triplet

    @staticmethod
    def from_calibration(a_position, tau, d_position, d_length, **params):
        from .analyzer import Analyzer
        from .triplet import Triplet

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
        from ..spatial import combine_triangulations
        return combine_triangulations([self.triangulate_analyzer(unit=unit), self.triangulate_detector(unit=unit)])

    def extreme_path_edges(self, sample: Variable):
        from scipp import concat
        from numpy import array as numpy_array, int as numpy_int
        from ..spatial import is_scipp_vector, perpendicular_directions
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
        from ..spatial import is_scipp_vector, perpendicular_directions
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
        from ..spatial import combine_assembly
        if unit is None:
            unit = 'mm'
        a = self.analyzer.to_cadquery(unit=unit)
        d = self.detector.to_cadquery(unit=unit)
        # combine a and d into an Assembly?
        return combine_assembly(analyzer=a, detector=d)

    def sample_space_angle(self, sample: Variable):
        return self.analyzer.sample_space_angle(sample)

    def coverage(self, sample: Variable):
        return self.analyzer.coverage(sample)

    def mcstas_analyzer_parameters(self, sample: Variable, source: str, sink: str) -> dict:
        from ..spatial import is_scipp_vector
        is_scipp_vector(sample, 'sample')

        perp_q, perp_plane, parallel_q = self.analyzer.central_blade.shape.to(unit='m').value
        hor_cov, ver_cov = self.analyzer.coverage(sample)
        params = dict(NH=self.analyzer.count, zwidth=perp_q, yheight=perp_plane, mosaic='mosaic', DM=3.355,
                      gap=0.002, show_construction='showconstruction', angle_h=ver_cov.to(unit='degree').value,
                      source=f'"{source}"', sink=f'"{sink}"')
        return params

    def mcstas_detector_parameters(self, sample: Variable) -> dict:
        #TODO make this more accurate -- insert vectors into the instrument defined parameters to use here?
        from scipp import sqrt, dot
        lv = [self.detector.tubes[x].to - self.detector.tubes[x].at for x in range(3)]
        cv = [(self.detector.tubes[x].to + self.detector.tubes[x].at)/2 for x in range(3)]
        length = sum([sqrt(dot(x, x)).to(unit='m').value for x in lv]) / 3
        radius = sum([self.detector.tubes[x].radius.to(unit='m').value for x in range(3)]) / 3
        width = sqrt(dot(cv[2] - cv[0], cv[2] - cv[0])).to(unit='m').value + 2 * radius
        # print(f"detectors have width f{width} m")
        params = dict(charge_a='"event_charge_left"', charge_b='"event_charge_right"', detection_time='"event_time"',
                      tube_index_name='"TUBE"', N=3, width=width, height=length, radius=radius)
        return params

    def to_mcstasscript(self, inst: ScriptInstrument, relative: ScriptComponent, name: str = None,
                        analyzer_when: str = None, analyzer_extend: str = None,
                        detector_when: str = None, detector_extend: str = None,
                        **kwargs):
        from scipp import concat, all, isclose, vector, dot, sqrt, atan2
        # For each channel we need to define the local coordinate system, relative to the provided sample
        origin = vector([0, 0, 0], unit='m')
        # relative_angles = [arm.sample_space_angle(origin) for arm in self.pairs]
        # ra0 = relative_angles[0]
        # if not all(isclose(concat(relative_angles, dim='arm'), ra0)):
        #     raise RuntimeError("different relative angles for same-channel analyzers?!")
        # ra0 = ra0.to(unit='degree').value

        sa_vec = self.analyzer.central_blade.position
        ad_vec = (self.detector.tubes[1].at + self.detector.tubes[1].to) / 2 - sa_vec

        sample_analyzer_distance = sqrt(dot(sa_vec, sa_vec)).to(unit='m')
        analyzer_detector_distance = sqrt(dot(ad_vec, ad_vec)).to(unit='m')

        x = dot(ad_vec, sa_vec / sample_analyzer_distance)
        y = dot(ad_vec, vector([0, 0, 1]))
        two_theta = atan2(y=y, x=x).to(unit='degree').value
        theta = two_theta / 2

        detector_position_name = f"{name}_detector_point"
        # Move to the center of the analyzer & reorient for monochromator scattering in vertical plane
        # inst.component("Arm", name=f"{name}_analyzer_point", RELATIVE=relative,
        #                AT=[0, 0, sample_analyzer_distance.value])
        # inst.component("Arm", name=f"{name}_analyzer_coordinate_change", RELATIVE="PREVIOUS", ROTATED=[0, 0, 90])
        # # Add the Rowland monochromator
        # mono = inst.component("Monochromator_Rowland", name=f"{name}_monochromator",
        #                       RELATIVE="PREVIOUS", ROTATED=[0, theta, 0],
        #                       WHEN=analyzer_when, EXTEND=analyzer_extend)
        # mono.set_parameters(**self.mcstas_analyzer_parameters(origin, relative.name, detector_position_name))
        # # Add the detector tubes
        # inst.component("Arm", name=f"{name}_detector_angle", RELATIVE="PREVIOUS",
        #                ROTATED=[0, theta, 0], ROTATED_RELATIVE=mono, WHEN=detector_when)
        # inst.component("Arm", name=detector_position_name, WHEN=detector_when,
        #                RELATIVE="PREVIOUS", AT=[0, 0, analyzer_detector_distance.value])
        # det = inst.component("Detector_tubes", name=f"{name}_triplet", RELATIVE="PREVIOUS",
        #                      WHEN=detector_when, EXTEND=detector_extend)
        # det.set_parameters(**self.mcstas_detector_parameters(origin))

        # Move to the center of the analyzer & reorient for monochromator scattering in vertical plane
        inst.component("Arm", name=f"{name}_analyzer_point", RELATIVE=relative, ROTATED_RELATIVE=relative,
                       AT=[0, 0, sample_analyzer_distance.value], ROTATED=[0, 0, 90])
        mono = inst.component("Monochromator_Rowland", name=f"{name}_monochromator",
                              RELATIVE="PREVIOUS", ROTATED=[0, theta, 0], ROTATED_RELATIVE="PREVIOUS",
                              WHEN=analyzer_when, EXTEND=analyzer_extend)
        mono.set_parameters(**self.mcstas_analyzer_parameters(origin, relative.name, f"{name}_triplet"))
        # Add an energy monitor just behind the analyzer?
        # Add the detector tubes
        inst.component("Arm", name=f"{name}_detector_angle", RELATIVE="PREVIOUS",
                       ROTATED=[0, theta, 0], ROTATED_RELATIVE=mono, WHEN=detector_when)
        # inst.component("Arm", name=detector_position_name, WHEN=detector_when,
        #                RELATIVE="PREVIOUS", AT=[0, 0, analyzer_detector_distance.value])
        det = inst.component("Detector_tubes", name=f"{name}_triplet", RELATIVE="PREVIOUS",
                             AT=[0, 0, analyzer_detector_distance.value],
                             WHEN=detector_when, EXTEND=detector_extend)
        det.set_parameters(**self.mcstas_detector_parameters(origin))
