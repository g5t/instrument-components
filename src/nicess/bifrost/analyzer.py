from dataclasses import dataclass


@dataclass
class Analyzer:
    from mcstasscript.interface.instr import McStas_instr as ScriptInstrument
    from mccode_antlr.assembler import Assembler
    from ..crystals import Crystal
    from scipp import Variable

    blades: tuple[Crystal, ...]  # 7-9 blades

    @property
    def central_blade(self):
        return self.blades[len(self.blades) >> 1]

    @property
    def count(self):
        return len(self.blades)

    @staticmethod
    def from_calibration(position: Variable, focus: Variable, tau: Variable, **params):
        from scipp import scalar, vector
        from scipp.spatial import rotation
        from ..spatial import is_scipp_vector
        from ..rowland import rowland_blades
        from ..crystals import Crystal
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
        from ..spatial import combine_triangulations
        vts = [blade.triangulate(unit=unit) for blade in self.blades]
        return combine_triangulations(vts)

    def extreme_path_corners(self, horizontal, vertical, unit=None):
        from ..spatial import combine_extremes
        vs = [blade.extreme_path_corners(horizontal, vertical, unit=unit) for blade in self.blades]
        return combine_extremes(vs, horizontal, vertical)
    #
    # def mcstas_parameters(self):
    #     from numpy import hstack
    #     return hstack((len(self.blades), self.central_blade.mcstas_parameters))

    def to_cadquery(self, unit=None):
        from ..spatial import combine_assembly
        b = {f'blade-{i}': blade.to_cadquery(unit=unit) for i, blade in enumerate(self.blades)}
        return combine_assembly(**b)

    def coverage(self, sample: Variable):
        from scipp import sqrt, dot, cross, max, min, atan2, vector
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
        from scipp import dot, atan2, vector
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

    def mcstas_parameters(self, sample: Variable, source: str, sink: str) -> dict:
        from mccode_antlr.instr import Instance
        from ..spatial import is_scipp_vector
        is_scipp_vector(sample, 'sample')
        if isinstance(source, Instance):
            source = source.name
        if isinstance(sink, Instance):
            sink = sink.name
        if not isinstance(source, str) or not isinstance(sink, str):
            raise ValueError(f'The source and sink are expected to be str values not {type(source)} and {type(sink)}')

        perp_q, perp_plane, parallel_q = self.central_blade.shape.to(unit='m').value
        hor_cov, ver_cov = self.coverage(sample)
        params = dict(NH=self.count, zwidth=perp_q, yheight=perp_plane, mosaic='mosaic', DM=3.355,
                      gap=0.002, show_construction='showconstruction', angle_h=ver_cov.to(unit='degree').value,
                      source=f'"{source}"', sink=f'"{sink}"')
        return params

    def to_mcstasscript(self, inst: ScriptInstrument, source: str, relative: str, sink: str, theta: float,
                        name: str = None, when: str = None, extend: str = None, origin: Variable = None):
        mono = inst.add_component(name, 'Monochromator_Rowland', RELATIVE=relative,
                                  ROTATED=[0, theta, 0], ROTATED_RELATIVE=relative, WHEN=when, EXTEND=extend)
        mono.set_parameters(**self.mcstas_parameters(origin, source, sink))

    def to_mccode(self, assembler: Assembler, source: str, relative: str, sink: str, theta: float, name: str,
                  when: str = None, extend: str = None, origin: Variable = None):
        mono = assembler.component(name, 'Monochromator_Rowland',
                                   at=((0, 0, 0), relative), rotate=((0, theta, 0), relative))
        mono.set_parameters(**self.mcstas_parameters(origin, source, sink))
        mono.WHEN(when)
        mono.EXTEND(extend)

