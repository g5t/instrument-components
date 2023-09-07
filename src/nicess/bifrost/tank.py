from dataclasses import dataclass

@dataclass
class Tank:
    from scipp import Variable
    from .channel import Channel
    from mcstasscript.interface.instr import McStas_instr as ScriptInstrument
    from mcstasscript.helper.mcstas_objects import Component as ScriptComponent
    from mccode.assembler import Assembler
    from mccode.instr import Instance

    channels: tuple[Channel, Channel, Channel, Channel, Channel, Channel, Channel, Channel, Channel]

    @staticmethod
    def from_calibration(**params):
        from scipp import array
        from .channel import Channel

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
        from scipp import array
        from .channel import Channel
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
        from scipp import vector
        from ..secondary import IndirectSecondary

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
        from ..spatial import combine_triangulations
        vts = [channel.triangulate_detectors(unit=unit) for channel in self.channels]
        return combine_triangulations(vts)

    def triangulate_analyzers(self, unit=None):
        from ..spatial import combine_triangulations
        vts = [channel.triangulate_analyzers(unit=unit) for channel in self.channels]
        return combine_triangulations(vts)

    def triangulate(self, unit=None):
        from ..spatial import combine_triangulations
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
        try:
            from meshplot import plot
        except:
            raise RuntimeError("Install working meshplot to use Tank.plot")
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

    def to_mcstasscript(self, instrument: ScriptInstrument, sample: ScriptComponent, settings: dict = None):
        from scipp import vector, concat, max
        from ..mcstasscript import ensure_user_var, declare_array
        ensure_user_var(instrument, 'int', 'secondary_cassette', 'Secondary spectrometer analyzer cassette index')

        origin = vector([0, 0, 0], unit='m')
        positions = [c.sample_space_angle(origin).to(unit='radian').value for c in self.channels]
        cov_xy = [c.coverage(origin) for c in self.channels]
        cov_x = 2 * max(concat([y for _, y in cov_xy], dim='channel')).value

        slits = instrument.add_component('slits', 'Slit_radial_multi', RELATIVE=sample)
        declared_positions = declare_array(instrument, 'double', f'{slits.name}_positions',
                                           f'{slits.name} position values', positions)
        # slits.set_parameters(slit_width='atan2(slitX, slitDistance)', offset='slitAngle*DEG2RAD',
        #                      number=len(self.channels), radius='slitDistance', height=0.2,
        #                      positions=declared_positions.name
        #                      )
        slits.set_parameters(slit_width=cov_x, offset='slitAngle*DEG2RAD',
                             number=len(self.channels), radius='slitDistance', height=0.2,
                             positions=declared_positions.name)
        slits.append_EXTEND("secondary_cassette = (SCATTERED) ? 1 + slit : -1;")

        for index, channel in enumerate(self.channels):
            name = f"channel_{1 + index}"
            when = f"{1 + index} == secondary_cassette"
            channel.to_mcstasscript(instrument, sample, name=name, when=when, settings=settings)

    def to_mccode(self, assembler: Assembler, sample: Instance, settings: dict = None):
        from scipp import vector, concat, max
        origin = vector([0, 0, 0], unit='m')
        positions = [c.sample_space_angle(origin).to(unit='radian').value for c in self.channels]
        cov_xy = [c.coverage(origin) for c in self.channels]
        cov_x = 2 * max(concat([y for _, y in cov_xy], dim='channel')).value

        slits_name = 'slits'
        declared_positions = f'{slits_name}_positions'
        assembler.declare_array('double', declared_positions, positions, source=__file__, line=173)
        slits = assembler.component(slits_name, 'Slit_radial_multi', at=((0, 0, 0,), sample))
        slits.set_parameters(slit_width=cov_x, offset='slitAngle*DEG2RAD',
                             number=len(self.channels), radius='slitDistance', height=0.2,
                             positions=declared_positions)
        slits.EXTEND("secondary_cassette = (SCATTERED) ? 1 + slit : -1;")

        for index, channel in enumerate(self.channels):
            name = f"channel_{1 + index}"
            when = f"{1 + index} == secondary_cassette"
            channel.to_mccode(assembler, sample, name=name, when=when, settings=settings)

