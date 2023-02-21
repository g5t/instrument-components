from dataclasses import dataclass


def __is_type__(x, t, name):
    if not isinstance(x, t):
        raise RuntimeError(f"{name} must be a {t}")


@dataclass
class Triplet:
    from mcstasscript.interface.instr import McStas_instr as ScriptInstrument
    from mcstasscript.helper.mcstas_objects import Component as ScriptComponent
    from ..detectors import He3Tube
    from scipp import Variable

    tubes: tuple[He3Tube, He3Tube, He3Tube]
    resistances: Variable

    @staticmethod
    def from_calibration(position: Variable, length: Variable, **params):
        """Take (fitting) calibration data and construct the object used to convert events to (Q,E)"""
        # The current crop of inputs is not sufficient to capture all degrees of freedom, but is a start.
        from scipp import sqrt, dot, vector, scalar, Variable
        from scipp.spatial import rotations
        from ..spatial import is_scipp_vector
        from ..detectors import He3Tube

        map(lambda x: is_scipp_vector(*x), ((position, 'position'),))
        if position.ndim != 1 or 'tube' not in position.dims or position.sizes['tube'] != 3:
            raise RuntimeError("Expected positions for 3 'tube'")

        ori = params.get('orient', None)
        ori = rotations(values=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dims=['tube']) if ori is None else ori

        pressure = params.get('pressure', scalar(1., unit='atm'))
        radius = params.get('radius', scalar(5., unit='mm'))
        elements = params.get('elements', 10)
        resistivity = params.get('resistivity', scalar(140., unit='Ohm/in').to(unit='Ohm/m'))
        map(lambda x: __is_type__(*x), ((pressure, Variable, 'pressure'), (length, Variable, 'length'),
                                        (radius, Variable, 'radius'), (elements, int, 'elements'),
                                        (resistivity, Variable, 'resistivity')))
        # pack the tube parameters
        pack = elements, radius, pressure

        # ensure that there is one resistivity per tube
        from ..utilities import is_scalar
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
        from ..spatial import combine_triangulations
        vts = [tube.triangulate(unit=unit) for tube in self.tubes]
        return combine_triangulations(vts)

    def extreme_path_corners(self, horizontal, vertical, unit=None):
        from ..spatial import combine_extremes
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
        from ..spatial import combine_assembly
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
