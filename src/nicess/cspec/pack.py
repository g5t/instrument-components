from dataclasses import dataclass
from ..decorators import needs

@dataclass
class Pack:
    from scipp import Variable
    from ..detectors import He3Tube

    tubes: tuple[He3Tube, ...]
    resistances: Variable

    @staticmethod
    def from_calibration(position: Variable, length: Variable, **params):
        from numpy import tile
        from scipp import vector, scalar, Variable, concat, ones
        from scipp.spatial import rotations
        from ..spatial import is_scipp_vector
        from ..utilities import is_type, is_scalar
        from ..detectors import He3Tube

        is_scipp_vector(position, 'position')
        if position.ndim != 1 or 'tube' not in position.dims:
            raise RuntimeError("Expected a 1-D list of 'tube' positions")

        ori = params.get('orient', None)
        ori = rotations(values=tile([0, 0, 0, 1], (position.sizes['tube'], 1)), dims=['tube']) if ori is None else ori

        pressure = params.get('pressure', scalar(1., unit='atm'))
        radius = params.get('radius', scalar(25.4/2, unit='mm'))
        elements = params.get('elements', 100)
        resistivity = params.get('resistivity', scalar(140., unit='Ohm/in').to(unit='Ohm/m'))

        map(lambda x: is_type(*x), ((pressure, Variable, 'pressure'), (length, Variable, 'length'),
                                    (radius, Variable, 'radius'), (elements, int, elements),
                                    (resistivity, Variable, 'resistivity')))
        pack = elements, radius, pressure

        if is_scalar(resistivity):
            resistivity = resistivity * ones(shape=(position.sizes['tube'], ), dims=['tube'])

        # Make the oriented tube axis vector(s)
        axis = ori * (length.to(unit=position.unit) * vector([0, 0, 1.]))  # may be a 0-D or 1-D tube vector array
        tube_at = position - 0.5 * axis
        tube_to = position + 0.5 * axis
        tubes = tuple(He3Tube(at, to, rho, *pack) for at, to, rho in zip(tube_at, tube_to, resistivity))

        # Define the contact resistance for the wires
        resistance = params.get('resistance', scalar(2, unit='Ohm'))
        if is_scalar(resistance):
            resistance = resistance * ones(shape=(position.sizes['tube'], 2), dims=['tube', 'end'])
        # allow overriding specific resistances ... somehow
        return Pack(tubes, resistance)

    def to_cadquery(self, unit=None):
        from ..spatial import combine_assembly
        t = {f'tube-{idx:2d}': tube.to_cadquery(unit=unit) for idx, tube in enumerate(self.tubes)}
        return combine_assembly(**t)

    @needs('mcstasscript')
    def to_mcstasscript(self, inst, relative, first_wire_index: int,
                        group_name: str, name: str, extend: str, parameters: dict):
        from numpy import hstack
        from scipp import vector
        from scipp.spatial import rotations_from_rotvecs as rfr
        from ..mcstasscript import declare_array

        lab_to_mcstas = rfr(vector([0, 0, -90], unit='degree')) * rfr(vector([0, -90, 0], unit='degree'))

        # Collect parameter vectors and write into McStas declare section
        declare_array(inst, 'double', f'{name}_positions', 'tube centers of mass',
                      hstack([(lab_to_mcstas * t.center()).to(unit='m').value for t in self.tubes]))
        declare_array(inst, 'double', f'{name}_ends', 'tube center of mass to end vectors',
                      hstack([(lab_to_mcstas * t.end()).to(unit='m').value for t in self.tubes]))
        declare_array(inst, 'double', f'{name}_radii', 'tube radii',
                      hstack([t.radius.to(unit='m').value for t in self.tubes]))
        declare_array(inst, 'double', f'{name}_rhos', 'per-tube wire resistivity',
                      hstack([t.resistivity.to(unit='Ohm/m').value for t in self.tubes]))
        declare_array(inst, 'double', f'{name}_preRs', 'tube pre-wire contact resistances',
                      self.resistances['end', 0].to(unit='Ohm').values)
        declare_array(inst, 'double', f'{name}_postRs', 'tube post-wire contact resistances',
                      self.resistances['end', 1].to(unit='Ohm').values)

        # Add the detector component to the instrument
        pack = inst.component("Detector_tubes", name=name, RELATIVE=relative, EXTEND=extend, GROUP=group_name)
        pack.set_parameters(N=len(self.tubes), first_wire=first_wire_index, pack_filename=f'"{name}_pack.dat"')
        pack.set_parameters({k: f'{name}_{k}' for k in ('positions', 'ends', 'radii', 'rhos', 'preRs', 'postRs')})
        pack.set_parameters(parameters)
