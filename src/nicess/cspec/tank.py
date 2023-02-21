from dataclasses import dataclass


def combined_parameters(user_params: dict, default_params: dict):
    combined = {k: user_params.get(k, default_params[k]) for k in default_params}
    return combined


@dataclass
class Tank:
    from scipp import Variable
    from .pack import Pack
    from ..mcstasscript import ScriptComponent, ScriptInstrument

    packs: tuple[Pack, ...]

    @staticmethod
    def from_calibration(**params):
        from scipp import array, vector, sqrt, tan, atan, asin, min
        from scipp.spatial import rotations_from_rotvecs
        from numpy import arange
        from .pack import Pack
        from .parameters import known_pack_params, tube_xy_displacement_to_quaternion
        combined = combined_parameters(params, known_pack_params())

        sample = combined['sample']  # the origin of the detector positions
        # (,), (tube,), (pack,)  or (pack, tube)
        detector_vector = vector([1, 0, 0]) * combined['sample_detector_distance']

        length = combined['detector_length'].to(unit='m')
        resistance = combined['resistance']
        resistivity = combined['resistivity']

        # (tube,)
        tube_rotations = rotations_from_rotvecs(combined['tube_angles'] * vector([0, 0, 1]))
        # (pack, )
        pack_rotations = rotations_from_rotvecs(combined['pack_angles'] * vector([0, 0, 1]))
        # (pack, tube)
        tube_positions = sample + pack_rotations * tube_rotations * detector_vector
        # (,), or (pack, tube)
        tube_orient = tube_xy_displacement_to_quaternion(length, combined['detector_orient'].to(unit='m'))
        # (pack, tube)
        tube_orient = pack_rotations * tube_rotations * tube_orient

        # we need to define (position: Variable, length: Variable, **params) for each pack
        n_packs = tube_positions.sizes['pack']

        orient_per, resistance_per, resistivity_per, length_per = ['pack' in x.dims for x in (tube_orient, resistance, resistivity, length)]

        pack_list = []
        for n in range(n_packs):
            pp = dict(sample=sample, orient=tube_orient['pack', n] if orient_per else tube_orient,
                      resistivity=resistivity['pack', n] if resistivity_per else resistivity,
                      resistance=resistance['pack', n] if resistance_per else resistance)
            pp['radius'] = combined['detector_radius']
            p = Pack.from_calibration(tube_positions['pack', n], length['pack', n] if length_per else length, **pp)
            pack_list.append(p)

        return Tank(tuple(pack_list))

    def to_cadquery(self, unit=None, add_sphere_at_origin=False):
        from cadquery import Assembly
        if unit is None:
            unit = 'mm'
        assembly = Assembly()
        for index, pack in enumerate(self.packs):
            assembly = assembly.add(pack.to_cadquery(unit=unit), name=f"pack-{index}")

        if add_sphere_at_origin:
            from cadquery import Workplane
            w = Workplane().sphere(radius=10)
            assembly.add(w, name="Origin")

        assembly.name = "CSPEC-secondary"
        return assembly

    def to_mcstasscript(self, inst: ScriptInstrument, relative: ScriptComponent, name: str, parameters: dict):
        from ..mcstasscript import ensure_user_var
        ensure_user_var(inst, 'int', 'flag', 'Flag indicates detection in a monitor')

        for pack_index, pack in enumerate(self.packs):
            pack_name = f"{name}_{1 + pack_index}"
            extend = "flag = (SCATTERED) ? 1 : 0;"
            pack.to_mcstasscript(inst, relative, pack_index * 24, pack_name, extend, parameters)
