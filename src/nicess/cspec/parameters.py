from scipp import Variable


def tube_xy_displacement_to_quaternion(length: Variable, displacement: Variable):
    from scipp import vector, scalar, any, sqrt, allclose
    from ..spatial import vector_to_vector_quaternion
    com_to_end = length * vector([0, 0, 0.5]) + displacement
    l2 = length * length
    x2 = displacement.fields.x * displacement.fields.x
    y2 = displacement.fields.y * displacement.fields.y

    com_to_end.fields.z = sqrt(0.25 * l2 - x2 - y2)

    z2 = displacement.fields.z * displacement.fields.z
    if any(z2 > scalar(0, unit=z2.unit)) and not allclose(com_to_end.fields.z, 0.5 * length - displacement.fields.z):
        raise RuntimeError("Provided tube-end displacement vector(s) contain wrong z-component value(s)")

    # The tube *should* point along z, but we were told it is displaced in x and y;
    # return the orienting Quaternion that takes (001) to the actual orientation
    quaternion = vector_to_vector_quaternion(vector([0, 0, 1]), com_to_end)
    return quaternion


def known_pack_params():
    from scipp import array, scalar, vector, vectors
    from numpy import arange
    known = dict()
    known['sample_detector_distance'] = scalar(3.5, unit='m')
    known['detector_length'] = scalar(3.5, unit='m')
    known['detector_radius'] = scalar(25.4/2, unit='mm')
    known['detector_orient'] = vector([0, 0, 0], unit='mm')
    known['sample'] = vector([0, 0, 0], unit='m')
    known['resistance'] = scalar(380, unit='Ohm')
    known['resistivity'] = scalar(140, unit='Ohm/in').to(unit='Ohm/m')
    # each pack is 32 tubes with 0.43 degrees between tubes
    known['tube_angles'] = array(values=0.43*(arange(32)-16+0.5), unit='degree', dims=['tube'])
    # 5 degree minimum scattering angle, centers 33*0.43 degrees apart such that the highest angle is 180-3
    known['pack_angles'] = array(values=5 + 16*0.43 + (33 * 0.43) * arange(7), unit='degree', dims=['pack'])
    return known
