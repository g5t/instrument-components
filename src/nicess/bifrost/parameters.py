from scipp import Variable

def known_channel_params():
    from scipp import array, scalar, vector, vectors
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
    from ..spatial import vector_to_vector_quaternion
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
