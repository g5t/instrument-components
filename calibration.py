from scipp import scalar, array, vectors, ones, zeros
from copy import deepcopy

dims = ['analyzer']

common_channel_params = dict(
    coverage=scalar(2., unit='degree'),
    d_spacing=scalar(3.355, unit='angstrom'),
    blade_count=array(values=[7, 7, 9, 9, 9], dims=dims),
    energy=array(values=[2.7, 3.2, 3.8, 4.4, 5.], unit='meV', dims=dims),
    detector_orient=vectors(dims=['tube', 'analyzer'], unit='mm',
                            values=[[[0., 0, 0.], [0., 0, 0.], [0., 0, 0.], [0., 0, 0.], [0., 0, 0.]],
                                    [[0., 0, 0.], [0., 0, 0.], [0., 0, 0.], [0., 0, 0.], [0., 0, 0.]],
                                    [[0., 0, 0.], [0., 0, 0.], [0., 0, 0.], [0., 0, 0.], [0., 0, 0.]]]),
    detector_offset=vectors(dims=['tube', 'analyzer'], unit='mm',
                            values=[[[0, -14., 0], [0, -14., 0], [0, -14., 0], [0, -14., 0], [0, -14., 0]],
                                    [[0, 0., 0], [0, 0., 0], [0, 0., 0], [0, 0., 0], [0, 0., 0]],
                                    [[0, 14., 0], [0, 14., 0], [0, 14., 0], [0, 14., 0], [0, 14., 0]]]),
    analyzer_detector_distance=array(values=[1.189, 1.316, 1.420, 1.521, 1.623], unit='m', dims=dims),
    resitivity=200. * ones(dims=['tube', 'analyzer'], shape=[3, 5], unit='Ohm/in').to(unit='Ohm/m'),
    resistance=380. * array(dims=['tube', 'analyzer'], unit='Ohm',
                            values=[[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]]),
)
# deep-copy ensures the changing any per-channel parameters will not affect the other channels
channel_params = {idx: deepcopy(common_channel_params) for idx in range(3)}

variants = {
    2: {'variant': 'l',
        'crystal_shape':
            vectors(unit='mm', dims=dims,
                    values=[[13.5, 150, 1], [15.0, 162, 1], [12.0, 171, 1], [13.0, 180, 1], [14.0, 189, 1]]),
        'detector_length': array(unit='mm', dims=dims, values=[243.9, 265.9, 284.9, 303.4, 321.9]),
        'sample_analyzer_distances': array(values=[1.276, 1.392, 1.497, 1.599, 1.701], unit='m', dims=dims),
        },
    1: {'variant': 'm',
        'crystal_shape':
            vectors(unit='mm', dims=dims,
                    values=[[12.5, 144, 1], [14.5, 156, 1], [11.5, 165, 1], [12.5, 174, 1], [13.5, 183, 1]]),
        'detector_length': array(unit='mm', dims=dims, values=[236.0, 259.0, 277.9, 296.3, 314.8]),
        'sample_analyzer_distance': array(values=[1.189, 1.316, 1.420, 1.521, 1.623], unit='m', dims=dims),
        },
    0: {'variant': 's',
        'crystal_shape':
            vectors(unit='mm', dims=dims,
                    values=[[12.0, 134, 1], [14.0, 147, 1], [11.5, 156, 1], [12.0, 165, 1], [13.5, 177, 1]]),
        'detector_length': array(unit='mm', dims=dims, values=[227.9, 252.0, 270.8, 289.2, 308.8]),
        'sample_analyzer_distance': array(values=[1.100, 1.238, 1.342, 1.433, 1.557], unit='m', dims=dims),
        }
}

for i, d in variants.items():
    channel_params[i].update(d)

for i in range(3, 9):
    channel_params[i] = deepcopy(channel_params[i % 3])

