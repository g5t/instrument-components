from scipp import scalar, array, vectors
from copy import deepcopy

dims = ['analyzer']

common_channel_params = dict(
    coverage=scalar(2., unit='degree'),
    d_spacing=scalar(3.355, unit='angstrom'),
    blade_count=array(values=[9, 9, 9, 7, 7], dims=dims),
    energy=array(values=[2.7, 3.2, 3.7, 4.4, 5.], unit='meV', dims=dims),
    detector_orient=vectors(dims=['tube', 'analyzer'], unit='mm',
                            values=[[[0., 0, 0.], [0., 0, 0.], [0., 0, 0.], [0., 0, 0.], [0., 0, 0.]],
                                    [[0., 0, 0.], [0., 0, 0.], [0., 0, 0.], [0., 0, 0.], [0., 0, 0.]],
                                    [[0., 0, 0.], [0., 0, 0.], [0., 0, 0.], [0., 0, 0.], [0., 0, 0.]]]),
    detector_offset=vectors(dims=['tube', 'analyzer'], unit='mm',
                            values=[[[0, -20., 0], [0, -20., 0], [0, -20., 0], [0, -20., 0], [0, -20., 0]],
                                    [[0, 0., 0], [0, 0., 0], [0, 0., 0], [0, 0., 0], [0, 0., 0]],
                                    [[0, 20., 0], [0, 20., 0], [0, 20., 0], [0, 20., 0], [0, 20., 0]]]),
    analyzer_detector_distance=array(values=[1.189, 1.316, 1.420, 1.521, 1.623], unit='m', dims=dims),
)
# deep-copy ensures the changing any per-channel parameters will not affect the other channels
channel_params = {idx: deepcopy(common_channel_params) for idx in range(3)}

variants = {
    2: {'variant': 'l',
        'crystal_shape':
            vectors(unit='mm', dims=dims,
                    values=[[13.5, 149.9, 2], [15.0, 161.0, 2], [12.0, 170.2, 2], [13.0, 179.3, 2], [14.0, 188.6, 2]]),
        'detector_length': array(unit='mm', dims=dims, values=[233.9, 255.9, 274.9, 293.4, 311.9]),
        'sample_analyzer_distances': array(values=[1.276, 1.388, 1.493, 1.595, 1.697], unit='m', dims=dims),
        },
    1: {'variant': 'm',
        'crystal_shape':
            vectors(unit='mm', dims=dims,
                    values=[[12.5, 142.0, 2], [14.5, 154.1, 2], [11.5, 163.2, 2], [12.5, 172.3, 2], [13.5, 181.6, 2]]),
        'detector_length': array(unit='mm', dims=dims, values=[226.0, 249.0, 267.9, 286.3, 304.8]),
        'sample_analyzer_distance': array(values=[1.189, 1.316, 1.420, 1.521, 1.623], unit='m', dims=dims),
        },
    0: {'variant': 's',
        'crystal_shape':
            vectors(unit='mm', dims=dims,
                    values=[[12.0, 134.0, 2], [14.0, 147.1, 2], [11.5, 156.2, 2], [12.0, 165.2, 2], [13.5, 175.6, 2]]),
        'detector_length': array(unit='mm', dims=dims, values=[217.9, 242.0, 260.8, 279.2, 298.8]),
        'sample_analyzer_distance': array(values=[1.100, 1.238, 1.342, 1.433, 1.544], unit='m', dims=dims),
        }
}

for i, d in variants.items():
    channel_params[i].update(d)

for i in range(3, 9):
    channel_params[i] = deepcopy(channel_params[i % 3])

