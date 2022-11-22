from scipp import Variable, DataArray
from .bifrost import Tank


def high_low_to_time(hdf_data, freq=None, unit=None):
    from scipp import scalar, array
    if freq is None:
        freq = scalar(88052499, unit='Hz')
    if unit is None:
        unit = 'us'
    times = 'Pulse', 'PrevPulse', 'Event'
    his = {x: array(values=hdf_data[f'{x}TimeHigh'].astype('int32'), dims=['event'], unit='s') for x in times}
    los = {x: array(values=hdf_data[f'{x}TimeLow'].astype('int32'), dims=['event'], unit='1') for x in times}
    clock = {x: his[x].to(unit=unit) + (los[x] / freq).to(unit=unit) for x in times}
    return his, los, clock


def high_low_to_fake_tof(hdf_data, freq=None, unit=None):
    from scipp import logical_and, any, scalar
    if freq is None:
        freq = scalar(88052499, unit='Hz')
    if unit is None:
        unit = 'us'
    his, los, clock = high_low_to_time(hdf_data, freq=freq, unit=unit)

    tof_hi = his['Event'] - his['Pulse']
    tof_lo = los['Event'] - los['Pulse']

    over_second = logical_and(los['Event'] < los['Pulse'], los['Event'] < los['Pulse'])
    if any(over_second):
        os = over_second.values
        tof_hi.values[os] -= 1
        tof_lo.values[os] = (los['Event'].values[os] + freq.to(unit='Hz').value) - los['Pulse'].values[os]

    too_early = logical_and(tof_hi < scalar(0, unit='s'), tof_lo < freq.to(unit='Hz').value)
    if any(too_early):
        print('One or more events are too early')
        tof_hi[too_early] = his['Event'][too_early] - his['PrevPulse'][too_early]
        tof_lo[too_early] = los['Event'][too_early] - los['PrevPulse'][too_early]
    clock['tof'] = tof_hi.to(unit=unit) + (tof_lo / freq).to(unit=unit)

    clock['tof_high'] = tof_hi
    clock['tof_low'] = tof_lo
    clock['event_low'] = los['Event']
    clock['event_high'] = his['Event']
    clock['pulse_low'] = los['Pulse']
    clock['pulse_high'] = his['Pulse']
    return clock


def cassette_pair_from_ring_tube(hdf_data):
    triplet = hdf_data['TubeId']  # *triplet* index: [0, 15]
    ring = hdf_data['RingId']  # *three-cassette group* index: [0,1, 2,3, 4,5]
    # fen = hdf_data['FENId'] # Not used by BIFROST
    pair = (triplet / 3).astype('int')  # in-cassette (analyzer-triplet) index
    cassette = 3 * (ring / 2).astype('int') + triplet % 3
    return cassette.astype('int32'), pair.astype('int32')


def x_from_a_b(hdf_data):
    a, b = hdf_data['AmpA'].astype('int32'), hdf_data['AmpB'].astype('int32')
    return (b-a)/(a+b)  # (a-b)/(a+b)


def continuous_events(hdf_data):
    from scipp import array, ones, DataArray
    from numpy import ones_like

    clocks = high_low_to_fake_tof(hdf_data)
    cassette, pair = cassette_pair_from_ring_tube(hdf_data)
    x = x_from_a_b(hdf_data)

    coords = {'cassette': (cassette, '1'), 'pair': (pair, '1'), 'ratio': (x, '1')}
    coords = {x: array(values=v, dims=['event'], unit=u) for x, (v, u) in coords.items()}
    coords['time_of_flight'] = clocks['tof'].to(unit='ms')

    i = ones(dims=['event'], shape=[len(x)], unit='counts')
    i.variances = ones_like(i.values)

    return DataArray(data=i, coords=coords)


def load_readouts(filename, groupname=None):
    from h5py import File
    with File(filename) as file:
        if groupname is None:
            groups = list(filter(lambda x: '_readouts' in x, list(file)))
            groupname = groups[0]
        data = continuous_events(file[groupname])
    return data


def load_bifrost_readouts(filename):
    from h5py import File
    with File(filename) as file:
        data = continuous_events(file['bifrost_readouts'])

    return data


def load_bifrost_readout_times(filename):
    from h5py import File
    with File(filename) as file:
        clocks = high_low_to_fake_tof(file['bifrost_readouts'])
    return clocks


def sub2x(ratio, edges):
    """
    For ratio = (A-B)/(A+B) values and pairs of ratio-edges defining the limits for multiple tubes,
    extract the ranges and rescale to be in the range [0, 1]
    """
    return (ratio - edges['edges', 0]) / (edges['edges', 1] - edges['edges', 0])


def event_position(x0, x1, x):
    """
    Convert from a unitless proportional value [0, 1] to the linear position between two endpoints
    :param x0: the x=0 position for events
    :param x1: the x=1 position for events
    :param x: one or more relative-position values where events occurred
    :return: the same positions expressed in the coordinate system of the endpoints
    """
    return x * (x1 - x0) + x0


def transformation_graph(secondary):
    from scipp import scalar, sin, cos
    def l1():
        return scalar(160, unit='m')

    def secondary_index(cassette, pair, tube):
        # index = 15 * cassette + 3 * pair + tube
        index = 15 + (2 - tube)
        return index

    def l2(secondary_index):
        return secondary.broadcast_continuous_analyzer_distance(secondary_index)

    def l3(secondary_index):
        return secondary.broadcast_continuous_detector_distance(secondary_index)

    def delta_a4(secondary_index, x):
        return secondary.broadcast_continuous_delta_a4(secondary_index, x).to(unit='degree')

    def a6(secondary_index):
        return secondary.broadcast_continuous_a6(secondary_index).to(unit='degree')

    def d(secondary_index):
        return secondary.broadcast_continuous_plane_spacing(secondary_index)

    def criteria(d_spacing, a6, delta_a4, l1, l2, l3):
        from scipp.constants import Planck as h, neutron_mass as m
        return ((2 * d_spacing * m / h) * sin(a6/2) * (l1 * cos(delta_a4) + l2 + l3)).to(unit='ms')

    def one(time_of_flight, criteria):
        return time_of_flight / criteria

    graph = dict(secondary_index=secondary_index, l1=l1, l2=l2, l3=l3, criteria=criteria, a6=a6, delta_a4=delta_a4,
                 d_spacing=d, one=one)

    return graph


def transform_to_criteria(data: DataArray, tank: Tank, shortcut=False):
    from scipp import array, group, collapse, concat
    from numpy import vstack
    # can this be done for multiple triplets at once?
    triplet = tank.channels[1].pairs[0].detector
    boundaries = triplet.a_minus_b_over_a_plus_b_edges().rename_dims({'tube': 'ratio'})
    tube_index = array(values=[-1, 0, -2, 1, -3, 2, -4], dims=['ratio'])

    binned = data.bin(ratio=boundaries)
    binned.coords['tube'] = tube_index

    edges = vstack((boundaries.values[:-1], boundaries.values[1:]))
    edges[:, 3] = edges[1, 3], edges[0, 3]
    binned.coords['edges'] = array(values=edges, dims=['edges', 'ratio'])

    # tubes = array(values=[-2, -3, 0, 1, 2], dims=['tube'])
    # tube_ratios = {f"tube {x:c}": group(binned, 'tube')['tube', x].copy() for x in tubes}
    # tube_plot = {x: t.bin(ratio=1000) for x, t in tube_ratios.items() if t.sum().value}
    # out = plot(tube_plot)
    # for (l, h), t in zip(binned.coords['edges'].values.T, tube_plot.values()):
    #     l, h = min([l, h]), max([l, h])
    #     out.ax.plot([l, l, h, h], t.max().value * [1, 0, 0, 1], '--k')
    #

    # # Remove any 'tube' bins which have no events:
    # pruned = concat([v for v in collapse(binned, 'tube').values() if v.values.sizes['event']], dim='tube')

    # convert the (A-B)/(A+B) subranges to [0, 1] unitless values along each tube:
    # this requires removing the coordinate named 'ratio' from the binned data
    del binned.coords['ratio']
    # bin in binned are [not tube, tube-0, not tube, tube-1, not tube, tube-2, not tube]
    # from which we only want to keep the tubes, hence the ['ratio', 1::2]
    ofx = binned['ratio', 1::2].transform_coords(['x'], graph={'x': sub2x})

    if shortcut:
        return ofx

    # # Re-arrange the data into tube groups, and append the tube endpoints
    # oft = ofx.group('tube')
    # oft.coords['x0'] = concat([t.at for t in triplet.tubes], dim='tube')

    # ofp = oft.transform_coords(['position'], graph={'position': event_position})
    # ofp.bin(tube=1).data.values[0].plot(projection='3d', positions='position')

    ofc = ofx.transform_coords(['time_of_flight', 'criteria', 'one', 'l1', 'l2', 'l3'], graph=transformation_graph(tank.to_secondary()))

    return ofc
