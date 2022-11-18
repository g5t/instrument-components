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
    tube = hdf_data['TubeId']
    ring = hdf_data['RingId']
    # fen = hdf_data['FENId'] # Not used by BIFROST
    pair = (tube / 3).astype('int')
    cassette = 3 * (ring / 2).astype('int') + tube % 3
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
