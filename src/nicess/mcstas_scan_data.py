def simple(x):
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        pass
    return x


def read_mcc_header_file(filename):
    from numpy import array
    with open(filename, 'r') as file:
        lines = file.readlines()
    # The first line *must* be '# Format: McCode list with text headers'
    
    header = filter(lambda x: x[0] == '#', lines)
    data = filter(lambda x: x[0] != '#', lines)
    
    headers = dict(Param={})
    for line in [x[2:].strip() for x in header]:
        kv = line.split()
        key, value = kv[0], ' '.join(kv[1:])
        if 'Param' in key:
            key, value = value.split('=')
            headers['Param'][key] = simple(value)
        else:
            headers[key.strip(':')] = simple(value)
            
    data = [[float(x) for x in line.split()] for line in data]
    
    return headers, array(data)
    
    
def mcc_units():
    known_units = dict(p='count/s', t='sec', E='meV', k='1/angstrom', v='m/s', radius='m', wavelength='angstrom', angle='degree')
    for n in ('k', 'v'):
        for ex in ('x', 'y', 'z', 'xy', 'yz', 'xz'):
            known_units[n+ex] = known_units[n]
    for ex in ('x', 'y', 'z', 'xy', 'yz', 'xz'):
        known_units[ex] = known_units['radius']
    for ex in ('vdiv', 'ydiv', 'dy', 'hdiv', 'divergence', 'xdiv', 'theta', 'longitude', 'phi', 'lattitude'):
        known_units[ex] = known_units['angle']
    for ex in ('energy', 'omega'):
        known_units[ex] = known_units['E']
    known_units['wavevector'] = known_units['k']
    known_units['time'] = known_units['t']
    
    return known_units 


def mcc_to_scipp(filename, user_variables=None):
    from scipp import array, scalar, DataArray
    meta, data = read_mcc_header_data(filename)
    units = mcc_units()
    translate = {}
    if user_variables is not None:
        for i, nu in enumerate(user_variables):
            translate[f'U{1+i}'] = nu[0]
            units[nu[0]] = nu[1]
    s = {}
    for i, n in enumerate(meta['ylabel'].split()):
        n = translate.get(n, n)
        s[n] = array(values=data[:,i], unit=units.get(n, '1'), dims=['event'])
        
    d = s['p']
    del s['p']
    
    da = DataArray(data=d, coords=s)
    for k, v in meta.items():
        da.attrs[k] = scalar(v)
        
    return da
    

def mcc_scan_to_pd(filename):
    from pandas import read_csv
    with open(filename, 'r') as file:
        lines = file.readlines()
    # The first line *must* be '# Format: McCode list with text headers'
    
    header = filter(lambda x: x[0] == '#', lines)
    data = filter(lambda x: x[0] != '#', lines)
    
    headers = dict(Param={})
    for line in [x[2:].strip() for x in header]:
        kv = line.split()
        key, value = kv[0], ' '.join(kv[1:])
        if 'Param' in key:
            key, value = value.split('=')
            headers['Param'][key] = simple(value)
        else:
            headers[key.strip(':')] = simple(value)

    data = read_csv(filename, comment='#', delimiter=' ', names=headers['variables'].split())
    
    return headers, data


def mcc_scan_to_scipp(scan_root, hdf_filename, units=None):
    from nicess.readouts import load_readouts
    from scipp.compat import from_pandas
    from scipp import ones, concat
    from pathlib import Path
    if units is None:
        units = {}
    if not isinstance(scan_root, Path):
        scan_root = Path(scan_root)
    if not scan_root.is_dir():
        raise RuntimeError(f"The provided scan root {scan_root} is not a directory!")

    dat_file = scan_root.joinpath('mccode.dat')
    if not dat_file.exists():
        raise RuntimeError(f"The scan information file {dat_file} does not exist")

    header, scan_data = mcc_scan_to_pd(dat_file)

    # A McStas scan is always 1-dimensional

    dims = [d.strip() for d in header['xvars'].split(',')]
    coords = {d: from_pandas(scan_data[d]) for d in dims}

    n_pts = scan_data.shape[0]
    data = [load_readouts(scan_root.joinpath(f"{i}", hdf_filename)) for i in range(n_pts)]

    coords = {d: k.data.rename_dims(**{k.dims[0]: d}) for d, k in coords.items()}

    for i in range(n_pts):
        for n, v in coords.items():
            data[i].coords[n] = v[i] * ones(sizes=data[i].sizes, unit=units.get(n, '1'))

    data = concat(data, 'event')

    return coords, data
