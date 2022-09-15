from numpy import ndarray
from scipp import Variable

def vector_serialize_types(v: Variable, name=None, dtype=None):
    from scipp.DType import vector3
    if (v.dtype != vector3):
        raise RuntimeError("Only intended for scipp vector3 data")
    if name is None:
        name = ''
    names = [f'{name}{index}/{v.unit}:{v.values.dtype}' for index in range(3)]
    if dtype is None:
        if str(v.values.dtype) == 'float64':
            dtype = 'f8'
        elif str(v.values.dtype) == 'float32':
            dtype = 'f4'
        else
            raise RuntimeError(f"No default dtype for {v.values.dtype}!")
    return ((dtype, n) for n in names)

def vector_deserialize(combination: ndarray, name: str, dim = None):
    import re
    from numpy import hstack
    from scipp import vectors
    if dim is None:
        dim = 'stack'
    rc = re.compile(rf'^{name}(?P<index>[0-2]/(?P<unit>.*):(?P<dtype>.*)$')
    vnames = list(filter(lambda n: rc.match(n), combination.dtype.names))
    if len(vnames) != 3:
        raise RuntimeError(f"{len(vnames)} of 3 expected names found!")
    matches = [rc.match(x) for x in vnames]
    units = [m.group('unit') for m in matches]
    dtypes = [m.group('dtype') for m in matches]
    if units[0] != units[1] or units[1] != units[2]:
        raise RuntimeError(f"All vector elements must have the same units")
    if dtypes[0] != dtypes[1] or dtypes[1] != dtypes[2]:
        raise RuntimeError(f"All vector elements must have the same dtype")
    # the elements are likely continguous, but we should not assume so:
    index = [int(x.group('index')) for x in matches]
    names = [m.string for _, m in sorted(zip(index, matches), key=lambda x: x[0])]
    values = hstack((combination[n] for n in names)).astype(dtypes[0])
    return vectors(values=values, dims=[dim], unit=units[0])
    
