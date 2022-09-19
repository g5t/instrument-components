from numpy import ndarray
from scipp import Variable
from typing import Tuple
from h5py import Group

def vector_serialize_types(v: Variable, name=None, dtype=None):
    from scipp import DType
    if (v.dtype != DType.vector3):
        raise RuntimeError("Only intended for scipp vector3 data")
    if name is None:
        name = ''
    names = [f'{name}{index}:{v.unit.name}:{v.values.dtype}' for index in range(3)]
    if dtype is None:
        if str(v.values.dtype) == 'float64':
            dtype = 'f8'
        elif str(v.values.dtype) == 'float32':
            dtype = 'f4'
        else:
            raise RuntimeError(f"No default dtype for {v.values.dtype}!")
    return ((n, dtype) for n in names)


def vector_deserialize(combination: ndarray, name: str, dim = None):
    import re
    from numpy import hstack
    from scipp import vector, vectors
    if dim is None:
        dim = 'stack'
    rc = re.compile(rf'^{name}(?P<index>[0-2]):(?P<unit>.*):(?P<dtype>.*)$')
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
    values = hstack([combination[n] for n in names]).astype(dtypes[0])
    if values.ndim > 1:
        return vectors(values=values, dims=[dim], unit=units[0])
    return vector(value=values, unit=units[0])


def vector_serialize(vec: Variable, name: str, dtype=None):
    from numpy.lib.recfunctions import unstructured_to_structured as u2s
    from numpy import dtype as numpy_dtype
    types = list(vector_serialize_types(vec, name=name, dtype=dtype))
    return u2s(vec.values, numpy_dtype(types))


def scalar_deserialize(combinatiton: ndarray, names: Tuple[str, ...]):
    scalars = [combinatiton[x].squeeze() for x in names]
    scalars = [x.item() if x.size == 1 else x for x in scalars]
    return scalars


def deserialize_valid_class(group: Group, cls):
    import sys
    if 'py_class' not in group.attrs:
        raise RuntimeError('The provided HDF5 group is not a serialized class object')
    g_cls = group.attrs['py_class']
    # get a handle to the `nicess` module
    module = sys.modules[globals()['__package__']]
    # and pull out the matching group name -- this will
    return issubclass(getattr(module, g_cls), cls) if hasattr(module, g_cls) else False
