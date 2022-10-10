from numpy import ndarray
from scipp import Variable
from typing import Tuple
from h5py import Group


def _default_dtype(dts: str):
    if dts == 'float64':
        return 'f8'
    if dts == 'float32':
        return 'f4'
    raise RuntimeError(f'No default dtype for {dts}')


def scalar_serialize_type(x: Variable, name=None, dtype=None):
    from .utilities import is_scalar
    if not is_scalar(x):
        raise ValueError("Input variable is not a scalar variable")
    if name is None:
        name = ''
    sname = f"{name}:{x.unit.name}:{x.values.dtype}"
    dtype = _default_dtype(str(x.values.dtype)) if dtype is None else dtype
    return sname, dtype


def vector_serialize_types(v: Variable, name=None, dtype=None):
    from scipp import DType
    if v.dtype != DType.vector3:
        raise RuntimeError("Only intended for scipp vector3 data")
    if name is None:
        name = ''
    names = [f'{name}{index}:{v.unit.name}:{v.values.dtype}' for index in range(3)]
    dtype = _default_dtype(str(v.values.dtype)) if dtype is None else dtype
    return ((n, dtype) for n in names)


def _deserialize_unit_dtype(matches: list):
    units = [m.group('unit') for m in matches]
    dtypes = [m.group('dtype') for m in matches]

    for i in range(1, len(matches)):
        if units[i] != units[0]:
            raise ValueError("All components must have the same units")
        if dtypes[i] != dtypes[0]:
            raise ValueError("All components must have the same dtype")

    return units[0], dtypes[0]


def scipp_scalar_deserialize(combination: ndarray, name: str, dim=None):
    import re
    from numpy import hstack
    from scipp import array, scalar
    if dim is None:
        dim = 'stack'
    rc = re.compile(rf'^{name}:(?P<unit>.*):(?P<dtype>.*)$)')
    sname = list(filter(lambda n: rc.match(n), combination.dtype.names))
    if len(sname) != 1:
        raise RuntimeError(f"{len(sname)} of 1 expected name found!")
    matches = [rc.match(x) for x in sname]
    unit, dtype = _deserialize_unit_dtype(sname, rc)
    values = combination[matches[0]]

    return array(values=values, dims=[dim], unit=unit) if values.size > 1 else scalar(value=values.item(), unit=unit)


def vector_deserialize(combination: ndarray, name: str, dim=None):
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
    unit, dtype = _deserialize_unit_dtype(matches)
    # the elements are likely continguous, but we should not assume so:
    index = [int(x.group('index')) for x in matches]
    names = [m.string for _, m in sorted(zip(index, matches), key=lambda x: x[0])]
    values = hstack([combination[n] for n in names]).astype(dtype)
    if values.ndim > 1:
        return vectors(values=values, dims=[dim], unit=unit)
    return vector(value=values, unit=unit)


def scipp_scalar_serialize(x: Variable, name: str, dtype=None):
    from numpy.lib.recfunctions import unstructured_to_structured as u2s
    from numpy import dtype as numpy_dtype
    from .utilities import is_scalar
    if not is_scalar(x):
        raise RuntimeError("Provided value is not a scalar")
    return u2s(x.values, numpy_dtype(scalar_serialize_type(x, name=name, dtype=dtype)))


def vector_serialize(vec: Variable, name: str, dtype=None):
    from numpy.lib.recfunctions import unstructured_to_structured as u2s
    from numpy import dtype as numpy_dtype
    types = list(vector_serialize_types(vec, name=name, dtype=dtype))
    return u2s(vec.values, numpy_dtype(types))


def scalar_deserialize(comb: ndarray, names: tuple[tuple[str, bool], ...]):
    scalars = [scipp_scalar_deserialize(comb, x) if isss else comb[x].squeeze() for x, isss in names]
    scalars = [x.item() if isinstance(x, ndarray) and x.size == 1 else x for x in scalars]
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
