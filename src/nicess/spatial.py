from scipp import Variable


def __is_vector__(x: Variable):
    from scipp import DType
    return x.dtype == DType.vector3


def is_scipp_vector(v: Variable, name: str):
    from scipp import DType
    if v.dtype != DType.vector3:
        raise RuntimeError(f"The {name} must be a scipp.DType('vector3')")


def __is_quaternion__(x: Variable):
    from scipp import DType
    return x.dtype == DType.rotation3


def vector_to_vector_quaternion(fr: Variable, to: Variable):
    if not __is_vector__(fr) or not __is_vector__(to):
        raise RuntimeError("Two vectors required!")
    from scipp import sqrt, dot, cross
    from numpy import concatenate, expand_dims
    # following http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
    u = fr / sqrt(dot(fr, fr))
    v = to / sqrt(dot(to, to))
    scalar_part = 0.5 * sqrt(2 + 2 * dot(u, v))
    vector_part = 0.5 * cross(u, v) / scalar_part
    values = concatenate((vector_part.values, expand_dims(scalar_part.values, axis=-1)), axis=-1)
    dims = vector_part.dims

    # we need to bypass the standard error checking in the scipp python module due to a bug:
    # from scipp.spatial import rotations
    from scipp._scipp import core as scipp_core
    q = scipp_core.rotations(values=values, dims=dims)
    return q


def combine_triangulations(vts: list[tuple[Variable, list[list[int]]]]):
    from scipp import concat
    from numpy import cumsum, hstack
    if any((v.ndim != 1 for v, t in vts)):
        raise RuntimeError("All vertices expected to be 1-D lists of vectors")
    vdims = [v.dims[0] for v, t in vts]
    if any((d != vdims[0] for d in vdims)):
        raise RuntimeError("All vertex arrays expected to have the same dimension name")
    vdim = vdims[0]

    lens = [len(v) for v, t in vts]
    offset = hstack((0, cumsum(lens)))[:-1]
    faces = [[off + i for i in t] for off, (v, ts) in zip(offset, vts) for t in ts]

    vertices = concat([v for v, t in vts], dim=vdim)

    return vertices, faces


def write_off_file(vertices, faces, filename):
    stream = f"OFF\n{len(vertices)} {len(faces)} 0\n"
    for v in vertices.values:
        s = " ".join([f"{x:3.9f}" for x in v])
        stream += s + "\n"
    for v in faces:
        s = " ".join([f"{x}" for x in v])
        stream += f"{len(v)} {s}\n"
    with open(filename, 'w') as f:
        f.write(stream)


def combine_extremes(vs: list[Variable], horizontal: Variable, vertical: Variable):
    from scipp import concat, dot, sqrt, scalar, isclose, cross
    from numpy import argmax, argmin, hstack, unique
    is_scipp_vector(horizontal, 'horizontal')
    is_scipp_vector(vertical, 'vertical')
    map(lambda p: is_scipp_vector(p, 'x'), vs)
    if any((v.ndim != 1 for v in vs)):
        raise RuntimeError("All vertices expected to be 1-D lists of vectors")
    dim = vs[0].dims[0]
    if any((v.dims[0] != dim for v in vs)):
        raise RuntimeError("All vertex arrays expected to have the same dimension name")
    vs = concat(vs, dim)
    y = horizontal / sqrt(dot(horizontal, horizontal))
    z = vertical / sqrt(dot(vertical, vertical))
    if not isclose(dot(z, y), scalar(0.)):
        z = z - dot(z, y) * y
        z = z / sqrt(dot(z, z))
    x = cross(y, z)
    v_yz = vs - dot(x, vs) * x
    v_yz_pp = dot(v_yz, y + z).values
    v_yz_pm = dot(v_yz, y - z).values
    v_yz_p0 = dot(v_yz, y).values
    v_yz_0p = dot(v_yz, z).values
    max_pp, max_pm, max_p0, max_0p = [argmax(x) for x in (v_yz_pp, v_yz_pm, v_yz_p0, v_yz_0p)]
    min_pp, min_pm, min_p0, min_0p = [argmin(x) for x in (v_yz_pp, v_yz_pm, v_yz_p0, v_yz_0p)]

    idxs = hstack([max_p0, max_pp, max_0p, min_pm, min_p0, min_pp, min_0p, max_pm])  # order is important
    _, unique_index = unique(idxs, return_index=True)
    idxs = idxs[sorted(unique_index)]  # only indexes which are unique, in the same order as provided in idxs

    return vs[idxs]


def perpendicular_directions(direction: Variable):
    from scipp import sqrt, dot, cross, scalar, isclose, abs, vector
    is_scipp_vector(direction, 'direction')

    direction /= sqrt(dot(direction, direction))

    horizontal = vector([1., 0, 0]) if isclose(abs(direction.fields.y), scalar(1.)) else vector([0, -1., 0])
    horizontal -= dot(horizontal, direction) * direction
    horizontal /= sqrt(dot(horizontal, horizontal))

    vertical = cross(horizontal, direction)

    return horizontal, vertical