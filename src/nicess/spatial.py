from scipp import Variable


def __is_vector__(x: Variable):
    from scipp import DType
    return x.dtype == DType.vector3


def __is_quaternion__(x: Variable):
    from scipp import DType
    return x.dtype == DType.rotation3


def vector_to_vector_quaternion(fr: Variable, to: Variable):
    if not __is_vector__(fr) or not __is_vector__(to):
        raise RuntimeError("Two vectors required!")
    from scipp import sqrt, dot, cross, scalar
    from scipp.spatial import rotation
    # following http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
    u = fr / sqrt(dot(fr, fr))
    v = to / sqrt(dot(to, to))
    m = sqrt(scalar(2) + 2 * dot(u, v))
    w = (1 / m) * cross(u, v)
    # rotation value [x, y, z, w] -> w + x*i + y*j + z*k
    return rotation(value=[*w.value, m.value / 2])

