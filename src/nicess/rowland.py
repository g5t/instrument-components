from scipp import Variable

def three_point_circle(p0: Variable, p1: Variable, p2: Variable):
    from scipp import sqrt, dot, cross, scalar
    va = p1 - p0
    vb = p2 - p0
    a = sqrt(dot(va, va))
    b = sqrt(dot(vb, vb))
    n = cross(vb, va) / a / b
    if sqrt(dot(n, n)) == scalar(0., unit=n.unit):
        raise RuntimeError("Points are colinear")
    y = cross(n, vb / b)
    ax = dot(va, vb) / b  # |a along b|
    ay = dot(va, y)       # |a perpendicular to b|
    c = 0.5 * vb + ((ax / ay) * 0.5 * (ax - b) + 0.5 * ay) * y
    r = sqrt(dot(c, c))
    return p0 + c, r, n


def angle_between(a: Variable, b: Variable):
    from scipp import acos, dot, sqrt
    return acos(dot(a, b) / sqrt(dot(a, a)) / sqrt(dot(b, b)))


def transfer_angular_range(origin: Variable, target: Variable, center: Variable, alpha: Variable):
    from scipp import sqrt, dot, cross, isclose, sin, cos, scalar
    # direction along s
    s = origin - target
    v = s / sqrt(dot(s, s))
    c = center - target
    n = cross(c, v) / sqrt(dot(c, c))
    if sqrt(dot(n, n)) == scalar(0., unit=n.unit):
        raise RuntimeError("vectors are colinear")
    xi = center - origin
    if not isclose(dot(xi, xi), dot(c, c)):
        raise RuntimeError("s is not on the circle centered at c and passing through the origin")
    # directions perpendicular to s in the plane with c
    hats = [sin(alpha) * x - cos(alpha) * v for x in (cross(n, v), cross(v, n))]
    # points relative to target where acos(dot(p - origin, target - origin)) == alpha
    ps = [s * h * 2 * dot(h, xi) for h in hats]
    if not all(isclose(angle_between(p-origin, target-origin), alpha) for p in ps)
        raise RuntimeError("Problem finding points at +- alpha")
    # Find the angular range relative to (target - center):
    beta = [angle_between(p - center, target-center) for p in ps]
    return tuple(sorted(beta, key=lambda x: x.value))


def rowland_blade_angles(coverage: tuple[Variable,Variable], radius: Variable, count: int, width: Variable):
    # the crystals cover from (min(beta), max(beta)) *around the Rowland circle center point*, rho
    #  min(beta)                                     max(beta)
    #  ----v------.------.------.------.------.------.---v----> rho
    #      |xxx|  |xxx|  |xxx|  |xxx|  |xxx|  |xxx|  |xxx|
    # So the angular range is broken up into N blade-width and (N-1) gap-width segments
    # The blade width is rho_blade ~= width / rowland_radius, so the gap width is given by
    #       rho_gap = (diff(beta) - N * rho_blade) / (N - 1)
    from scipp import atan2, isclose, scalar, concat
    r_width = 2 * atan2(0.5 * width, radius.to(unit=width.unit))
    r_gap = (coverage[1] - coverage[0] - count * r_width) / (count - 1)
    angles = [coverage[0] + 0.5 * r_width + index * (r_width + r_gap) for index in range(count)]
    if not isclose(angles[count >> 1], scalar(0., unit='radian')):
        raise RuntimeError("Central angle should be zero but is not")
    if not isclose(angles[-1], coverage[1] - 0.5 * r_width):
        raise RuntimeError("Last angle should be half a radial-width from the maximum coverage angle!")
    return concat(angles, 'blade')


def rowland_blades(source: Variable, position: Variable, focus: Variable, alpha: Variable, width: Variable, count: int):
    center, radius, normal = three_point_circle(source, position, focus)
    betas = transfer_angular_range(source, position, center, alpha)
    angles = rowland_blade_angles(betas, radius, count, width)

    # for each angle, create the rotation matrix needed to rotate (position - center)
    from scipp.spatial import rotations_from_rotvecs
    rotations = rotations_from_rotvecs(rotation_vectors=angles * normal)

    # *hopefully broadcast* the rotations to the center-to-analyzer-position vector
    blade_positions = rotations * (position - center) + center

    # find the crystal-normal directions; the central one bisects the scattering vector, while the remaining ones
    # are rotated by half their Rowland angles
    from scipp import sqrt, dot
    rotations = rotations_from_rotvecs(rotation_vectors=0.5 * angles * normal)
    tau = 0.5 * (source + focus) - position
    tau /= sqrt(dot(tau, tau))
    blade_taus = rotations * tau

    return blade_positions, blade_taus
