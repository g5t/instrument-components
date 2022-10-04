from scipp import Variable

def three_point_circle(p0: Variable, p1: Variable, p2: Variable):
    from scipp import sqrt, dot, cross, scalar, isclose
    # vectors along each side of the triangle
    va, vb, vc = p1 - p0, p2 - p0, p2 - p1
    # and their lengths
    a, b, c = [sqrt(dot(x, x)) for x in (va, vb, vc)]
    # the semi-perimeter, s, defines the area sqrt(s(s-a)...)
    s = (a+b+c)/2
    # and the circumscribing circle has radius:
    r = a*b*c / (4*sqrt(s*(s-a)*(s-b)*(s-c)))
    # to find the center we must define the plane normal
    n = cross(vb, va) / a / b
    if sqrt(dot(n, n)) == scalar(0., unit=n.unit):
        raise RuntimeError("Points are colinear")
    n /= sqrt(dot(n, n))
    # the bisector of any one of the sides points to the center
    perp = [p / sqrt(dot(p,p)) for v in (-va, vb, -vc) for p in (cross(n, v),)]
    # the length of the bisector is given by the pythagorean theorem
    h = [sqrt(r*r - i*i/4) * p for p, i in zip(perp, (a, b, c))]
    # but whether each h should be positive or negative is determined by the distance
    # to the other two points
    z = []
    for x0, y0, z0, v, bi in zip((p0, p0, p1), (p1, p1, p2), (p2, p2, p0), (va, vb, vc), h):
      t = x0 + 0.5*v + bi
      if isclose(dot(t-y0, t-y0), r*r) and isclose(dot(t-z0, t-z0), r*r):
        z.append(t)
      else:
        z.append(x0 + 0.5*v - bi)

    # all three z points should be the same:
    if not isclose(z[0], z[1]) or not isclose(z[1], z[2]):
      raise RuntimeError(f"Mismatched central points,\n{p0=}\n{p1=}\n{p2=}\n{z = }")
    if z[0] != z[1] or z[1] != z[2]:
        # Only find the average of the three points if they are not identical to avoid division-by-three weirdness
        return (z[0]+z[1]+z[2])/3, r, n
    return z[0], r, n


def angle_between(a: Variable, b: Variable):
    from scipp import acos, dot, sqrt
    return acos(dot(a, b) / sqrt(dot(a, a)) / sqrt(dot(b, b)))


def transfer_angular_range(origin: Variable, target: Variable, center: Variable, alpha: Variable):
    from scipp import sqrt, dot, cross, isclose, sin, cos, scalar
    # direction along s
    s = origin - target
    v = s / sqrt(dot(s, s))
    c = center - target
    n = cross(c, v)
    if sqrt(dot(n, n)) == scalar(0., unit=n.unit):
        raise RuntimeError("vectors are colinear")
    n = n / sqrt(dot(n, n))
    xi = center - origin
    if not isclose(dot(xi, xi), dot(c, c)):
        print(f"{origin = }\n{target = }\n{center = }\n{alpha = }\n{xi = }\n{c = }\n{dot(xi, xi) = }\n{dot(c, c) = }")
        raise RuntimeError("s is not on the circle centered at c and passing through the origin")
    # directions along the ray from origin to either side of target
    hats = [sin(alpha) * x/sqrt(dot(x, x)) - cos(alpha) * v for x in (cross(n, v), cross(v, n))]
    # points relative to target where acos(dot(p - origin, target - origin)) == alpha
    ps = [origin + h * 2 * dot(h, xi) for h in hats]
    if not all(isclose(angle_between(p-origin, target-origin), alpha) for p in ps):
        raise RuntimeError(f"Problem finding points at +- alpha\n{ps = }\n{origin = }\n{target = }\n{center = }\n{[angle_between(p-origin, target-origin) for p in ps] = }\n{alpha = }")
    if not all(isclose(dot(p-center, p-center), dot(c, c)) for p in ps):
        raise RuntimeError(f"Problem finding points at +- alpha on the Rowland circle")
    # Find the angular range relative to (target - center):
    beta = [angle_between(p - center, target-center) for p in ps]
    return (beta[0] + beta[-1])/2


def rowland_blade_angles(beta: Variable, radius: Variable, count: int, width: Variable, gap=None):
    # the crystals cover from (min(beta), max(beta)) *around the Rowland circle center point*, rho
    #    -beta                                          beta
    #  ----v------.------.------.------.------.------.---v----> rho
    #      |xxx|  |xxx|  |xxx|  |xxx|  |xxx|  |xxx|  |xxx|
    # So the angular range is broken up into N blade-width and (N-1) gap-width segments
    # The blade width is rho_blade ~= width / rowland_radius, so the gap width is given by
    #       rho_gap = (diff(beta) - N * rho_blade) / (N - 1)
    from scipp import atan2, isclose, scalar, concat, sin, arange
    if gap is None:
        r_width = 2 * atan2(y=0.5 * width, x=radius.to(unit=width.unit))
        # r_width = 2 * asin(width / (2 * radius.to(unit=width.unit)))
        r_gap = (2 * beta - count * r_width) / (count - 1)
    else:
        # Follow the RTP method to calculate radial 'width' of each blade -- which might be greater than actual width
        from numpy import pi
        r_gap = gap / (2 * pi * radius.to(unit=gap.unit)) * scalar(2 * pi, unit='radian')
        # r_gap = 2 * atan2(y=0.5 * gap, x=radius.to(unit=gap.unit))
        r_width = (2 * beta - (count - 1) * r_gap) / count

    real_gap = sin((r_width - 2*atan2(y=0.5 * width, x=radius.to(unit=width.unit))) + r_gap) * radius.to(unit=width.unit)
    print(f"Expected gap width {sin(r_gap) * radius.to(unit=width.unit): c}, Real {real_gap:c}")
    # print(f"arc length {2 * beta / scalar(1, unit='radian') * radius.to(unit=width.unit):c} blade width {width:c}")
    # angles = [-beta + 0.5 * r_width + index * (r_width + r_gap) for index in range(count)]
    half_count = count >> 1
    angles = (r_width + r_gap) * arange(start=-half_count, stop=half_count+1, dim='blade')
    if not isclose(angles['blade', half_count], scalar(0., unit='radian')):
        print(f"{beta = }\n{radius = }\n{count = }\n{width = }\n{r_width = }\n{r_gap = }\n{angles = :c}")
        raise RuntimeError(f"Central angle should be zero but is {angles[count>>1]}.")
    if gap is None and not isclose(angles[-1], beta - 0.5 * r_width):
        raise RuntimeError("Last angle should be half a radial-width from the maximum coverage angle!")
    return angles


def rowland_blades(source: Variable, position: Variable, focus: Variable, alpha: Variable, width: Variable, count: int,
                   tau: Variable, gap=None):
    center, radius, normal = three_point_circle(source, position, focus)
    beta = transfer_angular_range(source, position, center, alpha)
    angles = rowland_blade_angles(beta, radius, count, width, gap)

    # for each angle, create the rotation matrix needed to rotate (position - center)
    from scipp.spatial import rotations_from_rotvecs
    rotations = rotations_from_rotvecs(rotation_vectors=angles * normal)

    # *hopefully broadcast* the rotations to the center-to-analyzer-position vector
    blade_positions = rotations * (position - center) + center

    # find the crystal-normal directions; the central one is set by Bragg's law, while the remaining ones
    # are rotated by half their Rowland angles
    rotations = rotations_from_rotvecs(rotation_vectors=0.5 * angles * normal)
    blade_taus = rotations * tau

    return blade_positions, blade_taus
