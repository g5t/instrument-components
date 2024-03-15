from typing import Callable
from dataclasses import dataclass
from scipp import Variable, DataGroup


def _inner_faces(obj, normal, axis, tol=1):
    """
    Select the major interior faces pointing in the given direction

    A major interior face has its center at ~0 in the direction perpendicular
    to its normal and the specified axis. (This might not be good enough)
    """
    from cadquery import DirectionSelector
    perp = normal.cross(axis)
    perp /= perp.Length
    return [f for f in obj.faces(DirectionSelector(normal)) if abs(f.Center().dot(perp)) < tol and f.Center().dot(normal) < 0]


def _convert_cadquery_workplane(work, m_values: dict[str, float]):
    from cadquery import Vector
    from numpy import unique, array, argmin, abs, sum, einsum, sqrt
    axis = Vector(0, 0, 1)

    def side(x, y):
        return [i for obj in work.objects for i in _inner_faces(obj, Vector(x, y, 0), axis)]

    sides = dict(left=side(1, 0), right=side(-1, 0), bottom=side(0, 1), top=side(0, -1))

    # collect all vertices, only keeping the unique entries would be nice but fraught due to sorting and rounding
    vertices = array([[v.X, v.Y, v.Z] for s in sides.values() for f in s for v in f.vertices()])

    def indexes(face):
        # note Wires method used, not wires (which returns unordered edges?)
        face_verts = [array((v.X, v.Y, v.Z)) for v in [w for wire in face.Wires() for w in wire.vertices()]]
        return [argmin(einsum('ij,ij->i', vertices - v, vertices - v)) for v in face_verts]

    faces = [f for name, side in sides.items() for f in [DependentSupermirror(indexes(face), m_values[name]) for face in side]]
    return vertices, faces


def _convert_step_representation(file, m_values: dict[str, float]):
    from cadquery import importers
    return _convert_cadquery_workplane(importers.importStep(file), m_values)


def _get_step_units(file):
    # This is way too complex to work out now
    # extract the units used in the step file:
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TColStd import TColStd_SequenceOfAsciiString
    len_unit, ang_unit, sang_unit = [TColStd_SequenceOfAsciiString() for _ in range(3)]
    reader = STEPControl_Reader()
    reader.ReadFile(file)
    reader.FileUnits(len_unit, ang_unit, sang_unit)
    if len_unit.IsEmpty():
        raise RuntimeError('No file-defined length units. Please supply the units.')
    unit = str(len_unit.First())
    return unit


def _curvature(radius, position, offset, normal):
    from scipp.spatial import rotations_from_rotvecs
    from scipp import vector, dot, scalar
    from uuid import uuid4
    z_hat = vector(value=[0, 0, 1])
    z_part = dot(position, z_hat)
    rot_vecs = z_part / radius * normal * scalar(1, unit='radian')
    sizes = rot_vecs.sizes
    to = str(uuid4())
    rots = rotations_from_rotvecs(rot_vecs.flatten(to=to)).fold(dim=to, sizes=sizes)
    return rots * (position - z_part * z_hat - radius * offset) + radius * offset


def _horizontal_curvature(radius, position):
    """Transform the tube-along z positions to be on the line of constant curvature around the y-axis"""
    from scipp import vector
    return _curvature(radius, position, vector(value=[1, 0, 0]), vector(value=[0, 1, 0]))


def _vertical_curvature(radius, position):
    """Transform the tube-along z positions to be on the line of constant curvature around the x-axis"""
    from scipp import vector
    return _curvature(radius, position, vector(value=[0, 1, 0]), vector(value=[1, 0, 0]))

#
# def _both_curvature(radius_h, radius_v, position):
#     from scipp.spatial import rotations_from_rotvecs
#     from scipp import vector, dot, scalar
#     x_hat = vector(value=[1, 0, 0])
#     y_hat = vector(value=[0, 1, 0])
#     z_hat = vector(value=[0, 0, 1])
#     z_part = dot(position, z_hat)
#     v_rots = rotations_from_rotvecs(z_part / radius_v * x_hat * scalar(1, unit='radian'))
#     h_rots = rotations_from_rotvecs(z_part / radius_h * y_hat * scalar(1, unit='radian'))
#     return h_rots * (v_rots * (position - z_part * z_hat - radius_v * y_hat) + radius_v * y_hat - radius_h * x_hat) + radius_h * x_hat
#

@dataclass
class DependentSupermirror:
    indexes: list[int]
    m: float
    alpha: float = 6.07  # value used in the CSPEC primary simulation
    w: float = 0.003  # value used in the CSPEC primary simulation for non-Vertical_Bender components

    def to_roff(self) -> str:
        """Return the 'r-interoff' face line as specified in r-interoff-lib.c

        ... the sequence of vertices composing each face, contains also the following float-based parameters:
        m, alpha and W values, **** IN THIS ORDER ***
        """
        face = str(len(self.indexes)) + " " + " ".join(str(x) for x in self.indexes)
        extra = " ".join(str(x) for x in (self.m, self.alpha, self.w))
        return face + " " + extra


@dataclass
class OFFGuide:
    vertices: Variable
    faces: list[DependentSupermirror]
    at: Variable | None = None
    chi: Variable | None = None
    phi: Variable | None = None

    def __post_init__(self):
        from scipp import vector, scalar
        if not isinstance(self.at, Variable):
            self.at = vector(value=[0, 0, 0], unit='m')
        if not isinstance(self.chi, Variable):
            self.chi = scalar(0., unit='degree')
        if not isinstance(self.phi, Variable):
            self.phi = scalar(0., unit='degree')

    @classmethod
    def from_step(cls,
                  file,
                  m_left: float = 1, m_right: float = 1, m_top: float = 1, m_bottom: float = 1,
                  unit: str | None = None):
        """
        Extract guide surfaces from a STEP file

        :param file: The filename/path
        :param m_left: m-value for guide faces with center.x < 0 and norm.x > 0
        :param m_right: m-value for guide faces with center.x > 0 and norm.x < 0
        :param m_top: m-value for guide faces with center.y > 0 and norm.y < 0
        :param m_bottom: m-value for guide faces with center.y < 0 and norm.y > 0
        :param unit: length unit used in STEP file (default: mm)
        :return:
        """
        from scipp import vectors
        if str is None:
            unit = 'mm'
        v, faces = _convert_step_representation(file, dict(left=m_left, right=m_right, top=m_top, bottom=m_bottom))
        return cls(vectors(values=v, dims=['vertices'], unit=unit), faces)

    @classmethod
    def from_parameters(cls, m_left: float = 1, m_right: float = 1, m_top: float = 1, m_bottom: float = 1,
                        width: Variable | Callable[[Variable], Variable] | None = None,
                        height: Variable | Callable[[Variable],  Variable] | None = None,
                        length: Variable = 0, segments: int | None = None, maximum_angle_deviation: float = 0.0005,
                        radius_v: Variable | None = None, radius_h: Variable | None = None,
                        at: Variable | None = None, phi: Variable | None = None, chi: Variable | None = None
                        ):
        from scipp import concat, linspace, arange, array, isfinite, scalar, min, ceil, any, abs, vector
        from numpy import array as narr, inf
        if any(length <= (0 * length)).value:
            raise RuntimeError("Guide length must be positive definite")

        if segments is None:
            rh = abs(scalar(inf, unit='m') if radius_h is None else radius_h)
            rv = abs(scalar(inf, unit='m') if radius_v is None else radius_v)
            r = min(concat((rh, rv), dim='direction'))
            c = scalar(1, unit='radian') / scalar(maximum_angle_deviation, unit='degree').to(unit='radian')
            scipp_segments = 1 + ceil(c * length.to(unit=r.unit) / r)
            segments = scipp_segments.astype('int').value

        def half_width(path_length):
            if callable(width):
                return width(path_length / length)/2
            elif isinstance(width, (Variable, int, float)):
                return width/2 + 0 * path_length
            raise RuntimeError("Guide width must be defined as a scalar or scalar-functor of path-length / length")

        def half_height(path_length):
            if callable(height):
                return height(path_length / length)/2
            elif isinstance(height, (Variable, int, float)):
                return height/2 + 0 * path_length
            raise RuntimeError("Guide height must be defined as a scalar or scalar-functor of path-length / length")

        path = linspace(start=0.0, stop=1.0, dim='path', num=segments+1) * length
        hw = half_width(path) * vector(value=[1, 0, 0])
        hh = half_height(path) * vector(value=[0, 1, 0])
        path = path * vector(value=[0, 0, 1])

        tube = concat((path - hw - hh, path + hw - hh, path + hw + hh, path - hw + hh), dim='ring')

        # if radius_h is not None and radius_v is not None:
        #     tube = _both_curvature(radius_h, radius_v, tube)
        # to first-order at least, handle both curvatures separately
        if isinstance(radius_h, Variable) and isfinite(radius_h).all():
            tube = _horizontal_curvature(radius_h, tube)
        if isinstance(radius_v, Variable) and isfinite(radius_v).all():
            tube = _vertical_curvature(radius_v, tube)
        vertices = tube.flatten(dims=['ring', 'path'], to='vertices')

        rings = arange(start=0, stop=segments, dim='ring')
        faces = array(values=narr([[1, 0, 4, 5], [0, 3, 7, 4], [3, 2, 6, 7], [2, 1, 5, 6]]), dims=['face', 'vertices'])
        faces = faces + 4 * rings  # += doesn't allow shape broadcasting
        faces = faces.transpose(dims=['vertices', 'face', 'ring']).flatten(dims=['face', 'ring'], to='face')
        m_values = array(values=[m_bottom, m_right, m_top, m_left], dims=['face']) * rings
        m_values = m_values.transpose(dims=['face', 'ring']).flatten(dims=['face', 'ring'], to='face')

        n_face = faces.sizes['face']  # == 4 * segments, right?
        dsm = [DependentSupermirror(list(faces['face', f].values), m=m_values['face', f].value) for f in range(n_face)]
        return cls(vertices=vertices, faces=dsm, at=at, chi=chi, phi=phi)

    def to_roff(self, unit: str | None = None):
        if unit is None:
            unit = 'm'
        vertices = self.vertices if unit is None else self.vertices.to(unit=unit)

        stream = f"OFF\n{vertices.sizes['vertices']} {len(self.faces)} 0\n"
        stream += "\n".join(" ".join(str(x) for x in v) for v in vertices.values) + "\n"
        stream += "\n".join(f.to_roff() for f in self.faces)
        return stream
