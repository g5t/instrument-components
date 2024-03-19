"""Read the guide parameters from a spreadsheet"""


def _scalar_columns(table, dim):
    from scipp import array
    x2s = dict(
        element=('Element #', 'dimensionless'),
        chi=('chi(x)', 'degree'),
        phi=('phi(y)', 'degree'),
        length=('length (mm)', 'mm'),
        trajectory=('trajectory (m)', 'm'),
        curvature_horizontal=('hor. (m)', 'm'),
        curvature_vertical=('vert. (m)', 'm'),
        width_entry=('width entry', 'mm'),
        width_exit=('width exit', 'mm'),
        height_entry=('height entry', 'mm'),
        height_exit=('height exit', 'mm'),
        m_top=('top', 'dimensionless'),
        m_bottom=('bottom', 'dimensionless'),
        m_right=('right', 'dimensionless'),
        m_left=('left', 'dimensionless'),
        substrate=('substrate', 'dimensionless')
        )
    return {k: array(values=table[v[0]].values, unit=v[1], dims=[dim]) for k, v in x2s.items()}


def _vector_columns(table, dim):
    from scipp import vectors
    from numpy import vstack
    x2s = dict(
        ics_at=(('x start', 'y start', 'z start'), 'mm'),
        ics_to=(('x end', 'y end', 'z end'), 'mm'),
        mcstas_at=(('x start.1', 'y start.1', 'z start.1'), 'm'),
        mcstas_to=(('x end.1', 'y end.1', 'z end.1'), 'm'),
    )
    xyz = {k: vstack(list(table[v[0][i]].values for i in range(3))).T for k, v in x2s.items()}
    return {k: vectors(values=xyz[k], unit=v[1], dims=[dim]) for k, v in x2s.items()}


def _column_named(table, name):
    cols = [c for c in table.columns if name in str(c)]
    if len(cols) > 1:
        cols = [c for c in cols if name == str(c)]
    if len(cols) != 1:
        raise RuntimeError(f'No unique column {name} in {table.columns}')
    return cols[0]


def _fix_nboa(table):
    """According to NBOA_Reality_2022.pages.pdf the guide table does not match reality due to manufacturing changes

    The NBOA was specified as four guide segments but the first two were combined into a single element.
    Those segments should have been aligned (chi=0, phi=0) with the NBOA axis, but were rotated according to the
    radii of curvature (specified for the second element?).

    The PDF details how the misalignment could be corrected, and gives a table with optimised values of
        chi, phi, (x, y, z)-start, (x, y, z)-end

    To match reality, replace W03-01-011, W03-01-12 by W03-01-01, and update the parameters for those and
    W03-01-02 and W03-01-03.
    """
    from numpy import argwhere, sqrt
    element, chi, phi = [_column_named(table, x) for x in ('Element', 'chi', 'phi')]
    x0, y0, z0 = [_column_named(table, f'{x} start') for x in ('x', 'y', 'z')]
    x1, y1, z1 = [_column_named(table, f'{x} end') for x in ('x', 'y', 'z')]
    length, hor, ver = [_column_named(table, x) for x in ('length', 'hor', 'vert')]
    width, height = [_column_named(table, x) for x in ('width exit', 'height exit')]
    entry_height = _column_named(table, 'height entry')

    # The parameters for the NBOA guide are (maybe) wrong -- update the row(s) now
    names = 'W03-01-011', 'W03-01-012', 'W03-01-02', 'W03-01-03', 'W03-09-01'
    one_one, one_two, two, three, last = [argwhere(table[element] == name)[0, 0] for name in names]

    # prepare to copy-over W03-01-012 parameters missing from W03-01-011
    one_pars = {k: table[k][one_two] for k in (hor, ver, width, height)}
    one_pars[element] = 'W03-01-01'

    table.loc[[one_one, two, three], [chi, phi]] = [[-0.0123448, -0.0048611], [-0.037, -0.015], [-0.074, -0.030]]
    table.loc[[one_one, two, three], [x0, y0, z0, x1, y1, z1]] = [
        [1879.94, -0.005, 0, 2880.94, 0.0799, -0.22],
        [2880.94, 0.0799, -0.22, 3881.94, 0.3347, -0.87],
        [3881.94, 0.3347, -0.87, 5361.94, 1.1060, -2.77]
    ]
    table.loc[one_one, [length]] = [table[length][one_one] + table[length][one_two]]
    table.loc[two, [length]] = [1000.]

    for k, v in one_pars.items():
        table.loc[one_one, [k]] = [v]

    # The last guide element's height calculation differs from all others and includes invalid references,
    # This is a best-guess replacement for the expected heights:
    f176 = 1.76

    def h_eq(x):
        e3h_start = 124702.42
        e3h_a = 45026.5
        e3h_b = 73.89
        e3h_c = 0.25
        aj141 = 0  # TODO Find out what $AJ$141 _SHOULD HAVE BEEN_
        return sqrt((1 - (x - e3h_start - e3h_c)**2 / e3h_a ** 2) * e3h_b**2) + aj141

    table.loc[last, [entry_height, height]] = [h_eq(table[x0][last]),  h_eq(table[x1][last])]

    new_table = table.drop(index=one_two)
    return new_table


def read_xlsx(file, sheet: str | None = None):
    """This function is tailored to read the CSPEC guide specification spreadsheet, it may not be useful otherwise"""
    from pandas import read_excel, isnull
    from numpy import argwhere
    from scipp import array, Dataset

    # the header information is split over two lines for some columns, so we get not-so-useful names
    table = read_excel(file, sheet or 'Specification', header=1)

    table = _fix_nboa(table)

    # We only need rows that specify values for a guide element
    # Pandas doesn't drop the column dimension when indexing a single column, so argwhere gives 2-element indexes
    table.drop(table[table[_column_named(table, 'Element')].map(isnull)].index, inplace=True)

    # Also drop any rows without m-values (this is only the last chopper gap?)
    table.drop(table[table[_column_named(table, 'top')].map(isnull)].index, inplace=True)

    # The column naming is not great
    kept = _scalar_columns(table, 'element')
    kept.update(_vector_columns(table, 'element'))
    return Dataset(kept)


# def build_primary_guide(xlsx_file, step_directory, file_map: dict[str, str] | None = None, cheat: bool = True):
#     from pathlib import Path
#     from ..guide import OFFGuide
#     if file_map is None:
#         file_map = {}
#
#     if cheat:
#         # some files are missing :(
#         # some guide segments are identical, and only one file was provided
#         for i in range(2, 14):
#             file_map[f'W03-06-{i:02d}'] = 'W03-06-02_13'
#         for i in range(21, 33):
#             file_map[f'W03-06-{i:21d}'] = 'W03-06-21-32'
#         # hyphens vs. underscores are not completely consistent
#         file_map['W03-07-01'] = 'W03-07_01'
#
#     info = read_xlsx(xlsx_file)
#
#     # Partial filenames to match provided STEP files
#     names = {k: k for k in info['element'].values}
#     # Update the names using the (optionally) provided file map
#     names.update(file_map)
#

def _zero_one_linear_interpolator(first, last):
    def interpolator(x):
        return (1 - x) * first + x * last
    return interpolator


def _entry_exit_interpolator(values, name, unit):
    first = values[f'{name}_entry'].data.to(unit=unit)
    last = values[f'{name}_exit'].data.to(unit=unit)
    return _zero_one_linear_interpolator(first, last)


def build_guide(xlsx_file):
    from scipp import isfinite
    from scipp.spatial import linear_transform
    from ..guide import OFFGuide
    info = read_xlsx(xlsx_file)
    n_elements = info['element'].sizes['element']

    # ICS (x, y, z) -> McSTAS (z, x, y)
    r = linear_transform(value=[[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    elements = {}
    for i in range(n_elements):
        values = info['element', i]
        if not isfinite(values['length'].data):
            continue
        elements[values['element'].value] = OFFGuide.from_parameters(
            m_left=values['m_left'].value,
            m_right=values['m_right'].value,
            m_bottom=values['m_bottom'].value,
            m_top=values['m_top'].value,
            width=_entry_exit_interpolator(values, 'width', 'm'),
            height=_entry_exit_interpolator(values, 'height', 'm'),
            length=values['length'].data.to(unit='m'),
            radius_h=values['curvature_horizontal'].data.to(unit='m'),
            radius_v=values['curvature_vertical'].data.to(unit='m'),
            at=r * values['ics_at'].data.to(unit='m'),  # FIXME This should be adjusted for McStas coordinate system
            # at=values['mcstas_at'].data.to(unit='m'),
            chi=values['chi'].data.to(unit='degree'),
            phi=values['phi'].data.to(unit='degree')
        )

    return elements

