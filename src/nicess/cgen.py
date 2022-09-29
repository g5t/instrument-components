from numpy import ndarray
from typing import Union
from pathlib import Path


def array_to_c(name: str, data: ndarray):
    # this needs more error checking, but is good enough for now?
    if 'float32' in data.dtype.name:
        ctype = 'float'
        fmt = 'g'
    elif 'float64' in data.dtype.name:
        ctype = 'double'
        fmt = 'g'
    elif 'int' in data.dtype.name:
        ctype = 'int'
        fmt = 'd'
    else:
        raise RuntimeError(f"What should I do with a {data.dtype.name} array?")

    size = "".join(f"[{s:d}]" for s in data.shape)

    str = f"{ctype} {name}{size} =\n"

    from numpy import nditer
    itr = nditer(data, flags=['multi_index'])
    lidx = [-1 for _ in data.shape]
    depth = data.ndim
    for x in itr:
        tidx = [i if i != l else -1 for i, l in zip(itr.multi_index, lidx)]
        str += "{" * sum(i == 0 for i in tidx)

        str += "{x:{fmt}}".format(x=x, fmt=fmt)

        # str += f"{itr.multi_index}"

        r = reversed([i + 1 == d for i, d in zip(itr.multi_index, data.shape)])
        nend = 0
        for idx, tf in enumerate(r):
            if tf:
                nend += 1
            else:
                break
        else:
            # No break, so all indexes are at the end!
            str += "}" * data.ndim + ";"

        # str += f"{nend}"

        # str += "\n"

        if nend == 0:
            str += ", "
        elif nend < data.ndim:
            str += "}" * nend + ", "

        lidx = itr.multi_index

    return split_long_lines(str)


def split_long_lines(s: str, width=80):
    parts = s.split('\n')
    outparts = []
    for part in parts:
        if len(part) < width:
            outparts.append(part)
        else:
            idx = 1
            while idx > 0 and len(part) >= width:
                for idx in reversed(range(width)):
                    if part[idx] == ' ' or part[idx] == ',' or part[idx] == '}':
                        break
                if idx > 0:
                    outparts.append(part[:idx+1])
                    part = part[idx+1:]
            outparts.append(part)
    return '\n'.join(outparts)


def arrays_to_file(arrays: dict[str, ndarray], filename: Union[str, Path], overwrite=False):
    values = '\n'.join([array_to_c(k, v) for k, v in arrays.items()])

    if not isinstance(filename, Path):
        filename = Path(filename)

    if filename.exists() and not overwrite:
        raise RuntimeError(f"File {filename} exists. Pass overwrite=True to overwrite")

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)

    with open(filename, 'w') as f:
        f.write(values)

    return values