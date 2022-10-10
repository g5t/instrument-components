from scipp import Variable


def has_compatible_unit(x: Variable, unit):
    from scipp import UnitError
    try:
        x.to(unit=unit, copy=False)
    except UnitError:
        return False
    return True


def is_scalar(x: Variable):
    from scipp import DimensionError
    try:
        y = x.value
    except DimensionError:
        return False
    return True
