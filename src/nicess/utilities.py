from scipp import Variable


def is_type(x, t, name):
    if not isinstance(x, t):
        raise RuntimeError(f"{name} must be a {t}")


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


def draw_triplet_circuit(compact=True):
    import schemdraw
    import schemdraw.elements as elm
    import schemdraw.util as util
    d = schemdraw.Drawing(inches_per_unit=1/2.54)

    pars = dict(length=8, radius=0.5, leadlen=0.4)
    if compact:
        d += elm.Ground()
        hv1 = elm.SourceV().right().label('HV')
        d += hv1
        d.push()
        d += elm.MeterV().reverse().down().label(r'$A$')
        d += elm.Ground()
        d.pop()
        d += elm.Resistor().at(hv1.end).right().label(r'$R_A$')
        d0 = elm.cables.Coax(**pars).right().label(r'$\rho_0$')
        d += d0
        d += elm.Resistor().down().label(r'$R_{01}$')
        d1 = elm.cables.Coax(**pars).reverse().left().label(r'$\rho_1$')
        d += d1
        d += elm.Resistor().down().label(r'$R_{12}$')
        d2 = elm.cables.Coax(**pars).right().label(r'$\rho_2$')
        d += d2
        d += elm.Resistor().right().label(r'$R_{B}$')
        d.push()
        d += elm.SourceV().reverse().right().label('HV')
        d += elm.Ground()
        d.pop()
        d += elm.MeterV().up().label(r'$B$')
        d += elm.Ground().up()
        d += elm.Ground().at(d0.shieldstart)
        d += elm.Ground().at(d1.shieldend_top)
        d += elm.Ground().at(d2.shieldstart)
    else:
        d += elm.Ground()
        hv1 = elm.SourceV().right().label('HV')
        d += hv1
        d.push()
        va = elm.MeterV().reverse().down().label(r'$A$')
        d += va
        d += elm.Ground()
        d.pop()
        d += elm.Resistor().at(hv1.end).right().label(r'$R_A$')
        d0 = elm.cables.Coax(**pars).right().label(r'$\rho_0$')
        d += d0
        d += elm.Resistor().right().label(r'$R_{01}$')
        d1 = elm.cables.Coax(**pars).right().label(r'$\rho_1$')
        d += d1
        d += elm.Resistor().right().label(r'$R_{12}$')
        d2 = elm.cables.Coax(**pars).right().label(r'$\rho_2$')
        d += d2
        d += elm.Resistor().right().label(r'$R_{B}$')
        d.push()
        d += elm.SourceV().reverse().right().label('HV')
        d += elm.Ground()
        d.pop()
        vb = elm.MeterV().down().label(r'$B$')
        d += vb
        d += elm.Ground()
        d += elm.Ground().at(d0.shieldstart)
        d += elm.Ground().at(d1.shieldstart)
        d += elm.Ground().at(d2.shieldstart)
        x0 = va.center + util.Point((1, 0))
        x1 = vb.center - util.Point((1, 0))

        d += elm.lines.Arrow(color='red').at(x1).to(x0).label(r"$\frac{A-B}{A+B}$", loc='bottom', ofst=0.5)
    b = d.get_imagedata()

    return d