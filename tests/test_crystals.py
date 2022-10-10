def test_crystals():
    import pytest
    from scipp import vector, scalar, isclose
    from math import sqrt, pi
    from nicess.crystals import IdealCrystal
    p = vector([0, 0, 0], unit='m')
    t = vector([0, 1, 0], unit='1/angstrom')

    ic = IdealCrystal(p, t)

    assert ic.momentum == scalar(1, unit='1/angstrom')
    assert ic.momentum_vector == vector([0, -1, 0], unit='1/angstrom')

    assert isclose(ic.scattering_angle(k=scalar(1, unit='1/angstrom')), 2 * scalar(pi/6, unit='rad'))

    with pytest.raises(RuntimeError):
        ic.scattering_angle(k=scalar(0.5-1e-10, unit='1/angstrom'))
