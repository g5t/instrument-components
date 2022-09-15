from scipp import vector


def test_wire():
    from nicess.detectors import Wire
    at = vector([0, 0, 0], unit='m')
    to = vector([0, 0, 1], unit='m')

    wire = Wire(at, to)

    assert wire.center_of_mass == (at + to)/2
    assert wire.charge_position(10, 90) == 0.1 * at + 0.9 * to

    assert wire.charge_position(0, 100) == to
    assert wire.charge_position(33, 0) == at


def test_discrete_wire():
    from nicess.detectors import DiscreteWire
    at = vector([1, 2, 3], unit='mm')
    to = vector([104, 99, 30], unit='mm')
    wire = DiscreteWire(at, to, 3)

    # The middle discrete bin centered on the center-point (if there are an odd number of bins)
    assert wire.center_of_mass == wire.index_position(1)

    wire = DiscreteWire(at, to, 101)
    # The end bins are not centered on the ends of the detector
    assert wire.index_position(0) != at
    assert wire.index_position(0) == at + (to - at) * 0.5 / 101

    assert wire.index_position(100) != to
    assert wire.index_position(100) == at + (to - at) * 100.5 / 101


def test_discrete_tube():
    from nicess.detectors import DiscreteTube
    from numpy import random, pi
    at = vector(10 * random.rand(3), unit='mm')
    to = vector(100 + 10 * random.rand(3), unit='mm')
    wire = DiscreteTube(at, to, 3, pi)

    assert wire.radius == pi


def test_he3_tube():
    from nicess.detectors import He3Tube
    from numpy import random, pi
    at = vector(10 * random.rand(3), unit='mm')
    to = vector(100 + 10 * random.rand(3), unit='mm')
    wire = He3Tube(at, to, 3, pi, 10.)

    assert wire.pressure == 10.
