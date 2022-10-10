from scipp import vector
from nicess.secondary import DirectSecondary

from importlib.util import find_spec
import pytest

run_if_h5py_present = pytest.mark.skipif(find_spec('h5py') is None, reason="h5py required for IO tests")


def test_direct_secondary():
    from nicess.detectors import DiscreteTube
    from scipp import scalar
    at = vector([-0.1, 0, -0.2], unit='mm')
    to = vector([0.2, 200, -0.4], unit='mm')
    tube = DiscreteTube(at, to, 100, 0.25)

    dgs = DirectSecondary([tube, tube])

    assert dgs.sample_at == vector([0, 0, 0], unit='m')


@run_if_h5py_present
def test_direct_secondary_io():
    from pathlib import Path
    import tempfile
    import h5py
    from nicess.detectors import DiscreteTube
    at = vector([-0.1, 0, -0.2], unit='mm')
    to = vector([0.2, 200, -0.4], unit='mm')
    tube = DiscreteTube(at, to, 100, 0.25)

    dgs = DirectSecondary([tube, tube], vector([0.1, 0.2, 0.3], unit='um'))

    with tempfile.TemporaryDirectory() as td:
        fp = Path(td).joinpath('wire_io.h5')
        f = h5py.File(fp, 'w')
        dgs.add_to_hdf(f)
        f.close()
        f = h5py.File(fp, 'r')
        read = DirectSecondary.from_hdf(f['DirectSecondary'])
        f.close()

    # allow for floating point inaccuracies due to 64-bit -> 32-bit -> 64-bit round-trip
    assert dgs.approx(read)
