from scipp import vector
from nicess.bifrost import BIFROST

from importlib.util import find_spec
import pytest
run_if_h5py_present = pytest.mark.skipif(find_spec('h5py') is None, reason="h5py required for IO tests")


def test_tank_from_calibration():
    from nicess.bifrost import Tank
    tank = Tank.from_calibration()

    print(tank)


def test_bifrost_from_calibration():
    from nicess.bifrost import BIFROST
    bifrost = BIFROST.from_calibration()
    print(bifrost)


# @run_if_h5py_present
# def test_bifrost_secondary_io():
#     from pathlib import Path
#     import tempfile
#     import h5py
#     from nicess.bifrost import BIFROST
#     from nicess.secondary import IndirectSecondary
#     bifrost = BIFROST.from_calibration()
#
#     with tempfile.TemporaryDirectory() as td:
#         fp = Path(td).joinpath("bifrost_io.h5")
#         f = h5py.File(fp, "w")
#         bifrost.secondary.add_to_hdf(f)
#         f.close()
#
#         f = h5py.File(fp, 'r')
#         read = IndirectSecondary.from_hdf(f['IndirectSecondary'])
#         f.close()
#
#     assert read.approx(bifrost.secondary)