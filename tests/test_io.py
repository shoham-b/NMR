import h5py
import numpy as np
import pytest
from pathlib import Path
from nmr_analysis.io.loader import KeysightLoader
from nmr_analysis.analysis.processing import parse_time_from_filename


def test_loader_robustness(tmp_path):
    # Create valid synthetic hdf5
    file_path = tmp_path / "test_valid.h5"
    with h5py.File(file_path, "w") as f:
        # Create different structures
        # 1. Standard
        grp = f.create_group("Waveforms/Channel 1/Waveform 1")
        grp.create_dataset("RawData", data=np.array([1, 2, 3]))
        grp.attrs["XIncrement"] = 0.1

    loader = KeysightLoader(channel="Channel 1")
    data = loader.load(file_path)
    assert len(data.signal) == 3
    assert data.sample_rate == pytest.approx(10.0)

    # 2. Weird structure - no RawData but a dataset exists
    file_path2 = tmp_path / "test_weird.h5"
    with h5py.File(file_path2, "w") as f:
        grp = f.create_group("Waveforms/Channel 1/Waveform 1")
        grp.create_dataset("SomeOtherData", data=np.array([4, 5, 6]))
        grp.attrs["XIncrement"] = 0.5

    data2 = loader.load(file_path2)
    assert len(data2.signal) == 3

    # 3. Direct Dataset (unlikely for Keysight but robust check)
    # file_path3 = tmp_path / "test_direct.h5"
    # with h5py.File(file_path3, "w") as f:
    #     f.create_group("Waveforms/Channel 1")
    #     f["Waveforms/Channel 1/Waveform 1"] = np.array([7, 8, 9])
    #     # Cannot set attrs on dataset like that easily in one line without ref
    #     dset = f["Waveforms/Channel 1/Waveform 1"]
    #     dset.attrs["XIncrement"] = 1.0

    # data3 = loader.load(file_path3)
    # assert len(data3.signal) == 3


def test_filename_parsing():
    assert parse_time_from_filename("data_10ms.h5") == 0.01
    assert parse_time_from_filename("T1_0_005.HDF5") == 0.005
    assert parse_time_from_filename("0_005.HDF5") == 0.005
    assert parse_time_from_filename("200us_data") == 0.0002
    assert parse_time_from_filename("random_name") == 0.0
