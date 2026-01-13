import pytest
from nmr_analysis.io.loader import get_loader, CSVLoader, KeysightLoader


def test_get_loader_csv(tmp_path):
    f = tmp_path / "test.csv"
    f.touch()
    loader = get_loader(f)
    assert isinstance(loader, CSVLoader)


def test_get_loader_h5(tmp_path):
    f = tmp_path / "test.h5"
    f.touch()
    loader = get_loader(f)
    assert isinstance(loader, KeysightLoader)


def test_get_loader_hdf5(tmp_path):
    f = tmp_path / "test.hdf5"
    f.touch()
    loader = get_loader(f)
    assert isinstance(loader, KeysightLoader)


def test_get_loader_invalid(tmp_path):
    f = tmp_path / "test.txt"
    f.touch()
    with pytest.raises(ValueError, match="Unsupported file extension"):
        get_loader(f)
