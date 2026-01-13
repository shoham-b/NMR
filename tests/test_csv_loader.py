import numpy as np
from nmr_analysis.io.loader import CSVLoader


def test_csv_loader_basic(tmp_path):
    # Create valid CSV content
    # Col 0: MetaName, Col 1: MetaVal, Col 2: Empty, Col 3: Time, Col 4: Ch1, Col 5: Ch2 (Signal)
    # First 2 rows of data cols are headers.

    csv_content = """MetaKey1,10.5,,HeaderTime,HeaderCh1,HeaderCh2
MetaKey2,StringVal,,UnitTime,UnitCh1,UnitCh2
MetaKey3,20,,0.0,1.0,5.0
,,,0.1,1.1,4.5
,,,0.2,1.2,4.0
"""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)

    loader = CSVLoader()
    data = loader.load(csv_file)

    # Check Metadata
    assert data.metadata["MetaKey1"] == 10.5
    assert data.metadata["MetaKey2"] == "StringVal"
    assert data.metadata["MetaKey3"] == 20.0

    # Check Data
    # Should skip first 2 rows of data columns
    # Row 3 (0-based index 2) -> 0.0, 5.0
    # Row 4 -> 0.1, 4.5
    # Row 5 -> 0.2, 4.0

    # User changed loader to read column_5 (Index 4, Ch1 in this mock)
    # Row 3 (0-based index 2) -> 0.0, 1.0 (Ch1)
    # Row 4 -> 0.1, 1.1
    # Row 5 -> 0.2, 1.2

    expected_time = np.array([0.0, 0.1, 0.2])
    expected_signal = np.array([1.0, 1.1, 1.2])

    np.testing.assert_array_almost_equal(data.time, expected_time)
    np.testing.assert_array_almost_equal(data.signal, expected_signal)


def test_csv_loader_ragged(tmp_path):
    # Test case where metadata ends early or data continues
    csv_content = """Meta1,1,,HeaderTime,HeaderCh1,HeaderCh2
Meta2,2,,Unit,Unit,Unit
,,,0.0,0,10.0
,,,0.1,0,9.0
"""
    csv_file = tmp_path / "test_ragged.csv"
    csv_file.write_text(csv_content)

    loader = CSVLoader()
    data = loader.load(csv_file)

    assert data.metadata["Meta1"] == 1.0
    assert len(data.time) == 2
    assert len(data.time) == 2
    # Ch1 (index 4) has 0 in this mock (Col 4)
    assert data.signal[0] == 0.0


def test_csv_loader_channels(tmp_path):
    # Tests that selecting Channel 2 reads Col 5 (index 5) instead of Col 4
    csv_content = """MetaKey1,10.5,,HeaderTime,HeaderCh1,HeaderCh2
MetaKey2,StringVal,,UnitTime,UnitCh1,UnitCh2
MetaKey3,20,,0.0,1.0,5.0
,,,0.1,1.1,4.5
,,,0.2,1.2,4.0
"""
    csv_file = tmp_path / "test_channels.csv"
    csv_file.write_text(csv_content)

    # 1. Default (Channel 1) -> Should read 1.0, 1.1, 1.2
    loader1 = CSVLoader(channel="Channel 1")
    data1 = loader1.load(csv_file)
    assert data1.signal[0] == 1.0

    # 2. Channel 2 -> Should read 5.0, 4.5, 4.0
    loader2 = CSVLoader(channel="Channel 2")
    data2 = loader2.load(csv_file)
    assert data2.signal[0] == 5.0
