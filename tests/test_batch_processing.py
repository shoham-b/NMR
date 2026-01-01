import pytest
from pathlib import Path
from unittest.mock import MagicMock, call, patch
from nmr_analysis.cli.commands import analyze, ExperimentType


@pytest.fixture
def mock_run_analysis():
    with patch("nmr_analysis.cli.commands._run_analysis") as m:
        m.return_value = []
        yield m


def test_batch_processing_aliases_and_nesting(tmp_path, mock_run_analysis):
    # Setup directory structure
    root = tmp_path / "root"
    root.mkdir()

    # 1. Flat T2 folder with alias "T2_single"
    t2_flat = root / "T2_single"
    t2_flat.mkdir()
    (t2_flat / "data.h5").touch()

    # 2. Nested T2 Combined with alias "t2_multiple" (case insensitive)
    t2_nested_root = root / "t2_multiple"
    t2_nested_root.mkdir()

    sample_a = t2_nested_root / "SampleA"
    sample_a.mkdir()
    (sample_a / "data.csv").touch()

    sample_b = t2_nested_root / "SampleB"
    sample_b.mkdir()
    (sample_b / "data.h5").touch()

    # 3. Flat T2* alias "t2~"
    t2_star = root / "t2~"
    t2_star.mkdir()
    (t2_star / "data.h5").touch()

    # Run analyze on root
    # We need to simulate save_plots=True to check output path logic if possible,
    # but _run_analysis receives save_path.

    # We invoke analyze directly.
    # Note: analyze is a Typer command, but we can call the function if we import it.
    # However, Type enforces arguments. Let's call it as a python function.

    output_dir = tmp_path / "output"
    analyze(
        path=root,
        experiment=None,
        channel="Channel 1",
        plot=False,
        save_plots=True,
        output_dir=output_dir,
        interactive=False,
    )

    # Check calls to _run_analysis
    assert mock_run_analysis.call_count == 4

    # Expected calls (order might depend on iteration order, usually safe to check any order)
    # 1. T2_single (Flat) -> output/T2_single
    # 2. t2_multiple/SampleA (Nested) -> output/t2_multiple/SampleA
    # 3. t2_multiple/SampleB (Nested) -> output/t2_multiple/SampleB
    # 4. t2~ (Flat) -> output/t2~

    calls = mock_run_analysis.call_args_list

    # Helper to find call for specific path
    def find_call_for_path(p_name):
        for c in calls:
            # args[0] is path
            if c.args[0].name == p_name:
                return c
        return None

    # Verify T2_single
    c_t2 = find_call_for_path("T2_single")
    assert c_t2
    assert c_t2.args[1] == ExperimentType.T2
    assert c_t2.kwargs["save_path"] == output_dir / "T2_single"

    # Verify SampleA
    c_a = find_call_for_path("SampleA")
    assert c_a
    assert c_a.args[1] == ExperimentType.T2_COMBINED
    assert c_a.kwargs["save_path"] == output_dir / "t2_multiple" / "SampleA"

    # Verify SampleB
    c_b = find_call_for_path("SampleB")
    assert c_b
    assert c_b.args[1] == ExperimentType.T2_COMBINED
    assert c_b.kwargs["save_path"] == output_dir / "t2_multiple" / "SampleB"

    # Verify t2~
    c_star = find_call_for_path("t2~")
    assert c_star
    assert c_star.args[1] == ExperimentType.T2_STAR
    assert c_star.kwargs["save_path"] == output_dir / "t2~"
