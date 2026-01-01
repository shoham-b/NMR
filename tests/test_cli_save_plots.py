import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from typer.testing import CliRunner
from nmr_analysis.cli.commands import app
from nmr_analysis.core.types import ExperimentType, AnalysisResult, NMRData
import numpy as np

runner = CliRunner()


@pytest.fixture
def mock_loader():
    with patch("nmr_analysis.cli.commands.get_loader") as mock:
        loader_instance = mock.return_value
        data = MagicMock(spec=NMRData)
        data.time = np.linspace(0, 1, 100)
        data.signal = np.random.normal(0, 1, 100)
        data.metadata = {"tau": 0.01}  # default empty metadata -> with tau
        loader_instance.load.return_value = data
        yield mock


@pytest.fixture
def mock_fitter():
    with patch("nmr_analysis.cli.commands.Fitter") as mock:
        mock.fit_t2.return_value = (
            {"M0": 1, "T2": 0.5},
            np.zeros(10),
            np.zeros(10),
            0.99,
            {"M0": 0.1, "T2": 0.05},
        )
        mock.fit_t1.return_value = (
            {"M0": 1, "T1": 0.5},
            np.zeros(10),
            np.zeros(10),
            0.99,
            {"M0": 0.1, "T1": 0.05},
        )
        mock.fit_t2_star.return_value = AnalysisResult(
            experiment_type=ExperimentType.T2_STAR,
            dataset_name="Test T2*",
            params={"T2*": 0.1},
            fit_curve=np.zeros(100),
            residuals=np.zeros(100),
            r_squared=0.98,
        )
        yield mock


@pytest.fixture
def mock_plt():
    with patch("nmr_analysis.cli.commands.plt") as mock:
        # Configure subplots to return iterable mocks based on ncols
        def subplots_side_effect(*args, **kwargs):
            # subplots(nrows, ncols, ...)
            # We assume nrows=1 for our usage
            ncols = 1
            if len(args) >= 2:
                ncols = args[1]
            elif "ncols" in kwargs:
                ncols = kwargs["ncols"]

            fig = MagicMock()
            axes = tuple(MagicMock() for _ in range(ncols))

            mock.last_fig = fig
            mock.last_axes = axes

            # If ncols > 1, returns (fig, axes_array/tuple)
            # If ncols == 1, returns (fig, ax) usually, but we only use >1 here
            return fig, axes

        mock.subplots.side_effect = subplots_side_effect
        yield mock


@pytest.fixture
def mock_processing():
    with patch("nmr_analysis.cli.commands.preprocess_data") as mock_prep:
        # Returns: processed_data, tau, amp, peak_info
        # processed_data needs to be NMRData
        data = MagicMock(spec=NMRData)
        data.time = np.linspace(0, 1, 100)
        data.signal = np.ones(100)
        data.metadata = {"time_unit": "s"}

        mock_prep.return_value = (data, 0.1, 1.0, {"p1_idx": 0, "fit_idx": 10})
        yield mock_prep


@pytest.fixture
def mock_cm():
    with patch("nmr_analysis.cli.commands.cm") as mock:
        yield mock


def test_save_plots_t1(
    mock_loader, mock_fitter, mock_plt, mock_processing, mock_cm, tmp_path
):
    input_dir = tmp_path / "t1_data"
    input_dir.mkdir()
    (input_dir / "test1.h5").touch()
    output_dir = tmp_path / "output"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(input_dir),
            "--type",
            "t1",
            "--save-plots",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert mock_plt.savefig.called
    args, _ = mock_plt.savefig.call_args
    filepath = args[0]
    assert "t1_data_t1_fit.png" in str(filepath)
    assert str(output_dir) in str(filepath)


def test_save_plots_default_dir(
    mock_loader, mock_fitter, mock_plt, mock_processing, mock_cm, tmp_path
):
    input_dir = tmp_path / "t2_data"
    input_dir.mkdir()
    (input_dir / "test2.h5").touch()

    with patch("nmr_analysis.cli.commands.Path.mkdir") as mock_mkdir:
        result = runner.invoke(
            app, ["analyze", str(input_dir), "--type", "t2", "--save-plots"]
        )

    assert result.exit_code == 0
    assert mock_plt.savefig.called


def test_no_save_plots(
    mock_loader, mock_fitter, mock_plt, mock_processing, mock_cm, tmp_path
):
    input_dir = tmp_path / "t1_data"
    input_dir.mkdir()
    (input_dir / "test1.h5").touch()

    result = runner.invoke(app, ["analyze", str(input_dir), "--type", "t1"])

    assert result.exit_code == 0
    assert not mock_plt.savefig.called
    assert mock_plt.show.called


def test_plot_units_verification(
    mock_loader, mock_fitter, mock_plt, mock_processing, mock_cm, tmp_path
):
    input_dir = tmp_path / "t1_data_units"
    input_dir.mkdir()
    (input_dir / "test.h5").touch()

    loader_instance = mock_loader.return_value
    data_mock = MagicMock(spec=NMRData)
    data_mock.time = np.linspace(0, 1, 100)
    data_mock.signal = np.ones(100)
    data_mock.metadata = {"time_unit": "ms", "tau": 0.01}
    loader_instance.load.return_value = data_mock

    result = runner.invoke(
        app, ["analyze", str(input_dir), "--type", "t1", "--save-plots"]
    )
    assert result.exit_code == 0

    # Ensure Axes were created
    assert mock_plt.last_axes, "No axes captured"

    # Check verify set_xlabel was called with correct unit on at least one axis
    found_unit = False
    for ax in mock_plt.last_axes:
        for call in ax.set_xlabel.call_args_list:
            args, _ = call
            if "ms" in args[0]:
                found_unit = True
                break
    assert found_unit, "Expected 'ms' in set_xlabel call on axes"


def test_plot_t2_combined_label_unit(
    mock_loader, mock_fitter, mock_plt, mock_processing, mock_cm, tmp_path
):
    input_dir = tmp_path / "t2combined_units"
    input_dir.mkdir()
    (input_dir / "test.h5").touch()

    loader_instance = mock_loader.return_value
    data_mock = MagicMock(spec=NMRData)
    data_mock.time = np.linspace(0, 1, 100)
    data_mock.signal = np.ones(100)
    data_mock.metadata = {"time_unit": "my_unit"}
    loader_instance.load.return_value = data_mock

    with patch("nmr_analysis.cli.commands.extract_echo_train") as mock_echo:
        mock_echo.return_value = (np.array([1, 2, 3, 4]), np.array([1, 0.8, 0.6, 0.4]))

        result = runner.invoke(
            app, ["analyze", str(input_dir), "--type", "t2_combined", "--save-plots"]
        )
        assert result.exit_code == 0

        found = False
        for call in mock_plt.plot.call_args_list:
            _, kwargs = call
            if "label" in kwargs and "T2 Fit" in kwargs["label"]:
                label = kwargs["label"]
                assert "my_unit" in label, (
                    f"Expected 'my_unit' in fit label, got '{label}'"
                )
                found = True
                break
        assert found, "Did not find T2 Fit plot call"
