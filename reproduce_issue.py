import sys
import traceback
from nmr_analysis.cli.commands import app
from typer.testing import CliRunner

runner = CliRunner()

print("Starting debug run...")
try:
    # Run simple check first
    # result = runner.invoke(app, ["--help"])
    # print("Help check exit code:", result.exit_code)

    # Run actual command
    print("Running analyze data...")
    result = runner.invoke(
        app,
        ["analyze", "data", "--save-plots", "--output-dir", "output/water_analysis"],
    )

    print("Exit code:", result.exit_code)
    print("Stdout:")
    print(result.stdout)
    if result.exception:
        print("\nException captured:")
        print(result.exception)
        traceback.print_exception(*result.exc_info)

except Exception:
    traceback.print_exc()

print("Debug run finished.")
