import sys
import traceback
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from nmr_analysis.cli.commands import app
from typer.testing import CliRunner

# Force Agg backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

runner = CliRunner()

print("Starting diffusion verification (Batch Mode)...")
try:
    # Run batch analysis on 'data' folder
    # This should trigger:
    # 1. Detection of 'data' as Sample-like folder (because it contains t2, t2multiple)
    # 2. Sort t2multiple first
    # 3. Analyze t2multiple -> Capture T2
    # 4. Analyze t2 (mapped to DIFFUSION) -> Use captured T2 as constraint

    print("Running analyze data --save-plots...")
    args = [
        "analyze",
        "data",
        "--save-plots",
        "--output-dir",
        "output/verification_batch",
    ]
    result = runner.invoke(app, args)

    print("Exit code:", result.exit_code)
    print("Stdout:")
    print(result.stdout)
    if result.exception:
        print("\nException captured:")
        print(result.exception)
        traceback.print_exception(*result.exc_info)

except Exception:
    traceback.print_exc()

print("Verification finished.")
