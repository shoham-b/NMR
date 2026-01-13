from pathlib import Path
import sys

p = Path(sys.argv[1])
print(f"Scanning {p}")
for f in p.rglob("*"):
    if f.suffix.lower() in [".h5", ".hdf5", ".csv"]:
        print(f)
