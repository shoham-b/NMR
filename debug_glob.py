from pathlib import Path
import sys

p_str = sys.argv[1]
p = Path(p_str)

print(f"Path: {p}")
print(f"Exists: {p.exists()}")
print(f"Is Dir: {p.is_dir()}")

if p.is_dir():
    print("Globbing '*':")
    files = list(p.glob("*"))
    print(f"Found {len(files)} items.")
    for f in files:
        print(f" - {f.name} (suffix: {f.suffix})")

    target_files = [f for f in files if f.suffix.lower() in [".h5", ".hdf5", ".csv"]]
    print(f"Filtered targets: {len(target_files)}")
