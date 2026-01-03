import h5py
import os


def inspect_file(path):
    print(f"--- Inspecting {path} ---")
    try:
        with h5py.File(path, "r") as f:

            def print_attrs(name, obj):
                print(name)
                for key, val in obj.attrs.items():
                    print(f"  {name} attr: {key} = {val}")

            f.visititems(print_attrs)
            # Also print keys of root
            print("Root keys:", list(f.keys()))
            # Inspect data shape of one dataset if available
            if "data" in f:
                print("data shape:", f["data"].shape)

            # Check for generic data keys often used in NMR
            possible_keys = ["data", "acquisition", "fid", "signal"]
            for k in possible_keys:
                if k in f:
                    print(
                        f"Found {k}, shape: {f[k].shape if hasattr(f[k], 'shape') else 'group'}"
                    )
    except Exception as e:
        print(f"Error reading {path}: {e}")


base_dir = r"c:\Users\shoha\git\NMR\data"
# Check one t2 file
t2_files = os.listdir(os.path.join(base_dir, "t2"))
if t2_files:
    inspect_file(os.path.join(base_dir, "t2", t2_files[0]))

# Check one t2multiple file
t2m_files = os.listdir(os.path.join(base_dir, "t2multiple"))
if t2m_files:
    inspect_file(os.path.join(base_dir, "t2multiple", t2m_files[0]))
