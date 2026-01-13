import sys

fname = sys.argv[1]
try:
    with open(fname, "r", encoding="utf-8-sig") as f:
        print(f"Reading {fname}...")
        for i in range(20):
            line = f.readline()
            if not line:
                break
            print(f"{i}: {line.strip()}")
except Exception as e:
    print(f"Error: {e}")
