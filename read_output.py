import sys

try:
    with open("test_output_2.txt", "r", encoding="utf-16") as f:
        print(f.read())
except Exception as e:
    # Try utf-8 if utf-16 fails
    try:
        with open("test_output_2.txt", "r", encoding="utf-8") as f:
            print(f.read())
    except Exception as e2:
        print(f"Failed to read: {e}, {e2}")
