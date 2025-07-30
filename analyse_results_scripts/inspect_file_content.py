import numpy as np
import argparse

def inspect_npz(path, preview_count=1):
    print(f"Inspecting file: {path}\n")
    data = np.load(path, allow_pickle=True)

    planner_names = set()

    for key in data:
        arr = data[key]
        print(f"Key: '{key}'")
        print(f"  - Type: {type(arr)}")
        print(f"  - Dtype: {arr.dtype}")
        print(f"  - Shape: {arr.shape}")
        
        try:
            preview = arr[:preview_count]
            print(f"  - Preview (first {preview_count}):")
            for i, entry in enumerate(preview):
                print(f"    [{i}]: {type(entry)} -> {str(entry)[:30]}{'...' if len(str(entry)) > 200 else ''}")
                for el, val in entry.items():
                    print(f"      {el}: {str(val)[:20]} (type: {type(val)})")
                
            # Extract planner names if possible
            if isinstance(entry, dict) and 'planner' in entry:
                planner_names.update(e.get('planner') for e in arr if isinstance(e, dict) and 'planner' in e)

        except Exception as e:
            print(f"  - Could not preview entries: {e}")
        print()

    if planner_names:
        print("✅ Unique planner names found:")
        for name in sorted(planner_names):
            print(f"  - {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect contents of an .npz file")
    parser.add_argument("file", help="Path to .npz file")
    parser.add_argument("--count", type=int, default=3, help="How many items to preview per key")
    args = parser.parse_args()

    inspect_npz(args.file, preview_count=args.count)
