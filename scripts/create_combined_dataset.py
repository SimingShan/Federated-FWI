"""
Stack the 4 velocity-model families (foothill, marmousi, overthrust, bpsalt)
into a single combined dataset along the batch (axis-0) dimension.

Output directory: dataset/combined/
"""
import os
import numpy as np

FAMILIES = ["foothill", "marmousi", "overthrust", "bpsalt"]
SPLITS = ["geo_split", "freq_split", "snr_split", "density_split"]
CLIENTS = ["client1", "client2"]

BASE_IN = "dataset"
BASE_OUT = os.path.join(BASE_IN, "combined")


def stack_and_save(paths, out_path):
    arrays = [np.load(p) for p in paths]
    combined = np.concatenate(arrays, axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, combined)
    print(f"  {out_path}  shape={combined.shape}")


def main():
    # 1. Velocity models
    print("Velocity models:")
    vm_paths = [os.path.join(BASE_IN, f, "velocity_model", f"{f}.npy") for f in FAMILIES]
    stack_and_save(vm_paths, os.path.join(BASE_OUT, "velocity_model", "combined.npy"))

    # 2. Ground-truth seismic
    print("GT seismic:")
    gt_paths = [os.path.join(BASE_IN, f, "seismic_data", "gt", f"{f}.npy") for f in FAMILIES]
    stack_and_save(gt_paths, os.path.join(BASE_OUT, "seismic_data", "gt", "combined.npy"))

    # 3. Client seismic per split
    for split in SPLITS:
        print(f"Split: {split}")
        for client in CLIENTS:
            paths = [
                os.path.join(BASE_IN, f, "seismic_data", split, client, f"{f}.npy")
                for f in FAMILIES
            ]
            out = os.path.join(BASE_OUT, "seismic_data", split, client, "combined.npy")
            stack_and_save(paths, out)

    print("\nDone. Combined dataset written to", BASE_OUT)


if __name__ == "__main__":
    main()
