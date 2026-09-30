from unicodedata import name
import yaml

import h5py
from numpy import rint
import numpy as np

# Path to your local HDF5 file
file_path = "./datasets/ns_incom_inhom_2d_512-0.h5"

with h5py.File(file_path, "r") as f:
    # Print all top-level keys/groups
    print("Available keys:", list(f.keys()))

    # Inspect each top-level dataset
    for key in f.keys():
        item = f[key]
        if isinstance(item, h5py.Dataset):
            print(f"Dataset: '{key}' | Shape: {item.shape} | Type: {item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"Group: '{key}' (contains nested items)")


    print(dict(f.attrs))

    def inspect(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name, "shape:", obj.shape, "dtype:", obj.dtype)
        else:
            print(name, "(group)")

        if len(obj.attrs) > 0:
            print("  Attributes:", dict(obj.attrs))

    

    f.visititems(inspect)
    print("VISIT ITEMS")
    f.visit(print)
    print("VISIT ITEMS")
    print(dict(f.attrs))

    print("ATTRIBUTES")

    t = f["t"][()]

    print("Time array shape:", t.shape)
    print("First time values:", t.reshape(-1)[:10])

    print("File attributes:", dict(f.attrs))
    print("Velocity attributes:", dict(f["velocity"].attrs))
    print("Time attributes:", dict(f["t"].attrs))

    config = yaml.safe_load(f.attrs["config"])

    Lx, Ly = config["domain_size"]
    nx, ny = config["grid_size"]

    dx = Lx/nx
    dy = Ly/ny

    print(dx , dy)