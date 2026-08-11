import struct
from pathlib import Path

import numpy as np


def write_grid(path, sim_id, nx, ny, dx, dy, xmin, xmax, ymin, ymax):
    with open(path, "w") as f:
        f.write(f"{sim_id}\n")
        f.write("grid\n")
        f.write(f"{nx} {ny}\n")
        f.write(f"{dx} {dy}\n")
        f.write(f"{xmin} {xmax}\n")
        f.write(f"{ymin} {ymax}\n")


def read_grid(path):
    with open(path, "r") as f:
        lines = f.read().strip().split("\n")
    return {
        "sim_id": lines[0],
        "nx": int(lines[2].split()[0]),
        "ny": int(lines[2].split()[1]),
        "dx": float(lines[3].split()[0]),
        "dy": float(lines[3].split()[1]),
        "xmin": float(lines[4].split()[0]),
        "xmax": float(lines[4].split()[1]),
        "ymin": float(lines[5].split()[0]),
        "ymax": float(lines[5].split()[1]),
    }


def write_field(path, sim_id, field_type, values, nx, ny, dx, dy, xmin, ymin):
    with open(path, "w") as f:
        f.write(f"{sim_id}\n")
        f.write(f"{field_type}\n")
        for j in range(ny):
            for i in range(nx):
                x = xmin + i * dx
                y = ymin + j * dy
                idx = j * nx + i
                f.write(f"{x} {y} {values[idx]}\n")


def read_field(path):
    with open(path, "r") as f:
        lines = f.read().strip().split("\n")
    sim_id = lines[0]
    field_type = lines[1]
    values = []
    for line in lines[2:]:
        parts = line.split()
        values.append(float(parts[2]))
    return {
        "sim_id": sim_id,
        "field_type": field_type,
        "values": values,
    }


def read_field_compact(path):
    """
    Read a CSF1 compact binary field file (.bin).

    CSF1 header: magic(4) | sim_id_len(8) | sim_id(N) | field_type_len(8) |
                  field_type(M) | nx(8) | ny(8) | dtype(8) | data

    dtype 1 = Float64 scalar → returns (nx, ny) float64 array
    dtype 2 = Float64 vector → returns (nx, ny, 2) float64 array (vx,vy interleaved)
    dtype 3 = Float32 scalar → returns (nx, ny) float32 array
    dtype 4 = Float32 vector → returns (nx, ny, 2) float32 array (vx,vy interleaved)

    All arrays use Julia-compatible column-major (Fortran) order.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"CSF1":
            raise ValueError(f"Not a CSF1 compact file: {path}")

        sim_id_len = struct.unpack("q", f.read(8))[0]
        f.read(sim_id_len)

        field_type_len = struct.unpack("q", f.read(8))[0]
        f.read(field_type_len)

        nx = struct.unpack("q", f.read(8))[0]
        ny = struct.unpack("q", f.read(8))[0]
        dtype = struct.unpack("q", f.read(8))[0]

        if dtype == 1:  # Float64 scalar
            data = np.fromfile(f, dtype=np.float64, count=nx * ny)
            return data.reshape((nx, ny), order="F")
        elif dtype == 2:  # Float64 vector (vx,vy interleaved per point)
            data = np.fromfile(f, dtype=np.float64, count=nx * ny * 2)
            pairs = data.reshape((-1, 2))
            vx = pairs[:, 0].reshape((nx, ny), order="F")
            vy = pairs[:, 1].reshape((nx, ny), order="F")
            return np.stack([vx, vy], axis=-1)
        elif dtype == 3:  # Float32 scalar
            data = np.fromfile(f, dtype=np.float32, count=nx * ny)
            return data.reshape((nx, ny), order="F")
        elif dtype == 4:  # Float32 vector (vx,vy interleaved per point)
            data = np.fromfile(f, dtype=np.float32, count=nx * ny * 2)
            pairs = data.reshape((-1, 2))
            vx = pairs[:, 0].reshape((nx, ny), order="F")
            vy = pairs[:, 1].reshape((nx, ny), order="F")
            return np.stack([vx, vy], axis=-1)
        else:
            raise ValueError(f"Unsupported dtype {dtype} in {path}")
