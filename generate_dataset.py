#!/usr/bin/env python3
"""Generate training dataset of magnetic nanostructure configurations for EGNN.

Creates sim_XXXX folders, each containing:
  - sim.mx3       : Mumax3 simulation script
  - geometry_info.json : shape parameters + analytical surface equations

Also writes tasks.txt for SLURM Job Array integration.

Using
-------------
    python3 generate_dataset.py                        # 100 samples → ./dataset
    python3 generate_dataset.py --num_samples 50 \\
                                --output_dir test_dataset
"""

import os
import json
import math
import random
import argparse

# ---------------------------------------------------------------------------
# Physical constants (fixed for all simulations)
# ---------------------------------------------------------------------------
CELL = 2e-9    # cell size  [m]
PAD = 16e-9    # padding    [m] (8 cells per side) 

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_fft_friendly(n):
    """True if n factors entirely into primes 2, 3, 5, 7."""
    if n <= 0:
        return False
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1


def next_fft_friendly(n):
    """Smallest integer >= n whose prime factors are only 2, 3, 5, 7."""
    n = max(1, int(math.ceil(n)))
    while not is_fft_friendly(n):
        n += 1
    return n


def grid_dims(obj_x, obj_y, obj_z):
    """FFT-friendly grid dimensions (cells) for object bbox + padding."""
    nx = next_fft_friendly(math.ceil((obj_x + 2 * PAD) / CELL))
    ny = next_fft_friendly(math.ceil((obj_y + 2 * PAD) / CELL))
    nz = next_fft_friendly(math.ceil((obj_z + 2 * PAD) / CELL))
    return nx, ny, nz


def fv(v):
    """Format a float for insertion into an .mx3 script."""
    return f"{v:.10g}"


# ---------------------------------------------------------------------------
# mx3 assembly
# ---------------------------------------------------------------------------
def build_mx3(nx, ny, nz, geom_lines, seed):
    """Return complete .mx3 script as a string."""
    L = []
    L.append(f"SetGridSize({nx}, {ny}, {nz})")
    L.append(f"SetCellSize({fv(CELL)}, {fv(CELL)}, {fv(CELL)})")
    L.append("")
    L.append("// Geometry")
    L.extend(geom_lines)
    L.append("EdgeSmooth = 8")          # Edge Smooth
    L.append("SetGeom(geo)")
    L.append("")
    L.append("// Material parameters")
    L.append("Msat  = 8e5")             # A/m – saturation magnetization
    L.append("Aex   = 1.3e-11")         # J/m – exchange interaction constant
    L.append("alpha = 0.5")             # Hilbert damping
    L.append("")
    L.append("// Random initial magnetization")
    L.append(f"m = RandomMagSeed({seed})")
    L.append("")
    L.append("OutputFormat = OVF2_BINARY")
    L.append("")
    L.append("// Energy minimisation")
    L.append("Relax()")
    L.append("")
    L.append("// Save relaxed state")
    L.append("Save(m)")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Shape generators
# ---------------------------------------------------------------------------
# Each returns (geom_lines, params_dict, analytical_surface_dict, bbox_tuple)


def gen_nanotube(rng):
    R_out = rng.uniform(15e-9, 40e-9)
    wall = rng.uniform(6e-9, min(20e-9, R_out - 4e-9))
    R_in = R_out - wall
    length = rng.uniform(40e-9, 120e-9)

    geom = [
        f"geo := Cylinder({fv(2*R_out)}, {fv(length)})"
        f".Sub(Cylinder({fv(2*R_in)}, {fv(length)}))",
    ]

    params = dict(
        shape_type="nanotube",
        R_outer_m=R_out,
        R_inner_m=R_in,
        length_m=length,
        wall_thickness_m=wall,
    )

    surface = dict(
        implicit_body="R_inner^2 <= x^2 + y^2 <= R_outer^2  AND  |z| <= L/2",
        surfaces=[
            dict(name="outer_cylinder", type="cylinder",
                 equation="x^2 + y^2 - R_outer^2 = 0",
                 axis="z", R=R_out,
                 z_range=[-length / 2, length / 2]),
            dict(name="inner_cylinder", type="cylinder",
                 equation="x^2 + y^2 - R_inner^2 = 0",
                 axis="z", R=R_in,
                 z_range=[-length / 2, length / 2]),
            dict(name="top_cap", type="annular_disk",
                 equation="z - L/2 = 0",
                 z=length / 2, R_inner=R_in, R_outer=R_out),
            dict(name="bottom_cap", type="annular_disk",
                 equation="z + L/2 = 0",
                 z=-length / 2, R_inner=R_in, R_outer=R_out),
        ],
    )

    bbox = (2 * R_out, 2 * R_out, length)
    return geom, params, surface, bbox


def gen_torus(rng):
    R = rng.uniform(20e-9, 50e-9)
    r = rng.uniform(8e-9, min(20e-9, R - 4e-9))

    N = max(36, int(math.ceil(2 * math.pi * R / r)))
    d = 2 * r

    lines = []
    for i in range(N):
        a = i * 2 * math.pi / N
        x = R * math.cos(a)
        y = R * math.sin(a)
        sphere = f"Ellipsoid({fv(d)}, {fv(d)}, {fv(d)}).Transl({fv(x)}, {fv(y)}, 0)"
        if i == 0:
            lines.append(f"geo := {sphere}")
        else:
            lines.append(f"geo = geo.Add({sphere})")

    params = dict(
        shape_type="torus",
        R_major_m=R,
        r_minor_m=r,
        N_csg_spheres=N,
    )

    surface = dict(
        implicit_body="(sqrt(x^2 + y^2) - R_major)^2 + z^2 <= r_minor^2",
        surfaces=[
            dict(
                name="torus_surface", type="torus",
                equation="(sqrt(x^2 + y^2) - R_major)^2 + z^2 - r_minor^2 = 0",
                R_major=R, r_minor=r,
                parametric=dict(
                    x="(R_major + r_minor*cos(v))*cos(u)",
                    y="(R_major + r_minor*cos(v))*sin(u)",
                    z="r_minor*sin(v)",
                    u_range=[0, "2*pi"], v_range=[0, "2*pi"],
                ),
            ),
        ],
    )

    bbox = (2 * (R + r), 2 * (R + r), 2 * r)
    return lines, params, surface, bbox


def gen_hollow_sphere(rng):
    R_out = rng.uniform(20e-9, 50e-9)
    wall = rng.uniform(6e-9, min(25e-9, R_out - 4e-9))
    R_in = R_out - wall

    d_out, d_in = 2 * R_out, 2 * R_in
    geom = [
        f"geo := Ellipsoid({fv(d_out)}, {fv(d_out)}, {fv(d_out)})"
        f".Sub(Ellipsoid({fv(d_in)}, {fv(d_in)}, {fv(d_in)}))",
    ]

    params = dict(
        shape_type="hollow_sphere",
        R_outer_m=R_out,
        R_inner_m=R_in,
        wall_thickness_m=wall,
    )

    def _sphere_surf(name, R_val):
        return dict(
            name=name, type="sphere",
            equation=f"x^2 + y^2 + z^2 - R^2 = 0",
            R=R_val,
            parametric=dict(
                x="R*sin(theta)*cos(phi)",
                y="R*sin(theta)*sin(phi)",
                z="R*cos(theta)",
                theta_range=[0, "pi"], phi_range=[0, "2*pi"],
            ),
        )

    surface = dict(
        implicit_body="R_inner^2 <= x^2 + y^2 + z^2 <= R_outer^2",
        surfaces=[
            _sphere_surf("outer_sphere", R_out),
            _sphere_surf("inner_sphere", R_in),
        ],
    )

    bbox = (2 * R_out, 2 * R_out, 2 * R_out)
    return geom, params, surface, bbox


def gen_chiral_nanowire(rng):
    strand_r = rng.uniform(5e-9, 10e-9)
    helix_r = rng.uniform(6e-9, 14e-9)
    length = rng.uniform(80e-9, 180e-9)
    pitch = rng.uniform(30e-9, min(70e-9, length / 1.5))
    chirality = rng.choice([-1, 1])

    core_r = max(CELL, helix_r * 0.3)

    # Sphere spacing along z so that 3-D gap < strand_r (good overlap)
    arc_k = math.sqrt(1 + (2 * math.pi * helix_r / pitch) ** 2)
    dz = strand_r / arc_k
    n_pts = max(2, int(math.ceil(length / dz)) + 1)

    d = 2 * strand_r
    half = length / 2

    lines = [f"geo := Cylinder({fv(2*core_r)}, {fv(length)})"]
    for i in range(n_pts):
        t = i / (n_pts - 1)          # 0 → 1
        z = -half + t * length
        theta = chirality * 2 * math.pi * z / pitch

        for offset in (0.0, math.pi):  # two strands
            x = helix_r * math.cos(theta + offset)
            y = helix_r * math.sin(theta + offset)
            lines.append(
                f"geo = geo.Add(Ellipsoid({fv(d)}, {fv(d)}, {fv(d)})"
                f".Transl({fv(x)}, {fv(y)}, {fv(z)}))"
            )

    chi = "+" if chirality > 0 else "-"
    params = dict(
        shape_type="chiral_nanowire",
        helix_radius_m=helix_r,
        strand_radius_m=strand_r,
        core_radius_m=core_r,
        pitch_m=pitch,
        length_m=length,
        chirality=chirality,
        n_points_per_strand=n_pts,
    )

    surface = dict(
        implicit_body=(
            "min(dist(P,C1), dist(P,C2)) <= r_strand  "
            "OR  (x^2+y^2 <= r_core^2 AND |z| <= L/2)"
        ),
        chirality=chirality,
        centerlines=dict(
            strand_1=dict(
                type="helix",
                parametric=dict(
                    x=f"R_h*cos({chi}2*pi*t/P)",
                    y=f"R_h*sin({chi}2*pi*t/P)",
                    z="t",
                    t_range=[-half, half],
                ),
                R_h=helix_r, P=pitch,
            ),
            strand_2=dict(
                type="helix",
                parametric=dict(
                    x=f"R_h*cos({chi}2*pi*t/P + pi)",
                    y=f"R_h*sin({chi}2*pi*t/P + pi)",
                    z="t",
                    t_range=[-half, half],
                ),
                R_h=helix_r, P=pitch,
            ),
        ),
        core=dict(
            type="cylinder", equation="x^2 + y^2 - r_core^2 = 0",
            R=core_r, z_range=[-half, half],
        ),
        strand_tube=dict(
            description="Tubular surface of radius r_strand around each helix centreline",
            r_strand=strand_r,
        ),
    )

    total_r = helix_r + strand_r
    bbox = (2 * total_r, 2 * total_r, length)
    return lines, params, surface, bbox


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
GENERATORS = [gen_nanotube, gen_torus, gen_hollow_sphere, gen_chiral_nanowire]


def main():
    ap = argparse.ArgumentParser(
        description="Generate magnetic nanostructure dataset for EGNN training",
    )
    ap.add_argument("-n", "--num-sims", type=int, default=100,
                    help="number of simulations (default: 100)")
    ap.add_argument("-o", "--output-dir", type=str, default="dataset",
                    help="output directory (default: dataset)")
    ap.add_argument("-s", "--seed", type=int, default=42,
                    help="random seed (default: 42)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    task_paths = []

    for i in range(args.num_sims):
        sim_name = f"sim_{i + 1:04d}"
        sim_dir = os.path.join(args.output_dir, sim_name)
        os.makedirs(sim_dir, exist_ok=True)

        gen = rng.choice(GENERATORS)
        geom_lines, params, analytical_surface, bbox = gen(rng)
        nx, ny, nz = grid_dims(*bbox)

        # Write sim.mx3
        mx3 = build_mx3(nx, ny, nz, geom_lines, seed=i + 1)
        with open(os.path.join(sim_dir, "sim.mx3"), "w") as f:
            f.write(mx3)

        # Write geometry_info.json
        info = dict(
            shape_type=params["shape_type"],
            parameters=params,
            grid=dict(Nx=nx, Ny=ny, Nz=nz,
                      cell_size_m=CELL, padding_m=PAD),
            analytical_surface=analytical_surface,
        )
        with open(os.path.join(sim_dir, "geometry_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        task_paths.append(sim_dir)
        print(f"[{i+1:4d}/{args.num_sims}] {sim_name}:  "
              f"{params['shape_type']:20s}  grid {nx}x{ny}x{nz}")

    # Write task list for SLURM
    tasks_file = "tasks.txt"
    with open(tasks_file, "w") as f:
        for p in task_paths:
            f.write(p + "\n")

    print(f"\nDone. {args.num_sims} simulations in {args.output_dir}/")
    print(f"SLURM task list: {tasks_file}")


if __name__ == "__main__":
    main()
