#!/usr/bin/env python3
"""Generate training dataset of chiral nanowire configurations for EGNN.

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


def _sample_inversions(rng, length):
    """Sample chirality inversion z-positions.

    Probabilistic distribution:
      20 % → ideal helix (0 inversions)
      64 % → single chirality switch
      16 % → double chirality switch

    Spatial constraints (L = length):
      - No inversion within first/last 15 % of L.
      - For double: |z2 - z1| >= 0.20 * L.

    Returns a sorted list of 0, 1, or 2 absolute z-positions (metres).
    """
    roll = rng.random()
    if roll < 0.20:
        return []

    half = length / 2
    margin = 0.15 * length
    z_lo = -half + margin
    z_hi = half - margin

    if roll < 0.84:                             # 0.20 + 0.64
        return [rng.uniform(z_lo, z_hi)]

    # Double inversion — rejection sampling (acceptance ≈ 71 %)
    min_sep = 0.20 * length
    while True:
        z1 = rng.uniform(z_lo, z_hi)
        z2 = rng.uniform(z_lo, z_hi)
        if abs(z2 - z1) >= min_sep:
            return sorted([z1, z2])


def _build_segments(chirality, pitch, half, inversions):
    """Build piecewise helix segments with accumulated phase offsets.

    At each inversion point z_inv the chirality flips and a phase offset
    is accumulated so that θ(z) = h_k · 2π·z/P + φ_k is continuous:

        φ_next = φ_prev + 2 · h_prev · 2π · z_inv / pitch

    Returns list of (z_start, z_end, h, phase_offset) tuples.
    """
    boundaries = [-half] + list(inversions) + [half]
    segments = []
    h, phi = chirality, 0.0
    for k in range(len(boundaries) - 1):
        segments.append((boundaries[k], boundaries[k + 1], h, phi))
        if k < len(boundaries) - 2:
            z_inv = boundaries[k + 1]
            phi += 2 * h * 2 * math.pi * z_inv / pitch
            h = -h
    return segments


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

def gen_chiral_nanowire(rng):
    strand_r = rng.uniform(5e-9, 15e-9)
    # helix_r <= strand_r guarantees the strands reach the twist axis
    # (no central void) and that the two strands are mutually tangent or
    # overlapping — the defining property of a tight twisted pair.
    helix_r = rng.uniform(0.6 * strand_r, strand_r)
    length = rng.uniform(80e-9, 200e-9)
    pitch = rng.uniform(20e-9, min(60e-9, length / 1.5))
    chirality = rng.choice([-1, 1])

    # Sphere spacing along z so that 3-D arc step < strand_r (solid overlap)
    arc_k = math.sqrt(1 + (2 * math.pi * helix_r / pitch) ** 2)
    dz = strand_r / arc_k
    n_pts = max(2, int(math.ceil(length / dz)) + 1)

    d = 2 * strand_r
    half = length / 2

    # --- chirality inversions ---
    inversions = _sample_inversions(rng, length)
    segments = _build_segments(chirality, pitch, half, inversions)

    lines = []
    first = True
    for i in range(n_pts):
        t = i / (n_pts - 1)          # 0 → 1
        z = -half + t * length

        # Segment-aware angle (piecewise chirality with phase continuity)
        for _zs, z_end, h_seg, phi_seg in segments:
            if z <= z_end:
                theta = h_seg * 2 * math.pi * z / pitch + phi_seg
                break

        for offset in (0.0, math.pi):  # two strands, 180° apart
            x = helix_r * math.cos(theta + offset)
            y = helix_r * math.sin(theta + offset)
            sphere = (f"Ellipsoid({fv(d)}, {fv(d)}, {fv(d)})"
                      f".Transl({fv(x)}, {fv(y)}, {fv(z)})")
            if first:
                lines.append(f"geo := {sphere}")
                first = False
            else:
                lines.append(f"geo = geo.Add({sphere})")

    params = dict(
        shape_type="chiral_nanowire",
        helix_radius_m=helix_r,
        strand_radius_m=strand_r,
        pitch_m=pitch,
        length_m=length,
        chirality=chirality,
        n_inversions=len(inversions),
        inversion_z_m=inversions,
        n_points_per_strand=n_pts,
    )

    def _strand_dict(strand_phase):
        return dict(
            type="piecewise_helix",
            R_h=helix_r,
            P=pitch,
            strand_phase_rad=strand_phase,
            segments=[
                dict(z_start=zs, z_end=ze,
                     chirality=h, phase_offset_rad=phi)
                for zs, ze, h, phi in segments
            ],
        )

    surface = dict(
        implicit_body="min(dist(P,C1), dist(P,C2)) <= r_strand",
        initial_chirality=chirality,
        inversions_z_m=inversions,
        centerlines=dict(
            strand_1=_strand_dict(0.0),
            strand_2=_strand_dict(math.pi),
        ),
        strand_tube=dict(
            description="Tubular surface of radius r_strand around each helix centreline",
            r_strand=strand_r,
        ),
    )

    total_r = helix_r + strand_r
    bbox = (2 * total_r, 2 * total_r, length)
    return lines, params, surface, bbox


def gen_twisted_nanowire(rng):
    """Solid rod with elliptical cross-section that twists along Z."""
    D_major = rng.uniform(40e-9, 80e-9)
    D_minor = rng.uniform(15e-9, 30e-9)
    L = rng.uniform(100e-9, 300e-9)
    n_turns = rng.uniform(1.0, 4.0)
    chirality = rng.choice([-1, 1])

    # Slice thickness = cell size, adjusted for exact fit
    n_slices = max(2, math.ceil(L / CELL))
    dz = L / n_slices

    half = L / 2
    scale_y = D_minor / D_major

    lines = []
    first = True
    for i in range(n_slices):
        z_center = -half + i * dz + dz / 2
        theta = chirality * 2 * math.pi * n_turns * (z_center + half) / L

        sl = (f"Cylinder({fv(D_major)}, {fv(dz)})"
              f".Scale(1, {fv(scale_y)}, 1)"
              f".RotZ({fv(theta)})"
              f".Transl(0, 0, {fv(z_center)})")
        if first:
            lines.append(f"geo := {sl}")
            first = False
        else:
            lines.append(f"geo = geo.Add({sl})")

    params = dict(
        shape_type="twisted_nanowire",
        D_major_m=D_major,
        D_minor_m=D_minor,
        length_m=L,
        n_turns=n_turns,
        chirality=chirality,
        n_inversions=0,
        n_slices=n_slices,
        slice_thickness_m=dz,
    )

    surface = dict(
        implicit_body="rotate(x,y,-theta(z)) -> (x',y'); x'^2/a^2 + y'^2/b^2 <= 1",
        shape_type="twisted_nanowire",
        cross_section=dict(
            type="ellipse",
            semi_major_m=D_major / 2,
            semi_minor_m=D_minor / 2,
        ),
        twist=dict(
            chirality=chirality,
            n_turns=n_turns,
            z_range=[-half, half],
        ),
    )

    bbox = (D_major, D_major, L)
    return lines, params, surface, bbox


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
GENERATORS = [
    (gen_chiral_nanowire, 2),    # 2/3
    (gen_twisted_nanowire, 1),   # 1/3
]


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

        gens, weights = zip(*GENERATORS)
        gen = rng.choices(gens, weights=weights, k=1)[0]
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
        n_inv = params["n_inversions"]
        defect = f"{n_inv}inv" if n_inv > 0 else "ideal"
        print(f"[{i+1:4d}/{args.num_sims}] {sim_name}:  "
              f"{params['shape_type']:20s}  {defect:6s}  grid {nx}x{ny}x{nz}")

    # Write task list for SLURM
    tasks_file = "tasks.txt"
    with open(tasks_file, "w") as f:
        for p in task_paths:
            f.write(p + "\n")

    print(f"\nDone. {args.num_sims} simulations in {args.output_dir}/")
    print(f"SLURM task list: {tasks_file}")


if __name__ == "__main__":
    main()
