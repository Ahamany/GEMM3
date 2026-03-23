#!/usr/bin/env python3
"""Convert Mumax3 voxel data to PyTorch Geometric graphs with geometric projection.

Pipeline:
  1. Parse OVF2 binary → voxel positions + magnetization vectors
  2. Project voxel centers onto ideal analytical surfaces (gradient-based)
  3. Compute surface normals via autograd
  4. Assemble PyTorch Geometric Data objects

Projector hierarchy:
  BaseProjector (ABC)
  └── ChiralNanowireProjector — gradient-based (L-BFGS + Frenet frame)
"""

import json
import struct
import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: OVF Parser
# ═══════════════════════════════════════════════════════════════════════════════

def parse_ovf(filepath: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse an OVF2 binary file and return filtered voxel data.

    Reads the header for grid dimensions and cell sizes, then decodes the
    binary float32 payload.  Voxel centres are computed with index-based
    centering so they match the analytical geometry definitions exactly:

        coord[i] = (i - N/2 + 0.5) * cell_size

    Empty cells (|m| == 0) are discarded.

    Returns
    -------
    P_voxel : Tensor [N, 3]
        Origin-centred voxel coordinates (metres).
    m : Tensor [N, 3]
        Magnetisation unit vectors.
    """
    header = {}
    binary_offset = None

    with open(filepath, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in {filepath}")
            text = line.decode("latin-1").strip()

            if text.startswith("# Begin: Data Binary 4"):
                binary_offset = f.tell()
                break

            if text.startswith("#"):
                parts = text[2:].split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    val = parts[1].strip()
                    header[key] = val

        # --- read binary payload ---
        tag_bytes = f.read(4)
        tag = struct.unpack("<f", tag_bytes)[0]
        if abs(tag - 1234567.0) > 1.0:
            raise ValueError(
                f"OVF2 validation tag mismatch: expected 1234567.0, got {tag}"
            )

        nx = int(header["xnodes"])
        ny = int(header["ynodes"])
        nz = int(header["znodes"])
        total = nx * ny * nz

        raw = np.frombuffer(f.read(total * 3 * 4), dtype="<f4")
        if raw.size != total * 3:
            raise ValueError(
                f"Data size mismatch: expected {total*3} floats, got {raw.size}"
            )

    # reshape: OVF2 order is z-slowest, y-medium, x-fastest
    data = raw.reshape(nz, ny, nx, 3)

    dx = float(header["xstepsize"])
    dy = float(header["ystepsize"])
    dz = float(header["zstepsize"])

    # index-based centering (matches Mumax3 geometry origin)
    ix = np.arange(nx, dtype=np.float64)
    iy = np.arange(ny, dtype=np.float64)
    iz = np.arange(nz, dtype=np.float64)

    x_coords = (ix - nx / 2.0 + 0.5) * dx
    y_coords = (iy - ny / 2.0 + 0.5) * dy
    z_coords = (iz - nz / 2.0 + 0.5) * dz

    # meshgrid: shapes (nz, ny, nx) each
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
    positions = np.stack([xx, yy, zz], axis=-1)  # (nz, ny, nx, 3)

    # flatten
    positions = positions.reshape(-1, 3)
    mag = data.reshape(-1, 3).astype(np.float64)

    # filter empty cells
    mag_norm = np.linalg.norm(mag, axis=1)
    mask = mag_norm > 0.0

    P_voxel = torch.from_numpy(positions[mask]).float()
    m = torch.from_numpy(mag[mask]).float()

    # re-normalise magnetisation to unit vectors
    m_norm = m.norm(dim=1, keepdim=True).clamp(min=1e-12)
    m = m / m_norm

    return P_voxel, m


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Projector base class
# ═══════════════════════════════════════════════════════════════════════════════

class BaseProjector(ABC):
    """Abstract base for all geometric projectors."""

    @abstractmethod
    def project(
        self, P_voxel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project voxel centres onto the ideal analytical surface.

        Parameters
        ----------
        P_voxel : Tensor [N, 3]
            Origin-centred voxel coordinates.

        Returns
        -------
        P_ideal : Tensor [N, 3]
            Coordinates on the ideal surface.
        normals : Tensor [N, 3]
            Outward-pointing unit normals at the projected points.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2a: ChiralNanowireProjector (gradient-based)
# ═══════════════════════════════════════════════════════════════════════════════

class ChiralNanowireProjector(BaseProjector):
    """Gradient-based projection for chiral nanowire surfaces.

    Helix-tube surface around helix centrelines built directly in torch
    using the closed-form Frenet frame, plus analytical projection for
    the core cylinder.

    Optimiser: L-BFGS (vectorised over all N points simultaneously).
    Normals:   dS/dp0 × dS/dp1 via ``torch.autograd``.
    """

    def __init__(self, analytical_surface: dict,
                 lr: float = 1.0, num_steps: int = 60, max_iter: int = 20):
        self.lr = lr
        self.num_steps = num_steps
        self.max_iter = max_iter
        self.analytical_surface = analytical_surface

    @staticmethod
    def _length_scale(P: torch.Tensor) -> float:
        """Characteristic length for normalisation (keeps loss O(1))."""
        return float(P.abs().max().clamp(min=1e-15))

    @staticmethod
    def _build_helix_tube_fn(strand_info: dict, r_strand: float,
                             scale: float = 1.0):
        """Build a torch function for the tube surface around a helix.

        Uses the closed-form Frenet frame of a circular helix.
        When *scale* ≠ 1 the returned function produces coordinates in
        normalised (O(1)) space: all lengths are divided by *scale*, while
        angles are unaffected.
        """
        R_h = strand_info["R_h"] / scale
        P_pitch = strand_info["P"] / scale
        r_s = r_strand / scale
        t_range = strand_info["parametric"]["t_range"]
        t_lo = float(t_range[0]) / scale
        t_hi = float(t_range[1]) / scale

        # detect chirality from the parametric expression sign
        expr_x = strand_info["parametric"]["x"]
        chirality = -1.0 if "-2*pi" in expr_x else 1.0

        # detect phase offset (strand_2 has +pi)
        phase = 0.0
        if "+ pi" in expr_x or "+pi" in expr_x:
            phase = torch.pi

        omega = chirality * 2.0 * torch.pi / P_pitch

        def tube_fn(t, v):
            """S(t, v) = C(t) + r * (N(t)*cos(v) + B(t)*sin(v))"""
            angle = omega * t + phase

            # helix centreline C(t)
            cx = R_h * torch.cos(angle)
            cy = R_h * torch.sin(angle)
            cz = t

            # principal normal N(t) — always points toward helix axis
            nx = -torch.cos(angle)
            ny = -torch.sin(angle)
            nz = torch.zeros_like(t)

            # tangent (unnormalised)
            tx = -R_h * omega * torch.sin(angle)
            ty = R_h * omega * torch.cos(angle)
            tz = torch.ones_like(t)
            t_norm = torch.sqrt(tx ** 2 + ty ** 2 + tz ** 2)
            tx, ty, tz = tx / t_norm, ty / t_norm, tz / t_norm

            # binormal B = T × N
            bx = ty * nz - tz * ny
            by = tz * nx - tx * nz
            bz = tx * ny - ty * nx

            # tube surface
            cos_v = torch.cos(v)
            sin_v = torch.sin(v)
            sx = cx + r_s * (nx * cos_v + bx * sin_v)
            sy = cy + r_s * (ny * cos_v + by * sin_v)
            sz = cz + r_s * (nz * cos_v + bz * sin_v)

            return torch.stack([sx, sy, sz], dim=-1)

        return tube_fn, t_lo, t_hi

    def _project_chiral(self, P_voxel: torch.Tensor):
        info = self.analytical_surface
        centerlines = info["centerlines"]
        core = info["core"]
        r_strand = info["strand_tube"]["r_strand"]
        R_core = core["R"]
        z_lo, z_hi = core["z_range"]

        P = P_voxel.double()
        N = P.shape[0]
        scale = self._length_scale(P)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]

        strand_keys = sorted(centerlines.keys())

        # --- assign each voxel to nearest component ---
        # Use containment test: if point is inside a strand tube, assign
        # to that strand.  Only fall back to core for points not inside
        # any strand.  This avoids the pathology where points deep inside
        # a strand are closer to the core *surface* than to the tube wall.

        rho = torch.sqrt(x ** 2 + y ** 2)

        # compute distance to each strand centreline (dense helix sampling)
        strand_cl_dists = []  # distance to centreline per strand
        for sk in strand_keys:
            s = centerlines[sk]
            R_h = s["R_h"]
            P_pitch = s["P"]
            tlo = float(s["parametric"]["t_range"][0])
            thi = float(s["parametric"]["t_range"][1])
            expr_x = s["parametric"]["x"]
            chirality = -1.0 if "-2*pi" in expr_x else 1.0
            phase = torch.pi if ("+ pi" in expr_x or "+pi" in expr_x) else 0.0
            omega = chirality * 2.0 * torch.pi / P_pitch

            n_samples = max(200, int((thi - tlo) / (r_strand * 0.5)))
            t_samples = torch.linspace(tlo, thi, n_samples, dtype=torch.float64)
            angles = omega * t_samples + phase
            hx = R_h * torch.cos(angles)
            hy = R_h * torch.sin(angles)
            hz = t_samples

            helix_pts = torch.stack([hx, hy, hz], dim=-1)  # [S, 3]
            diff = P.unsqueeze(1) - helix_pts.unsqueeze(0)
            dists_to_cl = diff.norm(dim=-1)
            min_dist_cl = dists_to_cl.min(dim=1).values  # [N]
            strand_cl_dists.append(min_dist_cl)

        # containment: point is inside strand if dist_to_centreline < r_strand
        inside_strand = [d < r_strand for d in strand_cl_dists]
        inside_any_strand = torch.zeros(N, dtype=torch.bool)
        for mask in inside_strand:
            inside_any_strand |= mask

        # for points inside multiple strands, pick the nearest centreline
        labels = torch.zeros(N, dtype=torch.long)  # 0 = core (default)
        if inside_any_strand.any():
            cl_stack = torch.stack(strand_cl_dists, dim=-1)  # [N, n_strands]
            nearest_strand = cl_stack.argmin(dim=-1)  # [N]
            labels[inside_any_strand] = nearest_strand[inside_any_strand] + 1

        # points not inside any strand and not inside core: nearest surface
        inside_core = (rho < R_core) & (z.abs() < abs(z_hi))
        unassigned = ~inside_any_strand & ~inside_core
        if unassigned.any():
            d_core_surf = torch.abs(rho[unassigned] - R_core)
            strand_surf = torch.stack(
                [torch.abs(d - r_strand) for d in strand_cl_dists], dim=-1,
            )[unassigned]
            all_d = torch.cat([d_core_surf.unsqueeze(-1), strand_surf], dim=-1)
            labels[unassigned] = all_d.argmin(dim=-1)

        P_ideal = torch.zeros_like(P)
        normals = torch.zeros_like(P)

        # --- core: analytical cylinder projection ---
        m_core = labels == 0
        if m_core.any():
            Pc = P[m_core]
            xc, yc, zc = Pc[:, 0], Pc[:, 1], Pc[:, 2]
            rho_c = torch.sqrt(xc ** 2 + yc ** 2).clamp(min=1e-15)
            cos_phi = xc / rho_c
            sin_phi = yc / rho_c
            z_clamp = zc.clamp(z_lo, z_hi)
            P_ideal[m_core] = torch.stack(
                [R_core * cos_phi, R_core * sin_phi, z_clamp], dim=-1,
            )
            normals[m_core] = torch.stack(
                [cos_phi, sin_phi, torch.zeros_like(cos_phi)], dim=-1,
            )

        # --- strands: parametric optimisation (scaled coordinates) ---
        for i, sk in enumerate(strand_keys):
            m_strand = labels == (i + 1)
            if not m_strand.any():
                continue

            P_s = P[m_strand]
            P_s_norm = P_s / scale

            # scaled tube function for optimisation
            tube_fn_s, tlo_s, thi_s = self._build_helix_tube_fn(
                centerlines[sk], r_strand, scale=scale,
            )

            # initial guess: t ≈ z/scale, v from atan2 of offset from
            # centreline to get a good angular starting point
            t0 = P_s_norm[:, 2].clamp(tlo_s, thi_s)

            # compute initial v from the vector (P - C(t0))
            s_info = centerlines[sk]
            R_h_s = s_info["R_h"] / scale
            P_pitch_s = s_info["P"] / scale
            expr_x = s_info["parametric"]["x"]
            chi = -1.0 if "-2*pi" in expr_x else 1.0
            ph = torch.pi if ("+ pi" in expr_x or "+pi" in expr_x) else 0.0
            om = chi * 2.0 * torch.pi / P_pitch_s
            angle0 = om * t0 + ph
            cx0 = R_h_s * torch.cos(angle0)
            cy0 = R_h_s * torch.sin(angle0)
            dx = P_s_norm[:, 0] - cx0
            dy = P_s_norm[:, 1] - cy0
            dz = P_s_norm[:, 2] - t0
            # project onto N and B directions to get v0
            nx0 = -torch.cos(angle0)
            ny0 = -torch.sin(angle0)
            proj_N = dx * nx0 + dy * ny0  # component along N
            # B direction (simplified for initial guess)
            tx0 = -R_h_s * om * torch.sin(angle0)
            ty0 = R_h_s * om * torch.cos(angle0)
            tz0 = torch.ones_like(t0)
            tn = torch.sqrt(tx0**2 + ty0**2 + tz0**2)
            tx0, ty0, tz0 = tx0/tn, ty0/tn, tz0/tn
            bx0 = ty0 * 0 - tz0 * ny0
            by0 = tz0 * nx0 - tx0 * 0
            bz0 = tx0 * ny0 - ty0 * nx0
            proj_B = dx * bx0 + dy * by0 + dz * bz0
            v0 = torch.atan2(proj_B, proj_N)

            params_t = t0.clone().requires_grad_(True)
            params_v = v0.clone().requires_grad_(True)

            optimizer = torch.optim.LBFGS(
                [params_t, params_v], lr=self.lr, max_iter=self.max_iter,
                line_search_fn="strong_wolfe",
            )

            def closure(pt=params_t, pv=params_v, target=P_s_norm,
                        fn=tube_fn_s):
                optimizer.zero_grad()
                S = fn(pt, pv)
                loss = ((S - target) ** 2).sum()
                loss.backward()
                return loss

            for _ in range(self.num_steps):
                optimizer.step(closure)

            # normals: use unscaled tube function at the optimised params
            # (rescale t back to metres)
            tube_fn_orig, _, _ = self._build_helix_tube_fn(
                centerlines[sk], r_strand, scale=1.0,
            )
            params_t_orig = (params_t * scale).detach()
            params_v_orig = params_v.detach()

            Pi, Ni = self._normals_from_autograd(
                tube_fn_orig, params_t_orig, params_v_orig, P_s,
            )
            P_ideal[m_strand] = Pi
            normals[m_strand] = Ni

        return P_ideal.float(), normals.float()

    # ------------------------------------------------------------------
    # Autograd normals (shared)
    # ------------------------------------------------------------------
    @staticmethod
    def _normals_from_autograd(surf_fn, p0, p1, P_target):
        """Compute ideal positions and normals via cross product of partials.

        Uses the optimised parameters to evaluate S(p0, p1) and compute
        dS/dp0 × dS/dp1 with ``torch.autograd``.
        """
        u = p0.detach().requires_grad_(True)
        v = p1.detach().requires_grad_(True)

        S = surf_fn(u, v)  # [N, 3]
        Sx, Sy, Sz = S[:, 0], S[:, 1], S[:, 2]

        # 3 backward passes → 6 partial derivatives
        gx = torch.autograd.grad(Sx.sum(), [u, v], retain_graph=True)
        gy = torch.autograd.grad(Sy.sum(), [u, v], retain_graph=True)
        gz = torch.autograd.grad(Sz.sum(), [u, v], retain_graph=True)

        dS_du = torch.stack([gx[0], gy[0], gz[0]], dim=-1)  # [N, 3]
        dS_dv = torch.stack([gx[1], gy[1], gz[1]], dim=-1)  # [N, 3]

        cross = torch.linalg.cross(dS_du, dS_dv)
        cross_norm = cross.norm(dim=-1, keepdim=True)

        # handle singularities: fall back to (P_ideal - origin) direction
        singular = cross_norm.squeeze(-1) < 1e-12
        normals = cross / cross_norm.clamp(min=1e-12)

        if singular.any():
            fallback_dir = S[singular]
            fallback_norm = fallback_dir.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            normals[singular] = fallback_dir / fallback_norm

        return S.detach(), normals.detach()

    def project(self, P_voxel: torch.Tensor):
        return self._project_chiral(P_voxel)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Factory
# ═══════════════════════════════════════════════════════════════════════════════

def projector_factory(geometry_info: dict) -> BaseProjector:
    """Instantiate the correct projector from a geometry_info.json dict."""
    shape = geometry_info["shape_type"]
    surface = geometry_info["analytical_surface"]

    if shape == "chiral_nanowire":
        return ChiralNanowireProjector(analytical_surface=surface)
    else:
        raise ValueError(f"Unknown shape type: {shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Graph assembly
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph(
    P_ideal: torch.Tensor,
    m: torch.Tensor,
    normals: torch.Tensor,
    radius_cutoff: float = 4.0e-9,
) -> Data:
    """Assemble a PyTorch Geometric ``Data`` object.

    Parameters
    ----------
    P_ideal : Tensor [N, 3]   — corrected coordinates
    m       : Tensor [N, 3]   — magnetisation unit vectors
    normals : Tensor [N, 3]   — surface normals
    radius_cutoff : float     — edge cutoff radius (metres)

    Returns
    -------
    Data with ``pos``, ``x`` ([N, 6]), ``edge_index``.
    """
    edge_index = radius_graph(P_ideal, r=radius_cutoff, loop=False)

    return Data(
        pos=P_ideal,
        x=torch.cat([m, normals], dim=1),
        edge_index=edge_index,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def find_ovf(sim_dir: Path) -> Path:
    """Find the (first) .ovf file inside ``sim_dir/sim.out/``."""
    out_dir = sim_dir / "sim.out"
    ovf_files = sorted(out_dir.glob("*.ovf"))
    if not ovf_files:
        raise FileNotFoundError(f"No .ovf files in {out_dir}")
    return ovf_files[0]


def process_simulation(sim_dir: Path, radius_cutoff: float) -> Data:
    """Full pipeline for a single simulation directory."""
    ovf_path = find_ovf(sim_dir)
    json_path = sim_dir / "geometry_info.json"

    with open(json_path) as f:
        geometry_info = json.load(f)

    P_voxel, m = parse_ovf(ovf_path)
    projector = projector_factory(geometry_info)
    P_ideal, normals = projector.project(P_voxel)
    graph = build_graph(P_ideal, m, normals, radius_cutoff)

    return graph


def main():
    parser = argparse.ArgumentParser(
        description="Convert Mumax3 voxel data to PyG graphs",
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="dataset",
        help="Root dataset directory (default: dataset)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="graphs",
        help="Output directory for .pt files (default: graphs)",
    )
    parser.add_argument(
        "--radius", type=float, default=4.0e-9,
        help="Radius cutoff for edge construction in metres (default: 4e-9)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_dirs = sorted([
        d for d in dataset_dir.iterdir()
        if d.is_dir()
        and (d / "geometry_info.json").exists()
        and (d / "sim.out").exists()
    ])

    if not sim_dirs:
        print(f"No valid simulation directories found in {dataset_dir}")
        return

    print(f"Found {len(sim_dirs)} simulations in {dataset_dir}/\n")

    for sim_dir in sim_dirs:
        name = sim_dir.name
        with open(sim_dir / "geometry_info.json") as f:
            shape_type = json.load(f)["shape_type"]

        print(f"  {name}  ({shape_type:20s}) ... ", end="", flush=True)

        try:
            graph = process_simulation(sim_dir, args.radius)
            out_path = output_dir / f"{name}.pt"
            torch.save(graph, out_path)
            print(
                f"nodes={graph.num_nodes:5d}  "
                f"edges={graph.num_edges:6d}  "
                f"saved → {out_path}"
            )
        except Exception as e:
            print(f"FAILED: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
