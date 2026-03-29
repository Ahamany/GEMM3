#!/usr/bin/env python3
"""Convert Mumax3 voxel data to PyTorch Geometric graphs with geometric projection.

Pipeline:
  1. Parse OVF2 binary → voxel positions + magnetization vectors
  2. Project voxel centers onto ideal analytical surfaces (gradient-based)
  3. Compute surface normals via autograd
  4. Assemble PyTorch Geometric Data objects

Projector hierarchy:
  BaseProjector (ABC)
  ├── ChiralNanowireProjector — piecewise helix tubes  (L-BFGS + Frenet frame)
  └── TwistedNanowireProjector — twisted elliptical rod (L-BFGS)
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

    @staticmethod
    def _length_scale(P: torch.Tensor) -> float:
        """Characteristic length for normalisation (keeps loss O(1))."""
        return float(P.abs().max().clamp(min=1e-15))

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
            fallback_norm = fallback_dir.norm(dim=-1, keepdim=True).clamp(
                min=1e-12,
            )
            normals[singular] = fallback_dir / fallback_norm

        return S.detach(), normals.detach()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2a: ChiralNanowireProjector (piecewise helix tubes)
# ═══════════════════════════════════════════════════════════════════════════════

class ChiralNanowireProjector(BaseProjector):
    """Gradient-based projection for chiral nanowire surfaces.

    Supports ``piecewise_helix`` strands with chirality inversions.
    Each strand carries a ``segments`` list and a ``strand_phase_rad``.

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
    def _build_helix_tube_fn(strand_info: dict, r_strand: float,
                             scale: float = 1.0):
        """Build a torch function for the piecewise helix tube surface.

        Reads ``strand_info["segments"]`` (each with chirality and
        phase_offset_rad) and ``strand_info["strand_phase_rad"]``.
        Uses ``torch.where`` for differentiable segment selection.
        """
        R_h = strand_info["R_h"] / scale
        P_pitch = strand_info["P"] / scale
        r_s = r_strand / scale
        strand_phase = strand_info["strand_phase_rad"]
        segments = strand_info["segments"]

        # precompute per-segment data (scaled)
        seg_data = []
        for seg in segments:
            seg_data.append((
                seg["z_start"] / scale,
                seg["z_end"] / scale,
                seg["chirality"] * 2.0 * torch.pi / P_pitch,
                seg["phase_offset_rad"],
            ))

        t_lo = seg_data[0][0]
        t_hi = seg_data[-1][1]

        def tube_fn(t, v):
            """S(t, v) = C(t) + r * (N(t)*cos(v) + B(t)*sin(v))"""
            # piecewise omega and phase via torch.where
            omega = torch.zeros_like(t)
            phi = torch.zeros_like(t)
            for z_lo, z_hi, om, ph in seg_data:
                mask = (t >= z_lo) & (t <= z_hi)
                omega = torch.where(mask, torch.full_like(omega, om), omega)
                phi = torch.where(mask, torch.full_like(phi, ph), phi)

            angle = omega * t + phi + strand_phase

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
        r_strand = info["strand_tube"]["r_strand"]

        P = P_voxel.double()
        N = P.shape[0]
        scale = self._length_scale(P)

        strand_keys = sorted(centerlines.keys())

        # --- assign each voxel to nearest strand centreline ---
        strand_cl_dists = []
        for sk in strand_keys:
            s = centerlines[sk]
            R_h = s["R_h"]
            P_pitch = s["P"]
            strand_phase = s["strand_phase_rad"]
            segments = s["segments"]

            # dense sampling along the piecewise helix centreline
            all_pts = []
            for seg in segments:
                tlo = seg["z_start"]
                thi = seg["z_end"]
                chi = seg["chirality"]
                phi_off = seg["phase_offset_rad"]
                omega = chi * 2.0 * torch.pi / P_pitch

                n_seg = max(50, int((thi - tlo) / (r_strand * 0.5)))
                t_seg = torch.linspace(tlo, thi, n_seg, dtype=torch.float64)
                angles = omega * t_seg + phi_off + strand_phase
                hx = R_h * torch.cos(angles)
                hy = R_h * torch.sin(angles)
                hz = t_seg
                all_pts.append(torch.stack([hx, hy, hz], dim=-1))

            helix_pts = torch.cat(all_pts, dim=0)
            diff = P.unsqueeze(1) - helix_pts.unsqueeze(0)
            min_dist_cl = diff.norm(dim=-1).min(dim=1).values
            strand_cl_dists.append(min_dist_cl)

        # assign each voxel to the nearest strand
        cl_stack = torch.stack(strand_cl_dists, dim=-1)  # [N, n_strands]
        labels = cl_stack.argmin(dim=-1)                  # [N]

        P_ideal = torch.zeros_like(P)
        normals = torch.zeros_like(P)

        # --- strands: parametric optimisation (scaled coordinates) ---
        for i, sk in enumerate(strand_keys):
            m_strand = labels == i
            if not m_strand.any():
                continue

            P_s = P[m_strand]
            P_s_norm = P_s / scale

            tube_fn_s, tlo_s, thi_s = self._build_helix_tube_fn(
                centerlines[sk], r_strand, scale=scale,
            )

            # initial guess: t ≈ z/scale
            t0 = P_s_norm[:, 2].clamp(tlo_s, thi_s)

            # piecewise omega/phase for the initial angle at t0
            s_info = centerlines[sk]
            R_h_s = s_info["R_h"] / scale
            P_pitch_s = s_info["P"] / scale
            sp = s_info["strand_phase_rad"]

            omega0 = torch.zeros_like(t0)
            phase0 = torch.zeros_like(t0)
            for seg in s_info["segments"]:
                z_lo_seg = seg["z_start"] / scale
                z_hi_seg = seg["z_end"] / scale
                chi = seg["chirality"]
                phi = seg["phase_offset_rad"]
                mask = (t0 >= z_lo_seg) & (t0 <= z_hi_seg)
                om_val = chi * 2.0 * torch.pi / P_pitch_s
                omega0 = torch.where(
                    mask, torch.full_like(omega0, om_val), omega0,
                )
                phase0 = torch.where(
                    mask, torch.full_like(phase0, phi), phase0,
                )

            angle0 = omega0 * t0 + phase0 + sp
            cx0 = R_h_s * torch.cos(angle0)
            cy0 = R_h_s * torch.sin(angle0)
            dx = P_s_norm[:, 0] - cx0
            dy = P_s_norm[:, 1] - cy0
            dz = P_s_norm[:, 2] - t0

            # project onto N and B directions to get v0
            nx0 = -torch.cos(angle0)
            ny0 = -torch.sin(angle0)
            proj_N = dx * nx0 + dy * ny0

            tx0 = -R_h_s * omega0 * torch.sin(angle0)
            ty0 = R_h_s * omega0 * torch.cos(angle0)
            tz0 = torch.ones_like(t0)
            tn = torch.sqrt(tx0 ** 2 + ty0 ** 2 + tz0 ** 2)
            tx0, ty0, tz0 = tx0 / tn, ty0 / tn, tz0 / tn
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

    def project(self, P_voxel: torch.Tensor):
        return self._project_chiral(P_voxel)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2b: TwistedNanowireProjector (twisted elliptical rod)
# ═══════════════════════════════════════════════════════════════════════════════

class TwistedNanowireProjector(BaseProjector):
    """Gradient-based projection for twisted nanowire surfaces.

    The geometry is a solid rod whose elliptical cross-section rotates
    along Z.  Parametric surface S(z, φ):

        θ(z)  = chirality · 2π · n_turns · (z − z_lo) / L
        x     = a·cos(φ)·cos(θ) − b·sin(φ)·sin(θ)
        y     = a·cos(φ)·sin(θ) + b·sin(φ)·cos(θ)
        z_s   = z

    Optimiser: L-BFGS (vectorised over all N points simultaneously).
    Normals:   dS/dz × dS/dφ via ``torch.autograd``.
    """

    def __init__(self, analytical_surface: dict,
                 lr: float = 1.0, num_steps: int = 60, max_iter: int = 20):
        self.lr = lr
        self.num_steps = num_steps
        self.max_iter = max_iter
        self.analytical_surface = analytical_surface

    @staticmethod
    def _build_twisted_surface_fn(surface_info: dict, scale: float = 1.0):
        """Build a torch function for the twisted-ellipse lateral surface."""
        cs = surface_info["cross_section"]
        tw = surface_info["twist"]
        a = cs["semi_major_m"] / scale
        b = cs["semi_minor_m"] / scale
        chirality = tw["chirality"]
        n_turns = tw["n_turns"]
        z_lo = tw["z_range"][0] / scale
        z_hi = tw["z_range"][1] / scale
        L = z_hi - z_lo

        def surface_fn(z_s, phi):
            theta = chirality * 2.0 * torch.pi * n_turns * (z_s - z_lo) / L

            # ellipse boundary in local cross-section frame
            ex = a * torch.cos(phi)
            ey = b * torch.sin(phi)

            # rotate by theta into lab frame
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            sx = ex * cos_t - ey * sin_t
            sy = ex * sin_t + ey * cos_t
            sz = z_s

            return torch.stack([sx, sy, sz], dim=-1)

        return surface_fn, z_lo, z_hi

    def project(self, P_voxel: torch.Tensor):
        info = self.analytical_surface
        P = P_voxel.double()
        scale = self._length_scale(P)

        cs = info["cross_section"]
        tw = info["twist"]
        a_s = cs["semi_major_m"] / scale
        b_s = cs["semi_minor_m"] / scale
        chirality = tw["chirality"]
        n_turns = tw["n_turns"]
        z_range = tw["z_range"]
        z_lo_s = z_range[0] / scale
        z_hi_s = z_range[1] / scale
        L_s = z_hi_s - z_lo_s

        surf_fn_s, _, _ = self._build_twisted_surface_fn(info, scale=scale)

        P_norm = P / scale
        x, y, z = P_norm[:, 0], P_norm[:, 1], P_norm[:, 2]

        # initial guess: z_s0 = z, phi0 from rotated coordinates
        z0 = z.clamp(z_lo_s, z_hi_s)
        theta0 = chirality * 2.0 * torch.pi * n_turns * (z0 - z_lo_s) / L_s
        cos_t0 = torch.cos(theta0)
        sin_t0 = torch.sin(theta0)
        x_loc = x * cos_t0 + y * sin_t0
        y_loc = -x * sin_t0 + y * cos_t0
        phi0 = torch.atan2(y_loc / b_s, x_loc / a_s)

        params_z = z0.clone().requires_grad_(True)
        params_phi = phi0.clone().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [params_z, params_phi], lr=self.lr, max_iter=self.max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure(pz=params_z, pp=params_phi, target=P_norm,
                    fn=surf_fn_s):
            optimizer.zero_grad()
            S = fn(pz, pp)
            loss = ((S - target) ** 2).sum()
            loss.backward()
            return loss

        for _ in range(self.num_steps):
            optimizer.step(closure)

        # normals at unscaled coordinates
        surf_fn_orig, _, _ = self._build_twisted_surface_fn(info, scale=1.0)
        z_orig = (params_z * scale).detach()
        phi_orig = params_phi.detach()

        P_ideal, normals = self._normals_from_autograd(
            surf_fn_orig, z_orig, phi_orig, P,
        )

        return P_ideal.float(), normals.float()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Factory
# ═══════════════════════════════════════════════════════════════════════════════

def projector_factory(geometry_info: dict) -> BaseProjector:
    """Instantiate the correct projector from a geometry_info.json dict."""
    shape = geometry_info["shape_type"]
    surface = geometry_info["analytical_surface"]

    if shape == "chiral_nanowire":
        return ChiralNanowireProjector(analytical_surface=surface)
    elif shape == "twisted_nanowire":
        return TwistedNanowireProjector(analytical_surface=surface)
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
