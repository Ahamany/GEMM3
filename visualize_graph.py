#!/usr/bin/env python3
"""Interactive 3D visualisation of PyG graphs produced by voxel_to_graph.py.

Renders nodes coloured by a selectable feature channel, with toggleable
vector arrows (normals or magnetisation) and edge overlay.

Usage
-----
    python3 visualize_graph.py graphs/sim_0001.pt
    python3 visualize_graph.py graphs/sim_0010.pt --color-by mx --vectors mag
    python3 visualize_graph.py graphs/sim_0008.pt --cone --max-cones 3000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import pyvista as pv

# ── Feature channel mapping ─────────────────────────────────────────────────

CHANNEL_MAP = {
    "mx": 0, "my": 1, "mz": 2,
    "nx": 3, "ny": 4, "nz": 5,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _subsample_indices(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a sorted random subset of indices [0, n) of size max_n."""
    idx = rng.choice(n, size=max_n, replace=False)
    idx.sort()
    return idx


def _mean_edge_length(pos: np.ndarray, edge_index: np.ndarray) -> float:
    """Average Euclidean edge length (on a small sample for speed)."""
    n_edges = edge_index.shape[1]
    if n_edges == 0:
        return 1.0
    sample = min(n_edges, 2000)
    idx = np.random.default_rng(0).choice(n_edges, size=sample, replace=False)
    src = edge_index[0, idx]
    dst = edge_index[1, idx]
    lengths = np.linalg.norm(pos[src] - pos[dst], axis=1)
    return float(lengths.mean())


def _try_load_shape_type(pt_path: Path) -> str:
    """Try to find the matching geometry_info.json and return the shape type."""
    sim_name = pt_path.stem
    for parent in [pt_path.parent.parent, pt_path.parent.parent / "dataset"]:
        json_path = parent / sim_name / "geometry_info.json"
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)["shape_type"]
    # also try dataset/ relative to cwd
    json_path = Path("dataset") / sim_name / "geometry_info.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)["shape_type"]
    return "unknown"


# ── Mesh builders ───────────────────────────────────────────────────────────

def make_node_mesh(pos_nm: np.ndarray, scalars: np.ndarray,
                   scalar_name: str) -> pv.PolyData:
    """Point cloud mesh with a named scalar array."""
    mesh = pv.PolyData(pos_nm)
    mesh[scalar_name] = scalars
    return mesh


def make_vector_line_mesh(pos_nm: np.ndarray, vecs: np.ndarray,
                          arrow_len: float, scalars: np.ndarray,
                          scalar_name: str) -> pv.PolyData:
    """Line-segment arrows from each node to pos + arrow_len * vec."""
    tips = pos_nm + arrow_len * vecs
    n = pos_nm.shape[0]
    all_points = np.vstack([pos_nm, tips])
    cells = np.column_stack([
        np.full(n, 2, dtype=np.int64),
        np.arange(n, dtype=np.int64),
        np.arange(n, 2 * n, dtype=np.int64),
    ]).ravel()
    mesh = pv.PolyData(all_points, lines=cells)
    mesh[scalar_name] = np.concatenate([scalars, scalars])
    return mesh


def make_vector_glyph_mesh(pos_nm: np.ndarray, vecs: np.ndarray,
                           arrow_len: float, scalars: np.ndarray,
                           scalar_name: str, max_cones: int,
                           rng: np.random.Generator) -> pv.PolyData:
    """Arrow glyphs oriented by vector field (subsampled)."""
    n = pos_nm.shape[0]
    if n > max_cones:
        idx = _subsample_indices(n, max_cones, rng)
        pos_nm = pos_nm[idx]
        vecs = vecs[idx]
        scalars = scalars[idx]

    mesh = pv.PolyData(pos_nm)
    mesh["vectors"] = vecs
    mesh[scalar_name] = scalars
    glyphs = mesh.glyph(
        orient="vectors", scale=False, factor=arrow_len, geom=pv.Arrow(),
    )
    return glyphs


def make_edge_mesh(pos_nm: np.ndarray, edge_index: np.ndarray,
                   max_edges: int, rng: np.random.Generator) -> pv.PolyData:
    """Line mesh for graph edges (subsampled)."""
    n_edges = edge_index.shape[1]
    if n_edges > max_edges:
        idx = _subsample_indices(n_edges, max_edges, rng)
        edge_index = edge_index[:, idx]
        n_edges = max_edges

    src, dst = edge_index[0], edge_index[1]
    cells = np.column_stack([
        np.full(n_edges, 2, dtype=np.int64), src, dst,
    ]).ravel()
    mesh = pv.PolyData(pos_nm, lines=cells)
    return mesh


# ── Main ─────────────────────────────────────────────────────────────────────

def visualize(args):
    pt_path = Path(args.graph)
    graph = torch.load(pt_path, weights_only=False)

    pos = graph.pos.numpy().copy()
    features = graph.x.numpy().copy()
    edge_index = graph.edge_index.numpy().copy()

    rng = np.random.default_rng(42)

    # optional node subsampling
    n_nodes = pos.shape[0]
    if args.max_nodes > 0 and n_nodes > args.max_nodes:
        node_idx = _subsample_indices(n_nodes, args.max_nodes, rng)
        pos = pos[node_idx]
        features = features[node_idx]
        idx_set = set(node_idx.tolist())
        old_to_new = {old: new for new, old in enumerate(node_idx)}
        mask = np.array([
            edge_index[0, i] in idx_set and edge_index[1, i] in idx_set
            for i in range(edge_index.shape[1])
        ])
        edge_index = edge_index[:, mask]
        edge_index = np.vectorize(old_to_new.get)(edge_index)

    # convert to nanometres for display
    pos_nm = pos / 1e-9

    # colour channel
    ch = CHANNEL_MAP[args.color_by]
    scalars = features[:, ch]

    # arrow length: fraction of mean edge length
    mel = _mean_edge_length(pos_nm, edge_index if edge_index.shape[1] > 0
                            else np.zeros((2, 0), dtype=int))
    arrow_len = mel * args.vector_scale

    # select vector field
    if args.vectors == "normals":
        vecs = features[:, 3:6]
        vec_label = "Normals"
    elif args.vectors == "mag":
        vecs = features[:, 0:3]
        vec_label = "Magnetisation"
    else:
        vecs = None
        vec_label = None

    # ── build title ──
    shape_type = _try_load_shape_type(pt_path)
    n_total_edges = graph.edge_index.shape[1]
    title = (
        f"{pt_path.name}  |  {shape_type}  |  "
        f"{graph.num_nodes} nodes, {n_total_edges} edges  |  "
        f"color: {args.color_by}"
    )

    # ── PyVista plotter ──
    plotter = pv.Plotter(theme=pv.themes.DarkTheme(), window_size=(1400, 900))
    plotter.add_title(title, font_size=10)

    # nodes
    node_mesh = make_node_mesh(pos_nm, scalars, args.color_by)
    plotter.add_mesh(
        node_mesh,
        scalars=args.color_by,
        cmap=args.colorscale,
        clim=[-1, 1],
        point_size=5,
        render_points_as_spheres=True,
        show_scalar_bar=True,
        scalar_bar_args=dict(title=args.color_by),
    )

    # vectors
    if vecs is not None:
        if args.cone:
            glyph_mesh = make_vector_glyph_mesh(
                pos_nm, vecs, arrow_len, scalars,
                args.color_by, args.max_cones, rng,
            )
            plotter.add_mesh(
                glyph_mesh, scalars=args.color_by,
                cmap=args.colorscale, clim=[-1, 1],
                show_scalar_bar=False,
            )
        else:
            line_mesh = make_vector_line_mesh(
                pos_nm, vecs, arrow_len, scalars, args.color_by,
            )
            plotter.add_mesh(
                line_mesh, scalars=args.color_by,
                cmap=args.colorscale, clim=[-1, 1],
                line_width=2, show_scalar_bar=False,
            )

    # edges (hidden by default, checkbox to toggle)
    if args.max_edges != 0 and edge_index.shape[1] > 0:
        edge_mesh = make_edge_mesh(pos_nm, edge_index, args.max_edges, rng)
        edges_actor = plotter.add_mesh(
            edge_mesh, color="grey", line_width=1, show_scalar_bar=False,
        )
        edges_actor.visibility = False

        def toggle_edges(flag):
            edges_actor.visibility = flag

        plotter.add_checkbox_button_widget(
            toggle_edges, value=False, position=(10, 10),
            size=25, color_on="grey",
        )

    # axes
    plotter.show_bounds(
        xtitle="x (nm)", ytitle="y (nm)", ztitle="z (nm)", font_size=8,
    )

    plotter.show()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D visualisation of PyG graphs",
    )
    parser.add_argument(
        "graph", type=str,
        help="Path to a .pt graph file produced by voxel_to_graph.py",
    )
    parser.add_argument(
        "--vectors", choices=["normals", "mag", "none"], default="normals",
        help="Which vectors to render (default: normals)",
    )
    parser.add_argument(
        "--color-by", choices=list(CHANNEL_MAP.keys()), default="mz",
        help="Feature channel for node colouring (default: mz)",
    )
    parser.add_argument(
        "--colorscale", type=str, default="RdBu",
        help="Matplotlib colormap name (default: RdBu)",
    )
    parser.add_argument(
        "--cone", action="store_true",
        help="Use cone glyphs instead of line segments for vectors",
    )
    parser.add_argument(
        "--vector-scale", type=float, default=1.0,
        help="Arrow length multiplier relative to mean edge length (default: 1.0)",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=0,
        help="Max nodes to render, 0 = all (default: 0)",
    )
    parser.add_argument(
        "--max-edges", type=int, default=2000,
        help="Max edges to render, 0 = skip (default: 2000)",
    )
    parser.add_argument(
        "--max-cones", type=int, default=5000,
        help="Max cone glyphs when --cone is used (default: 5000)",
    )

    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
