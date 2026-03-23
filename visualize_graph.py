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
import plotly.graph_objects as go

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


# ── Trace builders ───────────────────────────────────────────────────────────

def make_node_trace(pos: np.ndarray, colors: np.ndarray, features: np.ndarray,
                    colorscale: str) -> go.Scatter3d:
    """Scatter3d trace for graph nodes."""
    hover = []
    labels = ["mx", "my", "mz", "nx", "ny", "nz"]
    for i in range(pos.shape[0]):
        vals = "  ".join(f"{labels[j]}={features[i, j]:+.3f}" for j in range(6))
        hover.append(f"node {i}<br>{vals}")

    return go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=colors,
            colorscale=colorscale,
            cmin=-1, cmax=1,
            colorbar=dict(title="value", thickness=15),
        ),
        hovertext=hover,
        hoverinfo="text",
        name="Nodes",
    )


def make_vector_lines(pos: np.ndarray, vecs: np.ndarray, colors: np.ndarray,
                      arrow_len: float, colorscale: str,
                      label: str) -> go.Scatter3d:
    """Line-segment arrows from pos to pos + arrow_len * vec."""
    tips = pos + arrow_len * vecs
    n = pos.shape[0]

    # interleave base, tip, None for each segment
    xs = np.empty(n * 3)
    ys = np.empty(n * 3)
    zs = np.empty(n * 3)
    xs[0::3] = pos[:, 0];  xs[1::3] = tips[:, 0];  xs[2::3] = np.nan
    ys[0::3] = pos[:, 1];  ys[1::3] = tips[:, 1];  ys[2::3] = np.nan
    zs[0::3] = pos[:, 2];  zs[1::3] = tips[:, 2];  zs[2::3] = np.nan

    # per-segment colour (repeat base colour for base, tip, gap)
    seg_colors = np.repeat(colors, 3)

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(
            color=seg_colors,
            colorscale=colorscale,
            cmin=-1, cmax=1,
            width=2,
        ),
        hoverinfo="skip",
        name=label,
    )


def make_vector_cones(pos: np.ndarray, vecs: np.ndarray, colors: np.ndarray,
                      arrow_len: float, colorscale: str, label: str,
                      max_cones: int, rng: np.random.Generator) -> go.Cone:
    """Cone glyphs for vector arrows (subsampled)."""
    n = pos.shape[0]
    if n > max_cones:
        idx = _subsample_indices(n, max_cones, rng)
        pos = pos[idx]
        vecs = vecs[idx]
        colors = colors[idx]

    scaled = arrow_len * vecs

    return go.Cone(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        u=scaled[:, 0], v=scaled[:, 1], w=scaled[:, 2],
        colorscale=colorscale,
        cmin=-1, cmax=1,
        sizemode="absolute",
        sizeref=arrow_len * 5,
        showscale=False,
        name=label,
    )


def make_edge_trace(pos: np.ndarray, edge_index: np.ndarray,
                    max_edges: int, rng: np.random.Generator) -> go.Scatter3d:
    """Thin grey lines for graph edges (subsampled)."""
    n_edges = edge_index.shape[1]
    if n_edges > max_edges:
        idx = _subsample_indices(n_edges, max_edges, rng)
        edge_index = edge_index[:, idx]
        n_edges = max_edges

    src = edge_index[0]
    dst = edge_index[1]

    xs = np.empty(n_edges * 3)
    ys = np.empty(n_edges * 3)
    zs = np.empty(n_edges * 3)
    xs[0::3] = pos[src, 0];  xs[1::3] = pos[dst, 0];  xs[2::3] = np.nan
    ys[0::3] = pos[src, 1];  ys[1::3] = pos[dst, 1];  ys[2::3] = np.nan
    zs[0::3] = pos[src, 2];  zs[1::3] = pos[dst, 2];  zs[2::3] = np.nan

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color="grey", width=1),
        hoverinfo="skip",
        name="Edges",
        visible="legendonly",  # hidden by default
    )


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
    node_idx = None
    if args.max_nodes > 0 and n_nodes > args.max_nodes:
        node_idx = _subsample_indices(n_nodes, args.max_nodes, rng)
        pos = pos[node_idx]
        features = features[node_idx]
        # remap edges to subsampled node set
        idx_set = set(node_idx.tolist())
        old_to_new = {old: new for new, old in enumerate(node_idx)}
        mask = np.array([
            edge_index[0, i] in idx_set and edge_index[1, i] in idx_set
            for i in range(edge_index.shape[1])
        ])
        edge_index = edge_index[:, mask]
        edge_index = np.vectorize(old_to_new.get)(edge_index)
        n_nodes = pos.shape[0]

    # convert to nanometres for display
    pos_nm = pos / 1e-9

    # colour channel
    ch = CHANNEL_MAP[args.color_by]
    colors = features[:, ch]

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

    # ── build traces ──
    traces = []
    traces.append(make_node_trace(pos_nm, colors, features, args.colorscale))

    if vecs is not None:
        if args.cone:
            traces.append(make_vector_cones(
                pos_nm, vecs, colors, arrow_len, args.colorscale,
                vec_label, args.max_cones, rng,
            ))
        else:
            traces.append(make_vector_lines(
                pos_nm, vecs, colors, arrow_len, args.colorscale, vec_label,
            ))

    if args.max_edges != 0 and edge_index.shape[1] > 0:
        traces.append(make_edge_trace(pos_nm, edge_index, args.max_edges, rng))

    # ── layout ──
    shape_type = _try_load_shape_type(pt_path)
    n_total_edges = graph.edge_index.shape[1]
    title = (
        f"{pt_path.name}  |  {shape_type}  |  "
        f"{graph.num_nodes} nodes, {n_total_edges} edges  |  "
        f"color: {args.color_by}"
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        title=title,
        scene=dict(
            xaxis_title="x (nm)",
            yaxis_title="y (nm)",
            zaxis_title="z (nm)",
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.show()


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
        help="Plotly colorscale name (default: RdBu)",
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
