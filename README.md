## Project: Machine Learning-Based Generation of 3D and Curved Magnetic Nanostructures
The project is dedicated to developing a high-precision machine learning pipeline for the rapid potential modeling and inverse design of topologically complex magnetic nanostructures.

### Challenges of Traditional Methods
The governing dynamics of magnetic nanostructures are described by the Landau-Lifshitz-Gilbert (LLG) equation.
* Traditional methods solve this equation iteratively over time, which requires enormous computational resources for large 3D systems.
* Standard solvers, such as OOMMF and Mumax3, use finite difference methods (FDM).
* On curved surfaces (such as chiral nanowires), the application of FDM creates a "staircase effect," leading to physical artifacts and an exponential slowdown in computations.

### Scientific Novelty and Theoretical Foundation
As a solution, the use of an E(3)-equivariant graph neural network is proposed.
* **Geometric Deep Learning:** Instead of voxel grids, data is represented as graphs. This allows for natural processing of complex 3D shapes (such as chiral nanowires), completely avoiding the "staircase effect."
* **E(3)-Equivariance:** This critical property ensures that when a physical structure is rotated, the predicted vector field rotates with it, preserving physical validity.
* This approach bypasses the limitations of grid-based solvers and offers a new paradigm for "Inverse Design."

### Methodology and Model Architecture
The project involves the use of Mumax3 (Go/CUDA) and Python tools, as well as the PyTorch Geometric and e3nn frameworks. The pipeline consists of the following stages:

1. **Data Generation and Preparation (Data Prep):**
   * A synthetic dataset is created using massively parallel simulations in Mumax3 (5,000 to 10,000 samples) using edge smoothing.
   * The input voxel grid (.ovf files) is converted into surface graphs via geometric projection (mapping voxel centers onto analytical surfaces).
   * This step corrects the FDM discretization artifacts.
2. **EGNN (Equivariant Graph Neural Network):**
   * Development and training of an E(3)-equivariant message passing neural network (MPNN) to predict magnetic states.
   * The core of the model is based on physical principles: message passing layers simulate the exchange interaction between nodes.
3. **Prediction:**
   * The output is a predicted vector field.
   * A hard constraint is applied: spin length normalization.
4. **Inverse Design:**
   * Implementation of a gradient-based inverse design module for topology optimization.
   * This module enables the search for non-trivial nanostructure shapes that possess specific magnetic properties.

### Development Plans
The project is divided into several phases:
* **Phase 1:** Creating the data generation pipeline with artifact correction.
* **Phase 2:** Training and optimizing the EGNN, as well as verifying accuracy against Mumax3.
* **Phase 3:** Implementing the gradient-based inverse design module for topology optimization.
* **Future:** Experimental validation of the AI-generated magnetic nanostructures.

---

## Scripts

### `generate_dataset.py`

Generates a synthetic dataset of magnetic nanostructure simulation inputs for EGNN training. The script produces two geometry types selected randomly with a **2:1 ratio** (double helix : twisted wire):

**1. Chiral Nanowire** (`chiral_nanowire`) — Two intertwined helical strands (180° apart) built from CSG `Ellipsoid` spheres. Randomised parameters: helix radius (0.6-1.0 x strand_r), strand radius (5-15 nm), pitch (20-60 nm), length (80-200 nm), chirality (+/-1). Supports **chirality inversion defects** — structural points along Z where the twist direction reverses:
   - 20% ideal (no inversions), 64% single inversion, 16% double inversion.
   - Inversions cannot occur within the first/last 15% of the wire length.
   - For double inversions, the two points must be at least 20% of the length apart.
   - Phase continuity is maintained at each inversion point so the helix has no geometric breaks.

**2. Twisted Nanowire** (`twisted_nanowire`) — A solid rod with an elliptical cross-section that rotates along Z, built from CSG `Cylinder` slices. Randomised parameters: major diameter (40-80 nm), minor diameter (15-30 nm), length (100-300 nm), number of turns (1-4), chirality (+/-1).

For each sample, the script produces:

- **`sim.mx3`** — A complete Mumax3 simulation script with FFT-friendly grid dimensions, edge smoothing (`EdgeSmooth = 8`), permalloy-like material parameters (Msat = 8e5 A/m, Aex = 1.3e-11 J/m), random initial magnetisation, and a `Relax()` energy minimisation step.
- **`geometry_info.json`** — Shape parameters, grid metadata, and analytical surface equations. Chiral nanowires use a `piecewise_helix` format with explicit segments (chirality, phase offset per segment). Twisted nanowires describe the cross-section ellipse and twist parameters. This file is consumed downstream by `voxel_to_graph.py` for geometric projection.

The script also writes a `tasks.txt` file listing all simulation directories, intended for SLURM job array integration.

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `-n`, `--num-sims` | `100` | Number of simulations to generate |
| `-o`, `--output-dir` | `dataset` | Output directory |
| `-s`, `--seed` | `42` | Random seed for reproducibility |

---

### `voxel_to_graph.py`

Converts Mumax3 simulation output (OVF2 binary voxel data) into PyTorch Geometric graph objects. This is the core artifact-correction step of the pipeline. The script:

1. **Parses OVF2 binary files** — Reads the header for grid dimensions and cell sizes, decodes the float32 payload, and computes origin-centred voxel coordinates. Empty cells (vacuum) are discarded.
2. **Projects voxel centres onto analytical surfaces** — Uses the shape equations from `geometry_info.json` to map discretised voxel positions onto the ideal continuous surface, eliminating the FDM "staircase effect." Two projectors are available:
   - **ChiralNanowireProjector** — Handles `piecewise_helix` strands with chirality inversions. Assigns each voxel to its nearest strand via dense centreline sampling, then optimises parametric tube coordinates `(t, v)` with L-BFGS. Uses `torch.where` for differentiable piecewise segment selection (chirality and phase offset per segment). Frenet frame (T, N, B) computed in closed form.
   - **TwistedNanowireProjector** — Projects voxels onto the lateral surface of a twisted elliptical rod. Parametric surface `S(z, phi)` rotates an ellipse boundary by the twist angle `theta(z)`. Optimises `(z, phi)` with L-BFGS.
3. **Computes surface normals** — Autograd cross-product of parametric partials (dS/dp0 x dS/dp1), shared by both projectors via `BaseProjector._normals_from_autograd`.
4. **Assembles PyG `Data` objects** — Each graph contains node positions (`pos`), a feature matrix `x` of shape [N, 6] (magnetisation `[mx, my, mz]` + surface normals `[nx, ny, nz]`), and an `edge_index` built via radius graph.

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-dir` | `dataset` | Root dataset directory containing `sim_XXXX/` folders |
| `--output-dir` | `graphs` | Output directory for `.pt` graph files |
| `--radius` | `4e-9` | Radius cutoff (metres) for graph edge construction |

---

### `visualize_graph.py`

Interactive 3D visualisation tool for inspecting the PyG graph files produced by `voxel_to_graph.py`. Uses PyVista (VTK-based desktop renderer) for high-performance 3D rendering:

- **Nodes** coloured by a selectable feature channel (`mx`, `my`, `mz`, `nx`, `ny`, `nz`), rendered as spheres with a diverging colorbar.
- **Vector arrows** (line segments or 3D arrow glyphs) showing magnetisation or surface normal directions.
- **Graph edges** as grey lines, hidden by default with a checkbox widget to toggle visibility.

Supports subsampling for large graphs and automatic detection of the shape type from the corresponding `geometry_info.json`.

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `graph` (positional) | — | Path to a `.pt` graph file |
| `--vectors` | `normals` | Vector field to render: `normals`, `mag`, or `none` |
| `--color-by` | `mz` | Feature channel for node colouring |
| `--colorscale` | `RdBu` | Matplotlib colormap name |
| `--cone` | off | Use 3D arrow glyphs instead of line segments |
| `--vector-scale` | `1.0` | Arrow length multiplier relative to mean edge length |
| `--max-nodes` | `0` (all) | Max nodes to render (0 = no limit) |
| `--max-edges` | `2000` | Max edges to render (0 = skip edges) |
| `--max-cones` | `5000` | Max arrow glyphs when `--cone` is used |

---

## Pipeline Usage

The three scripts form a sequential pipeline. Each step's output is the next step's input.

### Step 1: Generate simulation inputs

```bash
python generate_dataset.py --num-sims 500 --output-dir dataset --seed 42
```

This creates:
```
dataset/
  sim_0001/
    sim.mx3
    geometry_info.json
  sim_0002/
    ...
tasks.txt
```

### Step 2: Run Mumax3 simulations

Process the generated `.mx3` files with Mumax3. Each simulation produces an output directory `sim.out/` containing `.ovf` files with the relaxed magnetisation state.

**Single simulation:**
```bash
mumax3 dataset/sim_0001/sim.mx3
```

**Batch processing with SLURM (recommended for large datasets):**

The generated `tasks.txt` file lists all simulation directories and is designed for use with SLURM job arrays:
```bash
#!/bin/bash
#SBATCH --array=1-500
#SBATCH --gres=gpu:1

SIM_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" tasks.txt)
mumax3 "${SIM_DIR}/sim.mx3"
```

After Mumax3 finishes, the directory structure becomes:
```
dataset/
  sim_0001/
    sim.mx3
    geometry_info.json
    sim.out/
      m000000.ovf      <-- relaxed magnetisation
      ...
```

### Step 3: Convert voxel data to graphs

```bash
python voxel_to_graph.py --dataset-dir dataset --output-dir graphs --radius 4e-9
```

This reads each `sim_XXXX/` directory that contains both `geometry_info.json` and `sim.out/`, applies geometric projection to correct FDM artifacts, and saves one `.pt` file per simulation:
```
graphs/
  sim_0001.pt
  sim_0002.pt
  ...
```

Each `.pt` file is a PyTorch Geometric `Data` object ready for EGNN training.

### Step 4: Visualise and inspect results

```bash
# Basic visualisation (nodes coloured by mz, normal vectors shown)
python visualize_graph.py graphs/sim_0001.pt

# Colour by x-component of magnetisation, show magnetisation vectors as cones
python visualize_graph.py graphs/sim_0010.pt --color-by mx --vectors mag --cone

# Inspect graph connectivity
python visualize_graph.py graphs/sim_0008.pt --max-edges 10000

# Disable vectors entirely
python visualize_graph.py graphs/sim_0042.pt --vectors none
```

---

## Requirements

- **Python 3.10+**
- **Mumax3** (Go/CUDA) — for running the `.mx3` simulations
- Python packages:
  - `numpy`
  - `scipy`
  - `torch`
  - `torch_geometric` (+ `torch-cluster`)
  - `pyvista`
