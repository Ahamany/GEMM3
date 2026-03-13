## Project: Machine Learning-Based Generation of 3D and Curved Magnetic Nanostructures
The project is dedicated to developing a high-precision machine learning pipeline for the rapid potential modeling and inverse design of topologically complex magnetic nanostructures.

### Challenges of Traditional Methods
The governing dynamics of magnetic nanostructures are described by the Landau-Lifshitz-Gilbert (LLG) equation.
* Traditional methods solve this equation iteratively over time, which requires enormous computational resources for large 3D systems.
* Standard solvers, such as OOMMF and Mumax3, use finite difference methods (FDM).
* On curved surfaces, the application of FDM creates a "staircase effect," leading to physical artifacts and an exponential slowdown in computations.

### Scientific Novelty and Theoretical Foundation
As a solution, the use of an E(3)-equivariant graph neural network is proposed.
* **Geometric Deep Learning:** Instead of voxel grids, data is represented as graphs. This allows for natural processing of any 3D shapes (spheres, tori, tubes), completely avoiding the "staircase effect."
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

Generates a synthetic dataset of magnetic nanostructure simulation inputs for EGNN training. For each sample, the script randomly selects one of four shape types — **nanotube**, **torus**, **hollow sphere**, or **chiral nanowire** — with randomised geometric parameters. It then produces:

- **`sim.mx3`** — A complete Mumax3 simulation script with FFT-friendly grid dimensions, edge smoothing (`EdgeSmooth = 8`), permalloy-like material parameters (Msat = 8e5 A/m, Aex = 1.3e-11 J/m), random initial magnetisation, and a `Relax()` energy minimisation step.
- **`geometry_info.json`** — Shape parameters, grid metadata, and analytical surface equations (implicit form + parametric form where applicable). This file is consumed downstream by `voxel_to_graph.py` for geometric projection.

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
2. **Projects voxel centres onto analytical surfaces** — Uses the shape equations from `geometry_info.json` to map discretised voxel positions onto the ideal continuous surface, eliminating the FDM "staircase effect." Four projector types are implemented:
   - **NanotubeProjector** — Closed-form projection onto 4 surfaces (outer/inner cylinder walls + top/bottom annular caps).
   - **HollowSphereProjector** — Closed-form radial projection onto outer/inner concentric spheres.
   - **ParametricOptimizerProjector** — Gradient-based (L-BFGS) projection for torus and chiral nanowire shapes, using parametric surface equations and `torch.autograd` for surface normal computation.
3. **Computes surface normals** — Analytical normals for simple shapes; autograd cross-product of parametric partials (dS/dp0 x dS/dp1) for complex shapes.
4. **Assembles PyG `Data` objects** — Each graph contains node positions (`pos`), a feature matrix `x` of shape [N, 6] (magnetisation `[mx, my, mz]` + surface normals `[nx, ny, nz]`), and an `edge_index` built via radius graph.

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-dir` | `dataset` | Root dataset directory containing `sim_XXXX/` folders |
| `--output-dir` | `graphs` | Output directory for `.pt` graph files |
| `--radius` | `4e-9` | Radius cutoff (metres) for graph edge construction |

---

### `visualize_graph.py`

Interactive 3D visualisation tool for inspecting the PyG graph files produced by `voxel_to_graph.py`. Uses Plotly to render:

- **Nodes** coloured by a selectable feature channel (`mx`, `my`, `mz`, `nx`, `ny`, `nz`).
- **Vector arrows** (line segments or cone glyphs) showing magnetisation or surface normal directions.
- **Graph edges** (toggleable via legend, hidden by default).

Supports subsampling for large graphs and automatic detection of the shape type from the corresponding `geometry_info.json`.

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `graph` (positional) | — | Path to a `.pt` graph file |
| `--vectors` | `normals` | Vector field to render: `normals`, `mag`, or `none` |
| `--color-by` | `mz` | Feature channel for node colouring |
| `--colorscale` | `RdBu` | Plotly colorscale name |
| `--cone` | off | Use cone glyphs instead of line segments |
| `--vector-scale` | `1.0` | Arrow length multiplier relative to mean edge length |
| `--max-nodes` | `0` (all) | Max nodes to render (0 = no limit) |
| `--max-edges` | `5000` | Max edges to render (0 = skip edges) |
| `--max-cones` | `2000` | Max cone glyphs when `--cone` is used |

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
  - `torch`
  - `torch_geometric`
  - `sympy`
  - `plotly`
