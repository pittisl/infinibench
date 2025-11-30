# InfiniBench Agentic & Trajectory Enhancements

This document captures the additions layered on top of stock InfiniBench: agentic constraint generation, cluster-aware solvers, and both frontier and notebook-style camera trajectory optimizers. Start with Step 0 to install the codebase, then jump to the feature you care about.

---

## Step 0 – Install Infinigen (Linux x86_64, Python Module)

The workflow below mirrors the “Installing Infinigen as a Python Module” section of `Installation.md`, trimmed to the Linux x86_64 path that powers InfiniBench.

```bash
# System dependencies (Ubuntu / Debian / WSL / other Linux x86_64 distros)
sudo apt-get install wget cmake g++ libgles2-mesa-dev libglew-dev libglfw3-dev libglm-dev zlib1g-dev

# Clone Infinigen and create a Conda env
git clone https://github.com/princeton-vl/infinigen.git
cd infinigen
conda create --name infinigen python=3.11
conda activate infinigen

# Minimal install (good for InfiniBench + Infinigen-Indoors)
INFINIGEN_MINIMAL_INSTALL=True pip install -e .

# Or enable terrain + OpenGL GT if you need full-scene generation
pip install -e ".[terrain,vis]"
```

Additional variants:
- `pip install -e ".[sim]"` – bring in simulation assets (water, fire, etc.).
- `pip install -e ".[dev,terrain,vis]" && pre-commit install` – add linting/CI helpers.

:warning: If any command fails, rerun it with `-vv > logs.txt 2>&1` appended and attach `logs.txt` when filing an issue.

Once the environment is ready, proceed with the feature sections below.

---

## 1. Cluster-Based Layout Optimization

**Key files**
- `infinigen/core/constraints/example_solver/clusters.py`
- `infinigen/core/constraints/example_solver/moves/cluster.py`
- `infinigen/core/constraints/example_solver/propose_clusters.py`
- `infinigen/core/constraints/example_solver/solve.py`

**What changed**
- Furniture supported by a common parent (e.g., chairs around a table) is auto-grouped using `StableAgainst` relations.
- New moves (`cluster_translate`, `cluster_rotate`, `cluster_resample`) treat each cluster as a rigid body when exploring layouts.
- Collision tests first evaluate a cluster-level AABB to avoid expensive per-object checks when an entire move is invalid.

**How to use**
- Cluster moves are enabled by default during `Solver.solve_objects`.
- To constrain the search space, add a gin override (example):
  ```gin
  Solver.restrict_moves = ["addition", "cluster_translate", "cluster_rotate"]
  ```
- Logging continues to flow through `infinigen.core.constraints.example_solver.solve`, so existing tooling still works.

---

## 2. Agentic Constraint Generation

**Key files**
- `infinigen_examples/constraints/agentic_framework.py`
- `infinigen_examples/generate_indoors.py`

**Highlights**
- `AgenticConstraintGenerator` stitches together prompt templates, API docs, and in-context examples (default: `home_furniture_constraints`).
- `AgenticSceneGenerator` loops over {generate → compile → validate → optional feedback} to enforce chain-of-thought refinement.
- `compose_indoors()` accepts new CLI flags:
  - `scene_description`: natural-language description (“cozy studio with plants”).
  - `use_agentic_constraints`: toggle the agent on/off.
  - `agentic_max_iterations`: bound retries when compilation fails or the optimizer requests changes.

**Example**
```bash
python infinigen_examples/generate_indoors.py \
    --scene_description "compact studio apartment with plants and wall art" \
    --use_agentic_constraints True \
    --agentic_max_iterations 3 \
    -p solve_steps_large=400
```

Behind the scenes, the agent produces Python, compiles it via `agentic_result.final_program.to_callable(...)`, and injects the resulting constraint builder into the standard greedy + simulated annealing loop.

> **Note:** The scaffold ships with `DummyLLM`, which echoes the reference program. Point the generator at a real LLM client to obtain novel constraints.

---

## 3. Camera Trajectory Optimization

**File**: `infinigen_examples/trajectory_optimizer.py`

The module now exposes two complementary pipelines. Choose the one that matches your workflow.

### 3.1 Frontier Optimizer (JSON Output)

- Implements the four-step frontier loop:
  1. Pick the closest unvisited target object.
  2. Sample viewpoints (accessibility, FoV coverage, occlusion) around it.
  3. Run Dijkstra on a 2D navigation grid (constant camera height).
  4. Append translation + rotation poses to the trajectory.
- Outputs a JSON list of `{position, rotation_euler}` entries ready for downstream consumers.

**CLI**
```bash
blender --background --python infinigen_examples/trajectory_optimizer.py -- \
    --blend /path/to/scene.blend \
    --output /tmp/trajectory.json \
    --samples 1500 \
    --grid 0.6
```

### 3.2 Notebook-Derived Batch Pipeline

The former `traj_opt_batch.ipynb` has been ported to pure Python (`camera_traj` + `run_batch_pipeline`). It now:
- Applies automatic decimation to heavy `spawn_asset` meshes.
- Converts Blender meshes → `trimesh` → 2D polygons for navigation and collision tests.
- Uses pyrender segmentation to evaluate occlusion/visibility per object.
- Writes CSV summaries (`object_bbox_dimensions.csv`, `object_appearance.csv`, `trajectory_data.csv`), segmentation PNGs, and `visual.pdf`.
- Keyframes the Blender camera, renders PNG sequences, and (optionally) composes videos.

**Batch CLI**
```bash
blender --background --python infinigen_examples/trajectory_optimizer.py -- \
    --batch-folder /data/scenes \
    --batch-output /data/trajectories \
    --batch-room-type "living-room_0/0.ceiling" \
    --batch-grid 0.2 \
    --batch-samples 500 \
    --batch-rotation-steps 30
```

**Helpful flags**
- Sampling space: `--batch-height`, `--batch-min-distance`, `--batch-max-distance`, `--batch-max-sight`.
- Visibility: `--batch-occlusion`, `--batch-occlusion-checks`.
- Rendering: `--batch-frame-prefix`, `--batch-frame-step`, `--batch-resolution`, `--batch-video-name`.
- Navigation safety: `--batch-robot-radius`.

If `--batch-folder` is omitted, the script defaults back to the frontier optimizer described in § 3.1.

---

## 4. QA Task Generation from Trajectory Metadata

**File**: `infinigen_examples/qa_from_metadata.py`

After running the batch trajectory pipeline, each output directory contains metadata CSVs (`object_bbox_dimensions.csv`, `object_appearance.csv`, etc.). Use the QA generator to synthesize evaluation tasks for multimodal models:

```bash
python infinigen_examples/qa_from_metadata.py \
    --metadata-dir /data/trajectories/scene_001 \
    --output /data/trajectories/scene_001/qa_tasks.json \
    --measurement-tasks 5 \
    --perspective-tasks 5 \
    --spatiotemporal-tasks 3 \
    --seed 42
```

**Task families**
- **Measurement tasks** ask for precise dimensions with contextual cues (e.g., “What’s the height of the oak cabinet next to the sofa?”) and are scored with mean relative accuracy.
- **Perspective-taking tasks** pose counting questions conditioned on the rendered trajectory (mean relative accuracy).
- **Spatiotemporal tasks** request the appearance order of multiple objects across the trajectory video and are evaluated via exact-match accuracy.

Each run emits a JSON payload describing the prompts, answers, and evaluation metrics, making it easy to integrate into auto-grading pipelines.

---

## Troubleshooting & Tips

- **Cluster performance**: Cluster moves broaden the search space. Trim their probability via gin configs if solves slow down.
- **Agentic compilation errors**: Failures are logged and retried up to `agentic_max_iterations`. Increase the limit or lower expectations when pointing at less reliable LLMs.
- **Trajectory planning**: For huge rooms, increase `--grid` to coarsen Dijkstra, or raise `--samples` / `--batch-samples` for better viewpoints.
- **Rendering**: Batch mode touches Blender’s Cycles settings; ensure GPU drivers are available if you switch from CPU-only installs.

Feel free to mix-and-match these features or extend the registry in `agentic_framework.py` to describe new procedural APIs. Pull requests are welcome!
