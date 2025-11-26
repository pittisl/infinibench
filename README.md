# InfiniBench Agentic & Trajectory Enhancements

This README describes the new capabilities added on top of the standard InfiniBench codebase:

1. **Cluster-Based Scene Optimization** – furniture clusters (e.g., dining tables with chairs) can now be moved, rotated, and resampled as unified entities during the object-solving stage.
2. **Agentic Constraint Generation** – an LLM-powered loop that turns a free-form scene description into procedural constraints and iteratively refines them using optimizer feedback.
3. **Camera Trajectory Optimization** – a frontier-inspired algorithm that produces compact, collision-free camera paths covering every task-relevant object.

Each section below explains where the implementation lives and how to activate it.

---

## 1. Cluster-Based Layout Optimization

**Key files**
- `infinigen/core/constraints/example_solver/clusters.py`
- `infinigen/core/constraints/example_solver/moves/cluster.py`
- `infinigen/core/constraints/example_solver/propose_clusters.py`
- `infinigen/core/constraints/example_solver/solve.py`

**What changed**
- Clusters are dynamically identified from `StableAgainst` relations (e.g., chairs supported by the same table).
- New move primitives (`cluster_translate`, `cluster_rotate`, `cluster_resample`) keep all children rigidly coupled to their parent when exploring the search space.
- Collision checks use a cluster-wide axis-aligned bounding box before accepting a move.

**How to use**
- The moves are active by default during `Solver.solve_objects`. To restrict/weight them, edit your gin config:
  ```gin
  Solver.restrict_moves = ["addition", "cluster_translate", "cluster_rotate"]
  ```
- All logging goes through the existing solver logger (`infinigen.core.constraints.example_solver.solve`).

---

## 2. Agentic Constraint Generation

**Key files**
- `infinigen_examples/constraints/agentic_framework.py`
- `infinigen_examples/generate_indoors.py` (new CLI flags)

**What changed**
- `AgenticConstraintGenerator` composes prompts using API docs + in-context examples (the default example is `home_furniture_constraints`).
- `AgenticSceneGenerator` runs an iterative loop: generate constraints → compile/validate → (optionally) feed optimizer feedback back to the LLM, enforcing chain-of-thought reasoning.
- `compose_indoors()` supports two new flags:
  - `scene_description`: natural-language request (e.g., “a cluttered startup office”)
  - `use_agentic_constraints`: toggle agentic mode on/off
  - `agentic_max_iterations`: bound the CoT loop length

**Example**
```bash
python infinigen_examples/generate_indoors.py \
    --scene_description "compact studio apartment with plants and wall art" \
    --use_agentic_constraints True \
    --agentic_max_iterations 3 \
    -p solve_steps_large=400
```

Under the hood, the agent compiles the generated code, injects it via `constraint_builder = agentic_result.final_program.to_callable(...)`, and then the usual greedy + simulated annealing pipeline takes over.

> **Note:** By default, the provided `DummyLLM` simply replays the reference example. Point `AgenticConstraintGenerator` at a real LLM client to produce novel programs.

---

## 3. Camera Trajectory Optimization

**File**: `infinigen_examples/trajectory_optimizer.py`

**Concept**
- Implements the four-step loop from the paper:
  1. Select the nearest unvisited target object.
  2. Sample viewpoint candidates (checking accessibility, field of view coverage, and occlusions).
  3. Plan a path via Dijkstra on a 2D navigation graph (camera height assumed constant).
  4. Move to the best-scoring viewpoint and mark the object as visited.
- The resulting trajectory is exported as a JSON list of `{position, rotation_euler}` entries.

**CLI usage**
```bash
blender --background --python infinigen_examples/trajectory_optimizer.py -- \
    --blend /path/to/scene.blend \
    --output /tmp/trajectory.json \
    --samples 1500 \
    --grid 0.6
```

The script opens the `.blend` file, identifies mesh targets (excluding lights, windows, etc.), builds the navigation graph, and writes the trajectory to `--output`.

---

## Troubleshooting & Tips

- **Performance**: Cluster moves add more candidate proposals; reduce their weights via gin if your solve time increases too much.
- **LLM Failures**: If an LLM-generated constraint crashes compilation, the agent logs the exception and retries up to `agentic_max_iterations`.
- **Trajectory Planning**: For large scenes, increase `--grid` to coarsen the navigation graph, or raise `--samples` for higher-quality viewpoints.

Feel free to adapt the registry/examples in `agentic_framework.py` to teach the LLM agent about other procedural APIs or constraint sets.
