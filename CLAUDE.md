# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full benchmark (single-size accuracy + N-scaling, outputs a PNG chart)
python -m nbody_graph.benchmark.run_benchmark
```

The benchmark saves its chart to `/mnt/user-data/outputs/benchmark_results.png`.

## Architecture

This is a Python research framework for comparing N-body gravitational simulation methods. The package name is `nbody_graph`; the repo root acts as its parent directory.

### Core data flow

All methods share the same interface: accept a `Particles` object, return an `ndarray` of shape `(N, 3)` accelerations. The direct method is the reference; all others are measured for accuracy against it via RMS relative error.

**`utils/core.py`** — shared primitives:
- `Particles` dataclass holds `pos`, `vel`, `mass` (all `ndarray`), and auto-initialized `acc`
- `make_random_particles(N, seed)` — reproducible test data (AU/M☉ scale)
- `leapfrog_step(p, acc_fn, dt)` — Störmer-Verlet integrator
- Constants: `G = 6.674e-11`, `SOFTENING = 1e-3`

**`methods/`** — four acceleration solvers:
- `direct.py` — fully vectorized O(N²); used as ground truth
- `barnes_hut.py` — `_FullOctree` class + `compute_acceleration_barneshut(p, theta)`; theta controls accuracy/speed trade-off
- `fmm.py` — simplified cell-based FMM (monopole + dipole moments); `compute_acceleration_fmm(p, n_cells_per_dim, near_threshold)`
- `graph_compression.py` — **the primary research extension point** (see below)

**`benchmark/run_benchmark.py`** — orchestrates timing + accuracy comparison across all methods and produces matplotlib output. Note: `compute_acceleration_graph` returns `(acc, n_kept_edges, n_total_edges)`, unlike the other methods which return just `acc`.

### Graph compression method

`graph_compression.py` is designed to be extended. The pipeline is:

1. `build_interaction_graph(p, cutoff)` — builds edges `(i, j, weight)` where `weight = G·mᵢ·mⱼ / r²`
2. `compress_graph_threshold(edges, keep_ratio)` — **replace this function** with a better compression algorithm
3. `compute_acceleration_graph(p, keep_ratio, cutoff)` — calls steps 1–2 then computes forces over kept edges

Candidate compression replacements noted in the code: Spielman-Srivastava spectral sparsification, Louvain/Leiden community detection, coreset approximation, random-walk clustering.
