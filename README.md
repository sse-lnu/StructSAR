# StructSAR — Structural Graph Embeddings for Architecture Recovery

StructSAR is the reproducible replication package for the study
*Structural Graph Embeddings for Architecture Recovery*
(Jabeen, Ericsson, Nordqvist, and Wingkvist, Linnaeus University).

The package recovers the **module-level architecture** of a software system
directly from its source code. It learns file embeddings from static
dependency graphs in a self-supervised way and clusters them into modules, with
no labeled training data. It bundles the source code, the preprocessed
dependency graphs, the ground-truth architectures, and a single runner that
reproduces every method and metric reported in the paper.

## What the study investigates

The package is organized around two research questions:

- **RQ1 — Dependencies alone.** Can an architectural decomposition be recovered
  from the file-level *software dependency graph* (SDG) by itself, using only
  file-to-file static dependencies (calls, imports, inheritance, ...)?
- **RQ2 — Adding project layout.** Does augmenting the SDG with the project's
  folder structure (folder nodes and folder–file containment edges) yield a
  stronger recovery signal?

Each method represents a software system as a graph, learns a $d$-dimensional
embedding per file, and clusters those embeddings into a flat partition of
modules. Methods differ along two axes: the **input graph** (homogeneous
file–file SDG, or heterogeneous SDG + folders) and the **learning strategy**
(biased random walks with Skip-gram, or a GAT encoder trained inside a
variational graph autoencoder).

## Methods

| Key in config | Paper name | Input graph | Learner | Addresses |
|---------------|-----------|-------------|---------|-----------|
| `N2V`     | node2vec      | Homogeneous SDG                       | Biased random walks + Skip-gram, KMeans clustering | RQ1 |
| `M2V`     | MetaPath2Vec  | Heterogeneous (SDG + folders)         | Meta-path walks + type-aware Skip-gram | RQ2 |
| `GAT`     | GAT           | Homogeneous SDG                       | GAT encoder in a VGAE, with Laplacian positional encodings | RQ1 |
| `GAT_GDC` | GAT-DC        | Homogeneous SDG, PPR-diffused (GDC)   | GAT encoder in a VGAE over a diffusion-rewired graph | RQ1 |
| `HGAT`    | HGAT          | Heterogeneous (folders, merged deps)  | Relation-aware GAT encoder in a VGAE | RQ2 |
| `HGAT_TD` | HGAT-TD       | Heterogeneous, **typed** dependencies | Relation-aware GAT, one relation per dependency type | RQ2 |

All variants except `HGAT_TD` merge the different file–file dependency types
into a single edge; `HGAT_TD` keeps each dependency type as a separate relation.
All GAT-based variants share a self-supervised objective: a dot-product decoder
reconstructs the observed dependency edges from the learned embeddings.
Embeddings are clustered with agglomerative (Ward) clustering for every method
except `N2V`, which uses KMeans following the NEGAR setup. The number of
clusters is either fixed or selected from a candidate range via the silhouette
score.

## Benchmark systems

The package ships preprocessed graphs and ground-truth architectures for the
11 open-source Java, C, and C++ systems used in the paper. Static dependencies
were extracted with [Depends](https://github.com/multilang-depends/depends) and
aggregated to the file level to match the per-file ground truth.

| Language | System | Config name | GT modules | Files | Folders | Edges |
|----------|--------|-------------|-----------:|------:|--------:|------:|
| Java   | ArgoUML         | `A.UML`     | 14 | 766    | 58  | 3,754   |
| Java   | ArchStudio 4    | `AS4`       | 57 | 583    | 115 | 1,826   |
| Java   | Commons-Imaging | `C.Img`     | 21 | 329    | 41  | 1,565   |
| Java   | Hadoop          | `Hadoop`    | 62 | 591    | 55  | 3,648   |
| Java   | JabRef          | `JabRef`    | 6  | 1,180  | 127 | 6,126   |
| Java   | Lucene          | `Lucene`    | 7  | 1,054  | 188 | 6,424   |
| Java   | TeamMates       | `TeamMates` | 15 | 778    | 42  | 6,721   |
| C/C++  | Bash            | `Bash`      | 13 | 292    | 11  | 1,010   |
| C/C++  | Chromium        | `Chrome`    | 69 | 18,343 | 999 | 145,492 |
| C/C++  | Distributed Camera (OpenHarmony) | `HDC` | 11 | 207 | 87 | 543 |
| C/C++  | Drivers Framework (OpenHarmony)  | `HDF` | 9  | 153 | 26 | 535 |

Ground-truth architectures are reused from prior work
(Garcia et al., the SARIF benchmark, Olsson et al., and SAEroCon) so results are
directly comparable to earlier studies. Chromium is included as the
scalability stress test.

## Evaluation metrics

Because several reasonable decompositions can exist for the same system, the
runner reports complementary metrics rather than relying on a single criterion:

- **MoJoFM** — move/join edit distance to the ground truth (cluster level).
- **A2A** — architecture-to-architecture edit distance including add/remove/move
  of entities and clusters.
- **C2C coverage** (`c2c_cvg_33/50/66/80`) — fraction of ground-truth clusters
  sufficiently matched by some recovered cluster, at several coverage
  thresholds; `0.50` is the majority-match value emphasized in the paper.
- **Normalized TurboMQ** — ground-truth-independent modular quality (intra-cluster
  cohesion vs. inter-cluster coupling).
- **ARI** — Adjusted Rand Index, chance-corrected agreement with the ground truth.

## Repository layout

```text
src/structsar/            Source code
  run_experiments.py        Experiment runner (entry point)
  data_builders/            Homogeneous / heterogeneous / walk graph builders
  models/                   node2vec, metapath2vec, homogeneous & heterogeneous GAT
  eval/                     Clustering and metric evaluation
  metrics/                  MoJoFM, A2A, C2C, TurboMQ implementations
experiment_config.json    Datasets, methods, hyperparameters, evaluation settings
data/processed/           Preprocessed file (`*.csv`) and dependency (`*_deps.csv`) tables
data/GT/                  Ground-truth labels (`*_gt.csv` / `*_gt.json`)
requirements.txt          Python dependencies
Results/                  Output metrics and cluster assignments (created on run)
```

## Clone

The processed datasets are tracked with Git LFS.

```bash
git lfs install
git clone https://github.com/sse-lnu/StructSAR.git
cd StructSAR
git lfs pull
```

If you do not have Git LFS, install it from <https://git-lfs.com/> first.

## Install

Create a Python environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

PyTorch and PyTorch Geometric must match your CPU/CUDA setup. If the generic
install does not match your system, follow the official instructions:

- <https://pytorch.org/get-started/locally/>
- <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>

A GPU is recommended but not required; the smaller systems run comfortably on CPU.

## Quick start (sanity check)

Run one method on the smallest system to confirm the setup works. After
dependencies are installed, this finishes in seconds:

```bash
PYTHONPATH=src python -m structsar.run_experiments --config experiment_config.json --only GAT --datasets Bash
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m structsar.run_experiments --config experiment_config.json --only GAT --datasets Bash
```

## Reproduce the paper

Run every method on every benchmark system:

```bash
PYTHONPATH=src python -m structsar.run_experiments --config experiment_config.json
```

Select methods with `--only` and systems with `--datasets`:

```bash
# RQ1: dependency-only methods
PYTHONPATH=src python -m structsar.run_experiments --config experiment_config.json --only N2V GAT GAT_GDC

# RQ2: folder-aware methods
PYTHONPATH=src python -m structsar.run_experiments --config experiment_config.json --only M2V HGAT HGAT_TD

# A single method on selected systems
PYTHONPATH=src python -m structsar.run_experiments --config experiment_config.json --only GAT_GDC --datasets Bash Hadoop Chrome
```

Method names passed to `--only` are case-insensitive. Full help:

```bash
PYTHONPATH=src python -m structsar.run_experiments --help
```

## Configuration

All experiment settings live in `experiment_config.json`; the CLI only selects
which methods and datasets from that file to run.

Common options (`common` block):

- `evaluate` — `true` computes and writes metrics; `false` writes runtime-only
  rows plus file-to-cluster assignments.
- `exact_k` / `n_clusters` — fixed number of clusters for the `exact_k` row.
- `k_min`, `k_max` — silhouette search range for the `search` row.
- `k_range_overrides` — per-dataset search ranges (e.g. `AS4`, `Hadoop`, `Chrome`).
- `num_runs` — number of repeated runs (for stability analysis).
- `minibatch_threshold_files` — file count above which `GAT`/`GAT_GDC` switch to
  neighbor-sampled minibatch training, then compute final embeddings with
  full-graph inference (default `5000`, which triggers for Chrome).
- `batch_size`, `neighbor_batch_size` — minibatch training settings.

Per-method hyperparameters and Chrome-specific scalability overrides live in the
`experiments` block and mirror the values reported in the paper.

## Use your own system

Place two preprocessed CSV files in `data/processed`:

- a **file table** with a `File` column (and a `Module` column for evaluation);
- a **dependency table** with `Source_File`, `Target_File`, `Dependency_Type`,
  and `Dependency_Count` columns.

Register the dataset under `common`:

```json
"datasets": ["MySystem"],
"custom_datasets": {
  "MySystem": {
    "nodes": "mysystem.csv",
    "dependencies": "mysystem_deps.csv"
  }
}
```

Then run:

```bash
PYTHONPATH=src python -m structsar.run_experiments --config experiment_config.json --datasets MySystem
```

For user-supplied systems, `GAT_GDC` applies these fallback rules unless you
override them in the config:

- more than `500` files **and** more than `10` ground-truth modules → `1` GAT layer; otherwise `2`;
- fewer than `300` files → `gdc_k = 16`; `300`–`432` files → `gdc_k = 32`; more than `432` files → `gdc_k = 64`.

The paper systems keep their explicit settings in `experiment_config.json`.

## Outputs

Each method writes one combined CSV across all selected systems:

```text
Results/N2V/results.csv
Results/M2V/results.csv
Results/GAT/results.csv
Results/GAT_GDC/results.csv
Results/HGAT/results.csv
Results/HGAT_TD/results.csv
```

Metric CSV columns:

```text
Model, Dataset, run_id, clustering, algorithm, n_clusters, search_range,
mojofm, a2a, c2c_cvg_33, c2c_cvg_50, c2c_cvg_66, c2c_cvg_80,
ari, normalized_turbomq, turbomq, total_pipeline_seconds
```

When `common.evaluate` is `false`, runtime rows are written to a single file:

```text
Results/no_eval_results.csv
```

with columns:

```text
Method, Dataset, run_id, clustering, algorithm, n_clusters,
search_range, total_pipeline_seconds, assignment_file
```

Cluster-label assignments are saved as JSON under the corresponding
`Results/<Model>/` folder in both modes.

## Citation

If you use this package, please cite the accompanying paper. Author and venue
details are in the paper sources under `StructSAR/`.

## License

The source code is released under the BSD 3-Clause License. The data in `data/`
is dedicated to the public domain (CC0 1.0).
