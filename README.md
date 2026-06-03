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

**Homogeneous** methods (RQ1) operate on the file–file dependency graph alone.
**Heterogeneous** methods (RQ2) extend the same graph with folder nodes and
folder–file containment edges.

| Method | Config name | What it adds |
|--------|-------------|--------------|
| **Homogeneous** | | |
| node2vec | `N2V`     | Biased random walks + Skip-gram (KMeans clustering, NEGAR setup) |
| GAT      | `GAT`     | GAT encoder in a VGAE, with Laplacian positional encodings |
| GAT-DC   | `GAT_GDC` | GAT in a VGAE over a PPR-diffused (GDC) graph |
| **Heterogeneous** | | |
| MetaPath2Vec | `M2V`     | Meta-path random walks + type-aware Skip-gram |
| HGAT         | `HGAT`    | Relation-aware GAT in a VGAE (dependency types merged into one edge) |
| HGAT-TD      | `HGAT_TD` | Same as HGAT, but keeps each dependency type as a separate relation |

All GAT-based variants share one self-supervised objective: a dot-product
decoder reconstructs the observed dependency edges from the learned embeddings.
Embeddings are clustered with agglomerative (Ward) clustering for every method
except `N2V`, which uses KMeans. The number of clusters is either fixed or
selected from a candidate range via the silhouette score.

## Datasets and file formats

The package ships preprocessed graphs and ground-truth architectures for 11
open-source Java, C, and C++ systems used in the paper (ArgoUML, ArchStudio 4,
Commons-Imaging, Hadoop, JabRef, Lucene, TeamMates, Bash, Chromium, and two
OpenHarmony systems). Static dependencies were extracted with
[Depends](https://github.com/multilang-depends/depends) and aggregated to the
file level to match the per-file ground truth. The ground-truth architectures
are reused from prior work (Garcia et al., the SARIF benchmark, Olsson et al.,
and SAEroCon) so results are directly comparable to earlier studies. Chromium is
the largest system and serves as the scalability stress test.

Each system is stored as three files. Their config names are in
`experiment_config.json`; the per-system file names are in `PAPER_DATASETS` in
`src/structsar/run_experiments.py`.

### `data/processed/<system>.csv` — entities

One row per source-file member. Columns used by the pipeline:

- `File` — source filename, including path (the graph node identity).
- `Module` — the architectural module the file belongs to (ground truth).
- `ID`, `Member_ID` — unique integer IDs for the entity and member.
- `Member_Name`, `Member_Type` — name and kind of a member (e.g. a function).
- `Entity` — the fully qualified entity name (e.g. the Java class name).

### `data/processed/<system>_deps.csv` — dependencies

One row per static dependency. Many fields reference entities in the file above.

- `Source_File`, `Target_File` — the two files the dependency connects (the edge).
- `Dependency_Type` — the kind of dependency, e.g. `Call`, `Import`, `Extend`.
- `Dependency_Count` — number of such dependencies between source and target.
- `Source_ID`, `Target_ID` — entity ID references.
- `Source_Member`, `Target_Member`, `*_Member_Type`, `*_Member_ID` — member-level detail.
- `Is_Member_Level` — `True` if the dependency is between members rather than files.
- `Source_Module`, `Target_Module` — the ground-truth module of each endpoint.

The homogeneous methods read only `Source_File`/`Target_File` (merging all
dependency types into one edge); `HGAT_TD` additionally splits edges by
`Dependency_Type`; the folder relations used by the heterogeneous methods are
derived from the directory part of `File`.

### `data/GT/<system>_gt.csv` and `<system>_gt.json` — ground truth

The evaluation labels, provided in two equivalent forms:

- `*_gt.csv` — one row per entity with `Entity`, `Module` (the selected module),
  `Module_List` (all candidate modules when an entity maps to several), and
  `is_duplicated`.
- `*_gt.json` — the same architecture as a nested `group` → `item` structure.

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

## Example run

A minimal run — one method (`GAT`) on the smallest system (`Bash`). It finishes
in seconds and writes its results under `Results/GAT/`:

```bash
PYTHONPATH=src python -m structsar.run_experiments --config experiment_config.json --only GAT --datasets Bash
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m structsar.run_experiments --config experiment_config.json --only GAT --datasets Bash
```

From here, drop `--only`/`--datasets` to run everything, or list other methods
and systems (see below).

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

No license has been set for this repository yet. Until one is added, all rights
are reserved; please contact the authors before reusing the code or data.
