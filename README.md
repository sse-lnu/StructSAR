# StructSAR

StructSAR is a reproducible experiment package for software architecture recovery using graph embeddings and graph neural networks.

The repository includes the final source code, processed datasets, and ground-truth labels used by the experiment runner.

## Repository Layout

```text
src/structsar/          Source code
experiment_config.json  Experiment configuration
data/processed/         Processed file and dependency CSVs
data/GT/                Ground-truth labels
requirements.txt        Python dependencies
```

## Models

The runner supports:

- `N2V`: Node2Vec baseline with KMeans clustering.
- `M2V`: MetaPath2Vec baseline with agglomerative clustering.
- `GAT`: homogeneous GAT with dependency edges and Laplacian positional encodings.
- `GAT_GDC`: homogeneous GAT with GDC/PPR-diffused dependency edges.
- `HGAT`: heterogeneous GAT with file, folder, and collapsed dependency relations.

## Clone

This repository uses Git LFS for processed datasets.

```bash
git lfs install
git clone https://github.com/sse-lnu/StructSAR.git
cd StructSAR
git lfs pull
```

## Install

Create a Python environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

PyTorch and PyTorch Geometric should match your CPU/CUDA setup. If the generic install above does not match your system, install them using the official instructions:

- https://pytorch.org/get-started/locally/
- https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Run Experiments

Run all configured methods and datasets:

```bash
PYTHONPATH=src python -m structsar.run_experiments --config experiment_config.json
```

Run selected methods:

```bash
PYTHONPATH=src python -m structsar.run_experiments --config experiment_config.json --only N2V
PYTHONPATH=src python -m structsar.run_experiments --config experiment_config.json --only GAT GAT_GDC HGAT
```

Run selected datasets:

```bash
PYTHONPATH=src python -m structsar.run_experiments --config experiment_config.json --datasets Bash Chrome
PYTHONPATH=src python -m structsar.run_experiments --config experiment_config.json --only GAT --datasets Chrome
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m structsar.run_experiments --config experiment_config.json
```

## Configuration

Edit `experiment_config.json` to choose datasets, methods, model hyperparameters, and evaluation behavior.

Common options:

- `common.evaluate`: `true` writes metrics; `false` writes file-to-cluster assignments.
- `common.exact_k` or `common.n_clusters`: exact number of clusters for the `exact_k` row.
- `common.k_min` and `common.k_max`: cluster search range for the `search` row.
- `common.k_range_overrides`: dataset-specific search ranges.
- `common.num_runs`: number of repeated runs.
- `common.batch_size`: minibatch training batch size, default `1024`.
- `common.neighbor_batch_size`: minibatch neighbor fanout, default `10`.
- `common.minibatch_threshold_files`: file-count threshold for GAT minibatching, default `5000`.

`GAT` and `GAT_GDC` automatically use minibatch training for large systems such as Chrome.

## Outputs

Each method writes one combined CSV containing all selected datasets:

```text
Results/N2V/results.csv
Results/M2V/results.csv
Results/GAT/results.csv
Results/GAT_GDC/results.csv
Results/HGAT/results.csv
```

Metric CSV columns:

```text
Model, Dataset, run_id, clustering, algorithm, n_clusters, search_range,
mojofm, a2a, c2c_cvg_33, c2c_cvg_50, c2c_cvg_66, c2c_cvg_80,
ari, normalized_turbomq, turbomq, total_pipeline_seconds
```

`search_range` is written in a spreadsheet-safe format, for example:

```text
5_to_20
```

When `common.evaluate` is `false`, assignment JSON files are written under the corresponding `Results/<Model>/` folder.

## Help

```bash
PYTHONPATH=src python -m structsar.run_experiments --help
```
