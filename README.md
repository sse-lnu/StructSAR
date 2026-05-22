# StructSAR

StructSAR is a reproducible experiment package for software architecture recovery with graph embedding and graph neural network models.

## Repository Layout

- `src/structsar`: source code for graph construction, models, clustering, evaluation, and the experiment runner.
- `experiment_config.json`: experiment configuration used by the runner.
- `data/processed`: processed file/module and dependency CSV data.
- `data/GT`: ground-truth architecture labels.

Raw dependency-extraction outputs are not included.

## Models

- `N2V`: Node2Vec baseline.
- `M2V`: MetaPath2Vec baseline.
- `GAT`: homogeneous GAT with dependency edges and Laplacian positional encodings.
- `GAT_GDC`: homogeneous GAT with GDC/PPR-diffused dependency edges.
- `HGAT`: heterogeneous GAT with file, folder, and collapsed file-dependency relations.

## Clone

This repository uses Git LFS for processed datasets. Install Git LFS before cloning.

```bash
git lfs install
git clone https://github.com/sse-lnu/StructSAR.git
cd StructSAR
git lfs pull
```

## Install

Create and activate a Python environment, then install the required scientific Python and PyTorch Geometric stack for your platform.

Minimum Python packages used by the runner include:

```bash
pip install numpy pandas scikit-learn tqdm
```

Install PyTorch and PyTorch Geometric following the official instructions for your CUDA/CPU setup:

- https://pytorch.org/get-started/locally/
- https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Run

Run all configured datasets and all methods:

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

On Windows PowerShell, set `PYTHONPATH` first:

```powershell
$env:PYTHONPATH = "src"
python -m structsar.run_experiments --config experiment_config.json
```

## Configuration

The main configuration file is `experiment_config.json`.

Important options:

- `common.evaluate`: when `true`, writes evaluation metrics; when `false`, writes file-to-cluster assignments.
- `common.exact_k` or `common.n_clusters`: exact number of clusters for `exact_k`.
- `common.k_min` and `common.k_max`: search range for the `search` clustering row.
- `common.k_range_overrides`: dataset-specific search ranges.
- `common.neighbor_batch_size`: minibatch neighbor fanout, default `10`.
- `common.batch_size`: minibatch training batch size, default `1024`.
- `common.inference_batch_size`: minibatch inference batch size, default `1024`.

Clustering defaults:

- `N2V`: KMeans.
- `M2V`, `GAT`, `GAT_GDC`, `HGAT`: agglomerative clustering.

Each evaluated run writes two rows per dataset and method: one `exact_k` row and one `search` row.

## Minibatching

`GAT` and `GAT_GDC` automatically switch to minibatch training when a dataset has more than `common.minibatch_threshold_files` files. The default threshold is `5000`, so Chrome uses minibatching by default.

Minibatch training is implemented in `src/structsar/models/homogeneous_gat.py` and is launched through the single runner `src/structsar/run_experiments.py`.

## Outputs

Results are written under `Results/<Model>/results.csv`.

Example:

```text
Results/N2V/results.csv
Results/M2V/results.csv
Results/GAT/results.csv
Results/GAT_GDC/results.csv
Results/HGAT/results.csv
```

Each method writes one combined CSV containing all selected datasets.

Result columns:

```text
Model, Dataset, run_id, clustering, algorithm, n_clusters, search_range,
mojofm, a2a, c2c_cvg_33, c2c_cvg_50, c2c_cvg_66, c2c_cvg_80,
ari, normalized_turbomq, turbomq, total_pipeline_seconds
```

`search_range` uses a spreadsheet-safe format such as `k=5..20`.

When `common.evaluate` is `false`, assignment JSON files are written instead, containing each file and its assigned cluster.

## Help

```bash
PYTHONPATH=src python -m structsar.run_experiments --help
```
