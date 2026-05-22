# StructGAT Experiments

Clean runnable project code for the final architecture-recovery experiments.

## Repository Layout

- `src/structsar`: source code for the models, graph builders, evaluation, and runner.
- `experiment_config.json`: paper experiment configuration.
- `data/processed`: processed dependency and file/module CSV data.
- `data/GT`: ground-truth labels.

## Data

This repository includes processed experiment data and ground-truth labels:

- `data/processed`
- `data/GT`

Raw Depends outputs are not committed.

## Methods

- `N2V`: Node2Vec baseline.
- `M2V`: MetaPath2Vec baseline.
- `GAT`: homogeneous GAT with raw dependencies and LPE.
- `GAT_GDC`: homogeneous GAT with PPR-diffused dependencies and LPE.
- `HGAT`: heterogeneous GAT with file, folder, and collapsed file-dependency relations.

The N2V and M2V variants and model hyperparameters are kept in `experiment_config.json`.

## Run Experiments

From Anaconda Prompt:

```powershell
cd C:\Users\JABEERAK\Architecture_Recovery\NAIM\StructGAT
conda activate torchDL
$env:PYTHONPATH = "src"
python -m structsar.run_experiments --config experiment_config.json
```

Run only selected methods:

```powershell
python -m structsar.run_experiments --config experiment_config.json --only N2V
python -m structsar.run_experiments --config experiment_config.json --only GAT GAT_GDC HGAT
```

Run only selected datasets:

```powershell
python -m structsar.run_experiments --config experiment_config.json --datasets Bash Chrome
python -m structsar.run_experiments --config experiment_config.json --only GAT --datasets Chrome
```

You can also run with the full Python path:

```powershell
$env:PYTHONPATH = "src"
C:\Users\JABEERAK\.conda\envs\torchDL\python.exe -m structsar.run_experiments --config experiment_config.json
```

## Clustering Setting

The final config uses agglomerative clustering globally, with N2V explicitly set to KMeans:

- `common.clustering_algorithm`: `"agglomerative"`
- `N2V.clustering_algorithm`: `"kmeans"`

When `common.evaluate` is `true`, every run writes two rows: one for `exact_k` and one for `search`.

When `common.evaluate` is `false`, the runner skips evaluation metrics and writes JSON files with file-to-cluster assignments:

```json
{
  "runs": [
    {
      "Dataset": "Bash",
      "Model": "GAT",
      "run_id": 1,
      "clustering": "exact_k",
      "n_clusters": 2,
      "search_range": "5-20",
      "total_pipeline_seconds": 12.34,
      "assignments": [
        {"file": "src/example.c", "cluster": 0}
      ]
    }
  ]
}
```

## GAT Minibatching

`GAT` and `GAT_GDC` automatically switch to minibatch training when a dataset has more than `common.minibatch_threshold_files` files. The default threshold is `5000`, matching the large-system setting used for Chrome. Smaller systems keep the normal full-graph GAT path.

Minibatch training is implemented inside `src/structsar/models/homogeneous_gat.py` and is launched through the single runner, `src/structsar/run_experiments.py`. Defaults are:

- `common.neighbor_batch_size`: `10`
- `common.batch_size`: `1024`
- `common.inference_batch_size`: `1024`

Users can override those values in `experiment_config.json`.

## GAT_GDC Top-K

The selected `gdc_k` values are stored in the `GAT_GDC` dataset overrides:

- `HDC`, `HDF`: `16`
- `Bash`, `C.Img`: `32`
- `AS4`, `Hadoop`, `JabRef`: `64`
- `TeamMates`, `A.UML`: `128`
- `Chrome`: `16`

Chrome is kept at `16` for the current experiments because its graph is much larger; larger GDC top-k values make the diffused graph expensive before batch-size experiments are finalized.

## Outputs

Results are saved in:

- `Results/N2V`
- `Results/M2V`
- `Results/GAT`
- `Results/GAT_GDC`
- `Results/HGAT`

Each method writes one combined CSV:

```text
Results/<Model>/results.csv
```

Each result CSV is kept clean for paper reporting:

```text
Model, Dataset, run_id, clustering, algorithm, n_clusters, search_range,
mojofm, a2a, c2c_cvg_33, c2c_cvg_50, c2c_cvg_66, c2c_cvg_80,
ari, normalized_turbomq, turbomq, total_pipeline_seconds
```

`n_clusters` is the exact or selected number of clusters. `search_range` records the searched cluster range, for example `5-20`. With evaluation enabled, `total_pipeline_seconds` includes model training, clustering, and metric evaluation. With evaluation disabled, the JSON assignment output reports the total runtime without metric evaluation.

HGAT collapses dependency types into one `depends_on` relation by default through `common.collapse_hgat_dependencies: true`. Set it to `false` only when you explicitly want separate dependency-type relations.

## Useful Help

```powershell
$env:PYTHONPATH = "src"
python -m structsar.run_experiments --help
```
