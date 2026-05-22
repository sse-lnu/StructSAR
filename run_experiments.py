r"""Run the final architecture-recovery experiments.

Typical use from Anaconda Prompt:

    cd C:\Users\JABEERAK\Architecture_Recovery\NAIM\StructGAT
    conda activate torchDL
    python run_experiments.py --config experiment_config.json

The config controls datasets, methods, output folders, and clustering. The
global clustering choice can be "kmeans" or "agglomerative"; an experiment can
override it. When common.evaluate is true, the runner writes evaluation metrics
and total pipeline runtime. When common.evaluate is false, it writes JSON
file-to-cluster assignments and total pipeline runtime.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

FAST_HELP = """usage: run_experiments.py [-h] [--config CONFIG] [--only [ONLY ...]] [--datasets [DATASETS ...]]

Run final N2V, M2V, GAT, GAT_GDC, and HGAT experiments.

options:
  -h, --help         show this help message and exit
  --config CONFIG    Path to the experiment JSON config.
  --only [ONLY ...]  Optional method names to run, for example: --only N2V GAT_GDC HGAT
  --datasets [DATASETS ...]
                     Optional dataset names to run, for example: --datasets Bash Chrome

How to run:
  cd C:\\Users\\JABEERAK\\Architecture_Recovery\\NAIM\\StructGAT
  conda activate torchDL
  python run_experiments.py --config experiment_config.json

Run only selected methods:
  python run_experiments.py --config experiment_config.json --only N2V
  python run_experiments.py --config experiment_config.json --only GAT GAT_GDC HGAT

Run only selected datasets:
  python run_experiments.py --config experiment_config.json --datasets Bash Chrome

Clustering is controlled in experiment_config.json:
  common "clustering_algorithm": "agglomerative"
  N2V experiment "clustering_algorithm": "kmeans"
  With evaluation enabled, each selected algorithm writes exact_k + silhouette search rows.

GAT minibatching:
  GAT and GAT_GDC use minibatching automatically when file count exceeds common.minibatch_threshold_files.

GAT_GDC uses dataset-specific top-k values when known. If gdc_k is "auto",
the runner picks the nearest known profile by graph size and average degree.

Outputs are saved under Results/N2V, Results/M2V, Results/GAT, Results/GAT_GDC, and Results/HGAT.
With common.evaluate=false, outputs are JSON file-to-cluster assignments.
"""

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print(FAST_HELP)
    sys.exit(0)

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from StructGAT.eval import Clusterer, append_csv, clean_csv_schema
    from StructGAT.data_builders import HeterogeneousData, Node2VecGraph, build_metapath_graph
    from StructGAT.data_builders.homogeneous_graph import normalize_df
    from StructGAT.models import (
        train_heterogeneous_gat_embeddings,
        train_homogeneous_gat_embeddings,
        train_metapath2vec_embeddings,
        train_node2vec_embeddings,
    )
    from StructGAT.models.heterogeneous_gat import lpe_features as hetero_lpe_features
    from StructGAT.models.homogeneous_gat import (
        HomogeneousFileGraphData,
        gdc_edge_index_approx,
        gdc_edge_index,
        lpe_input as homo_lpe_input,
        train_homogeneous_gat_minibatch_embeddings,
    )
else:
    from .eval import Clusterer, append_csv, clean_csv_schema
    from .data_builders import HeterogeneousData, Node2VecGraph, build_metapath_graph
    from .data_builders.homogeneous_graph import normalize_df
    from .models import (
        train_heterogeneous_gat_embeddings,
        train_homogeneous_gat_embeddings,
        train_metapath2vec_embeddings,
        train_node2vec_embeddings,
    )
    from .models.heterogeneous_gat import lpe_features as hetero_lpe_features
    from .models.homogeneous_gat import (
        HomogeneousFileGraphData,
        gdc_edge_index_approx,
        gdc_edge_index,
        lpe_input as homo_lpe_input,
        train_homogeneous_gat_minibatch_embeddings,
    )


DATASETS = {
    "A.UML": ("argouml.csv", "argouml_deps.csv"),
    "Ant": ("ant.csv", "ant_deps.csv"),
    "AS4": ("archstudio.csv", "archstudio_deps.csv"),
    "Bash": ("bash.csv", "bash_deps.csv"),
    "C.Img": ("commons.csv", "commons_deps.csv"),
    "Hadoop": ("hadoop.csv", "hadoop_deps.csv"),
    "HDC": ("hdc.csv", "hdc_deps.csv"),
    "HDF": ("hdf.csv", "hdf_deps.csv"),
    "JabRef": ("jabref.csv", "jabref_deps.csv"),
    "LibXML": ("libxml.csv", "libxml_deps.csv"),
    "Lucene": ("lucene.csv", "lucene_deps.csv"),
    "OODT": ("oodt.csv", "oodt_deps.csv"),
    "Pandas": ("pandas.csv", "pandas_deps.csv"),
    "SH-3D": ("sweetHome.csv", "sweetHome_deps.csv"),
    "TeamMates": ("teammates.csv", "teammates_deps.csv"),
    "Chrome": ("chromium.csv", "chromium_deps.csv"),
}

METHOD_DEFAULT_ALGORITHM = {
    "negar": "agglomerative",
    "metapath2vec": "agglomerative",
    "homogeneous_gat": "agglomerative",
    "heterogeneous_gat": "agglomerative",
    "structural_gat": "agglomerative",
}

GDC_K_BY_DATASET = {
    "HDC": 16,
    "HDF": 16,
    "Bash": 32,
    "C.Img": 32,
    "AS4": 64,
    "Hadoop": 64,
    "JabRef": 64,
    "A.UML": 128,
    "TeamMates": 128,
}

GDC_K_PROFILES = {
    "HDC": 16,
    "HDF": 16,
    "Bash": 32,
    "C.Img": 32,
    "AS4": 64,
    "Hadoop": 64,
    "JabRef": 64,
    "A.UML": 128,
    "TeamMates": 128,
}


def slug(value):
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(value)).strip("_")


def deep_update(base, update):
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_random_state(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def selected_datasets(common):
    names = list(common.get("datasets") or DATASETS.keys())
    include = common.get("include_datasets") or []
    exclude = set(common.get("exclude_datasets") or [])
    if include:
        allowed = set(include)
        names = [name for name in names if name in allowed]
    names = [name for name in names if name not in exclude]
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset names: {unknown}. Available datasets: {sorted(DATASETS)}")
    return names


def experiment_dir(out_dir, exp):
    parts = exp.get("output_path") or [exp.get("settings", {}).get("Method", exp["name"])]
    if isinstance(parts, str):
        parts = [parts]
    return Path(out_dir).joinpath(*(slug(part) for part in parts))


def result_path(exp_dir, dataset=None):
    return Path(exp_dir) / "results.csv"


def assignment_path(exp_dir, dataset):
    return Path(exp_dir) / f"{slug(dataset)}_clusters.json"


def completed_run_ids(path, dataset, method_name, rows_per_run):
    path = Path(path)
    if not path.exists():
        return set()
    clean_csv_schema(path)
    try:
        df = pd.read_csv(path, usecols=lambda col: col in {"Dataset", "Model", "dataset", "Method", "run_id"})
    except (ValueError, pd.errors.EmptyDataError):
        return set()
    if df.empty or "run_id" not in df.columns:
        return set()
    if "Dataset" in df.columns:
        df = df[df["Dataset"] == dataset]
    elif "dataset" in df.columns:
        df = df[df["dataset"] == dataset]
    if "Model" in df.columns:
        df = df[df["Model"] == method_name]
    elif "Method" in df.columns:
        df = df[df["Method"] == method_name]
    counts = pd.to_numeric(df["run_id"], errors="coerce").dropna().astype(int).value_counts()
    return set(counts[counts >= int(rows_per_run)].index.astype(int).tolist())


def completed_assignment_run_ids(path, dataset, method_name):
    path = Path(path)
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    runs = payload.get("runs", []) if isinstance(payload, dict) else []
    done = set()
    for run in runs:
        if not isinstance(run, dict):
            continue
        run_dataset = run.get("Dataset", run.get("dataset"))
        run_model = run.get("Model", run.get("method"))
        if run_dataset == dataset and run_model == method_name and "run_id" in run:
            done.add(int(run["run_id"]))
    return done


def append_assignment_json(path, run_payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"runs": []}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
                payload = existing
        except json.JSONDecodeError:
            pass
    payload["runs"].append(run_payload)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def normalize_clustering_algorithm(value):
    if value is None:
        return "default"
    value = str(value).strip().lower().replace("_", "-")
    if value in {"", "default", "paper", "auto"}:
        return "default"
    if value in {"kmeans", "k-means", "k-mean"}:
        return "kmeans"
    if value in {"agglomerative", "agg", "hierarchical"}:
        return "agglomerative"
    raise ValueError("clustering_algorithm must be 'default', 'kmeans', or 'agglomerative'.")


def clustering_algorithm_for_experiment(exp, method, common):
    configured = normalize_clustering_algorithm(
        exp.get("clustering_algorithm", exp.get("settings", {}).get("clustering_algorithm", common.get("clustering_algorithm")))
    )
    if configured != "default":
        return configured
    return METHOD_DEFAULT_ALGORITHM.get(method, "agglomerative")


def clustering_methods_for_algorithm(algorithm):
    if algorithm == "kmeans":
        return ["kmeans_exact_k", "kmeans_search"]
    if algorithm == "agglomerative":
        return ["agglomerative_exact_k", "agglomerative_search"]
    raise ValueError("clustering algorithm must be 'kmeans' or 'agglomerative'.")


def exact_k_from_config(common, dataset):
    value = common.get("exact_k", common.get("n_clusters"))
    if isinstance(value, dict):
        value = value.get(dataset, value.get("default"))
    return None if value is None else int(value)


def has_ground_truth_labels(df):
    if "File" not in df.columns or "Module" not in df.columns:
        return False
    labels = df["Module"].astype(str).str.strip().str.lower()
    labels = labels.replace({"": None, "nan": None, "none": None, "unmapped": None, "__none__": None})
    return df.loc[labels.notna(), "File"].nunique() == df["File"].nunique()


def require_ground_truth(dataset, df, evaluate):
    if evaluate and not has_ground_truth_labels(df):
        raise ValueError(
            f"{dataset} does not have complete GT labels. "
            "Use common.evaluate=false with common.exact_k/common.n_clusters to save File-to-cluster assignments without metrics."
        )


def dataset_k_range(common, dataset):
    overrides = common.get("k_range_overrides", {})
    values = overrides.get(dataset, {})
    k_min = int(values.get("k_min", common.get("k_min", 5)))
    k_max = int(values.get("k_max", common.get("k_max", 20)))
    return range(k_min, k_max + 1)


def format_search_range(k_range):
    values = list(k_range)
    return f"{int(min(values))}-{int(max(values))}"


def load_dataset(data_dir, dataset):
    node_file, dep_file = DATASETS[dataset]
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / node_file)
    deps = pd.read_csv(data_dir / dep_file)
    return normalize_df(df), normalize_df(deps)


def graph_profile(df, deps):
    num_files = int(df["File"].nunique()) if "File" in df.columns else int(len(df))
    if {"Source_File", "Target_File"}.issubset(deps.columns):
        num_edges = int(deps[["Source_File", "Target_File"]].drop_duplicates().shape[0])
    else:
        num_edges = int(len(deps))
    avg_degree = float(num_edges / num_files) if num_files else 0.0
    return num_files, num_edges, avg_degree


def choose_gdc_k_from_profile(dataset, df, deps):
    if dataset in GDC_K_BY_DATASET:
        return GDC_K_BY_DATASET[dataset]

    target_files, _, target_degree = graph_profile(df, deps)
    target_log_size = np.log2(max(target_files, 2))

    best_dataset = None
    best_distance = float("inf")
    for profile_dataset in GDC_K_PROFILES:
        if profile_dataset not in DATASETS:
            continue
        if CURRENT_DATA_DIR is None:
            raise ValueError("CURRENT_DATA_DIR is not set before GDC top-k selection.")
        profile_df, profile_deps = load_dataset(CURRENT_DATA_DIR, profile_dataset)
        profile_files, _, profile_degree = graph_profile(profile_df, profile_deps)
        profile_log_size = np.log2(max(profile_files, 2))
        distance = ((target_log_size - profile_log_size) / 4.0) ** 2 + ((target_degree - profile_degree) / 3.0) ** 2
        if distance < best_distance:
            best_dataset = profile_dataset
            best_distance = distance

    return GDC_K_PROFILES[best_dataset]


CURRENT_DATA_DIR = None


def resolve_dataset_settings(settings, method, dataset, df, deps):
    settings = dict(settings)
    if method == "homogeneous_gat" and settings.get("use_gdc", False):
        gdc_k = settings.get("gdc_k", "auto")
        if gdc_k is None or str(gdc_k).strip().lower() in {"", "auto", "best"}:
            settings["gdc_k"] = choose_gdc_k_from_profile(dataset, df, deps)
            files, edges, degree = graph_profile(df, deps)
            print(
                f"GAT_GDC | {dataset} | auto gdc_k={settings['gdc_k']} "
                f"(files={files}, edges={edges}, avg_degree={degree:.2f})"
            )
    return settings


def build_dataset_cache(df, deps, common=None):
    common = {} if common is None else dict(common)
    structural = HeterogeneousData(
        df,
        deps,
        folder_nodes=True,
        collapse_dependency_types=bool(common.get("collapse_hgat_dependencies", True)),
    )
    return {
        "structural": structural,
        "node2vec": Node2VecGraph(df, deps),
        "metapath": {},
        "minibatch": None,
        "lpe": {},
        "gdc": {},
    }


def cached_lpe_features(dataset_cache, method, settings):
    lpe_dim = settings.get("lpe_dim")
    if lpe_dim is None:
        return None
    default_undirected = method == "homogeneous_gat"
    key = (method, int(lpe_dim), bool(settings.get("lpe_is_undirected", default_undirected)))
    if key in dataset_cache["lpe"]:
        return dataset_cache["lpe"][key]

    graph = dataset_cache["structural"]
    if method == "homogeneous_gat":
        features = homo_lpe_input(graph, lpe_dim=key[1], is_undirected=key[2])
    elif method in {"heterogeneous_gat", "structural_gat"}:
        features = hetero_lpe_features(graph, key[1], is_undirected=key[2])
    else:
        features = None
    dataset_cache["lpe"][key] = features
    return features


def embedding_cache_key(method, dataset, run_id, settings, seed):
    params = {
        key: value
        for key, value in settings.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    params["_method"] = method
    params["_dataset"] = dataset
    params["_run_id"] = int(run_id)
    params["_seed"] = seed
    return json.dumps(params, sort_keys=True, default=str)


def embedding_disk_path(cache_dir, prefix, key):
    key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{slug(prefix)}_{key_hash}.npy"


def load_cached_embedding(cache_dir, memory_cache, prefix, key):
    if key in memory_cache:
        return memory_cache[key]
    if cache_dir is None:
        return None
    path = embedding_disk_path(cache_dir, prefix, key)
    if path.exists():
        memory_cache[key] = np.load(str(path))
        return memory_cache[key]
    return None


def save_cached_embedding(cache_dir, memory_cache, prefix, key, embedding):
    memory_cache[key] = embedding
    if cache_dir is None:
        return
    path = embedding_disk_path(cache_dir, prefix, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), embedding)


def maybe_cached_embedding(method, dataset, run_id, settings, seed, cache_dir, memory_cache, train_fn):
    prefix = settings.get("Method", method)
    key = embedding_cache_key(method, dataset, run_id, settings, seed)
    cached = load_cached_embedding(cache_dir, memory_cache, prefix, key)
    if cached is not None:
        return cached
    embedding = train_fn()
    save_cached_embedding(cache_dir, memory_cache, prefix, key, embedding)
    return embedding


def embed(method, df, deps, settings, dataset, run_id, dataset_cache, memory_cache, seed, cache_dir=None):
    if method == "negar":
        graph = dataset_cache["node2vec"]
        z = maybe_cached_embedding(
            method, dataset, run_id, settings, seed, cache_dir, memory_cache,
            lambda: train_node2vec_embeddings(graph, settings),
        )
        return z, graph.y_true, graph.num_classes, graph.node_list

    if method == "metapath2vec":
        if "graph" not in dataset_cache["metapath"]:
            dataset_cache["metapath"]["graph"] = build_metapath_graph(
                df,
                deps,
                base_graph=dataset_cache["structural"].base_graph,
            )
        graph = dataset_cache["metapath"]["graph"]
        z = maybe_cached_embedding(
            method, dataset, run_id, settings, seed, cache_dir, memory_cache,
            lambda: train_metapath2vec_embeddings(graph, settings),
        )
        return z, graph["y_true"], graph["num_classes"], graph["file_names"]

    if method == "homogeneous_gat":
        if settings.get("file_feature_init") == "negar":
            raise ValueError("GAT variants no longer support file_feature_init='negar'.")
        graph = dataset_cache["structural"]
        run_settings = dict(settings)
        use_big_graph_minibatch = (
            bool(run_settings.get("auto_minibatch", True))
            and int(graph["file"].num_nodes) > int(run_settings.get("minibatch_threshold_files", 5000))
        )
        if use_big_graph_minibatch:
            if dataset_cache["minibatch"] is None:
                dataset_cache["minibatch"] = HomogeneousFileGraphData(df, deps, dataset_name=dataset)
            mini_graph = dataset_cache["minibatch"]
            gdc_pos = None
            gdc_attr = None
            if bool(run_settings.get("use_gdc", False)):
                gdc_key = (
                    "approx_ppr",
                    float(run_settings.get("gdc_alpha", 0.15)),
                    float(run_settings.get("gdc_threshold", 1e-3)),
                    int(run_settings.get("gdc_k", 20)),
                )
                if gdc_key not in dataset_cache["gdc"]:
                    dataset_cache["gdc"][gdc_key] = gdc_edge_index_approx(
                        mini_graph,
                        alpha=gdc_key[1],
                        threshold=gdc_key[2],
                        topk=gdc_key[3],
                        use_cache=True,
                    )
                gdc_pos, gdc_attr = dataset_cache["gdc"][gdc_key]
            z = maybe_cached_embedding(
                method,
                dataset,
                run_id,
                run_settings,
                seed,
                cache_dir,
                memory_cache,
                lambda: train_homogeneous_gat_minibatch_embeddings(
                    mini_graph,
                    run_settings,
                    precomputed_pos=gdc_pos,
                    precomputed_attr=gdc_attr,
                    run_seed=seed,
                ),
            )
            return z, mini_graph.y.detach().cpu().numpy(), mini_graph.num_classes, mini_graph.file_names

        lpe = cached_lpe_features(dataset_cache, method, settings)
        if lpe is not None:
            run_settings["file_features"] = lpe
        gdc_pos = None
        gdc_attr = None
        if bool(run_settings.get("use_gdc", False)):
            gdc_key = (
                run_settings.get("gdc_method", "ppr"),
                float(run_settings.get("gdc_alpha", 0.15)),
                int(run_settings.get("gdc_k", 64)),
            )
            if gdc_key not in dataset_cache["gdc"]:
                dataset_cache["gdc"][gdc_key] = gdc_edge_index(
                    graph,
                    method=gdc_key[0],
                    alpha=gdc_key[1],
                    k=gdc_key[2],
                )
            gdc_pos, gdc_attr = dataset_cache["gdc"][gdc_key]
        z = maybe_cached_embedding(
            method, dataset, run_id, settings, seed, cache_dir, memory_cache,
            lambda: train_homogeneous_gat_embeddings(
                graph,
                run_settings,
                precomputed_pos=gdc_pos,
                precomputed_attr=gdc_attr,
            ),
        )
        y_true = graph["file"].y.detach().cpu().numpy() if hasattr(graph["file"], "y") else None
        return z, y_true, graph.num_classes, graph.file_names

    if method in {"heterogeneous_gat", "structural_gat"}:
        if settings.get("file_feature_init") == "negar":
            raise ValueError("GAT variants no longer support file_feature_init='negar'.")
        graph = dataset_cache["structural"]
        file_features = cached_lpe_features(dataset_cache, method, settings)
        z = maybe_cached_embedding(
            method, dataset, run_id, settings, seed, cache_dir, memory_cache,
            lambda: train_heterogeneous_gat_embeddings(graph, settings, file_features=file_features),
        )
        y_true = graph["file"].y.detach().cpu().numpy() if hasattr(graph["file"], "y") else None
        return z, y_true, graph.num_classes, graph.file_names

    raise ValueError(f"Unknown method: {method}")


def result_row(dataset, method_name, run_id, cluster_row, timing, k_range):
    row = {
        "Model": method_name,
        "Dataset": dataset,
        "run_id": int(run_id),
    }
    row.update(cluster_row)
    row["search_range"] = format_search_range(k_range)
    row.update(timing)
    return row


def run(config, only_experiments=None, only_datasets=None):
    global CURRENT_DATA_DIR
    common = config.get("common", {})
    data_dir = common.get("data_dir", "data/processed")
    CURRENT_DATA_DIR = data_dir
    out_dir = Path(common.get("out_dir", "Results"))
    datasets = selected_datasets(common)
    if only_datasets:
        wanted_datasets = set(only_datasets)
        unknown = sorted(wanted_datasets - set(DATASETS))
        if unknown:
            raise ValueError(f"Unknown dataset names: {unknown}. Available datasets: {sorted(DATASETS)}")
        datasets = [dataset for dataset in datasets if dataset in wanted_datasets]
        if not datasets:
            raise ValueError(f"No datasets matched --datasets {sorted(wanted_datasets)}")
    num_runs = int(common.get("num_runs", 1))
    flush_every = int(common.get("flush_every", 1))
    evaluate = bool(common.get("evaluate", True))
    random_state = common.get("random_state")
    use_embedding_cache = bool(common.get("use_embedding_cache", False))
    cache_dir_value = common.get("embedding_cache_dir")
    cache_dir = Path(cache_dir_value) if cache_dir_value else out_dir / "embedding_cache"
    cache_dir = cache_dir if use_embedding_cache else None
    memory_cache = {}

    experiments = list(config.get("experiments", []))
    if only_experiments:
        wanted = {slug(name).lower() for name in only_experiments}
        experiments = [
            exp for exp in experiments
            if slug(exp.get("name")).lower() in wanted
            or slug(exp.get("settings", {}).get("Method", "")).lower() in wanted
        ]
        if not experiments:
            raise ValueError(f"No experiments matched --only {sorted(wanted)}")

    dataset_store = {}
    for dataset in datasets:
        data_start = time.perf_counter()
        df, deps = load_dataset(data_dir, dataset)
        require_ground_truth(dataset, df, evaluate)
        dataset_store[dataset] = {
            "df": df,
            "deps": deps,
            "cache": build_dataset_cache(df, deps, common=common),
            "data_build_seconds": round(time.perf_counter() - data_start, 4),
        }

    for exp in experiments:
        method = exp["method"]
        base_settings = dict(exp.get("settings", {}))
        method_name = base_settings.get("Method", exp.get("name", method))
        exp_dir = experiment_dir(out_dir, exp)
        algorithm = clustering_algorithm_for_experiment(exp, method, common)
        clustering_methods = clustering_methods_for_algorithm(algorithm)
        rows_per_run = 1 if not evaluate else len(clustering_methods)

        for dataset in datasets:
            store = dataset_store[dataset]
            df = store["df"]
            deps = store["deps"]
            settings = deep_update(base_settings, exp.get("dataset_overrides", {}).get(dataset, {}))
            settings = resolve_dataset_settings(settings, method, dataset, df, deps)
            method_name = settings.get("Method", method_name)
            k_range = dataset_k_range(common, dataset)
            save_path = result_path(exp_dir, dataset) if evaluate else assignment_path(exp_dir, dataset)
            done = (
                completed_run_ids(save_path, dataset, method_name, rows_per_run=rows_per_run)
                if evaluate
                else completed_assignment_run_ids(save_path, dataset, method_name)
            )
            rows = []

            if done:
                print(f"{method_name} | {dataset} | skipped {len(done)} completed run(s)")

            for run_id in range(1, num_runs + 1):
                if run_id in done:
                    continue

                seed = None if random_state is None else int(random_state) + run_id - 1
                apply_random_state(seed)
                run_settings = dict(settings)
                run_settings["random_state"] = seed
                run_settings["minibatch_threshold_files"] = int(common.get("minibatch_threshold_files", 5000))
                run_settings["auto_minibatch"] = bool(common.get("auto_minibatch_gat", True))
                run_settings.setdefault("neighbor_batch_size", int(common.get("neighbor_batch_size", 10)))
                run_settings.setdefault("batch_size", int(common.get("batch_size", 1024)))
                run_settings.setdefault("inference_batch_size", int(common.get("inference_batch_size", run_settings["batch_size"])))

                pipeline_start = time.perf_counter()
                z, y_true, model_k, file_names = embed(
                    method,
                    df,
                    deps,
                    run_settings,
                    dataset=dataset,
                    run_id=run_id,
                    dataset_cache=store["cache"],
                    memory_cache=memory_cache,
                    seed=seed,
                    cache_dir=cache_dir,
                )
                exact_k = exact_k_from_config(common, dataset) or model_k
                if exact_k is None or int(exact_k) <= 0:
                    raise ValueError(
                        f"{dataset} needs common.exact_k or common.n_clusters before clustering can run."
                    )
                if evaluate and y_true is None:
                    raise ValueError(
                        f"{dataset} has no GT labels for {method_name}. "
                        "Set common.evaluate=false to save cluster assignments without evaluation metrics."
                    )

                clusterer = Clusterer(z, y_true, k_range=k_range, df_deps=deps, node_names=file_names)
                if not evaluate:
                    labels = clusterer.exact_k_labels(exact_k, algorithm=algorithm)
                    pipeline_seconds = round(float(store["data_build_seconds"]) + (time.perf_counter() - pipeline_start), 4)
                    assignments = [
                        {"file": str(file_name), "cluster": int(label)}
                        for file_name, label in zip(file_names, labels.astype(int))
                    ]
                    append_assignment_json(save_path, {
                        "Model": method_name,
                        "Dataset": dataset,
                        "run_id": int(run_id),
                        "algorithm": algorithm,
                        "clustering": "exact_k",
                        "n_clusters": int(exact_k),
                        "search_range": format_search_range(k_range),
                        "total_pipeline_seconds": pipeline_seconds,
                        "assignments": assignments,
                    })
                    print(f"{method_name} | {dataset} | run {run_id}/{num_runs} | {algorithm} exact_k")
                    continue

                cluster_rows = clusterer.run_all(exact_k=exact_k, methods=clustering_methods)
                data_build_seconds = float(store["data_build_seconds"])
                timing = {
                    "total_pipeline_seconds": round(data_build_seconds + (time.perf_counter() - pipeline_start), 4),
                }
                rows.extend(
                    result_row(dataset, method_name, run_id, cluster_row, timing, k_range)
                    for cluster_row in cluster_rows
                )

                print(f"{method_name} | {dataset} | run {run_id}/{num_runs} | {algorithm} exact_k+search")
                if len(rows) >= flush_every * rows_per_run:
                    append_csv(pd.DataFrame(rows), save_path)
                    rows.clear()

            append_csv(pd.DataFrame(rows), save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run final N2V, M2V, GAT, GAT_GDC, and HGAT experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "How to run:\n"
            "  cd C:\\Users\\JABEERAK\\Architecture_Recovery\\NAIM\\StructGAT\n"
            "  conda activate torchDL\n"
            "  python run_experiments.py --config experiment_config.json\n\n"
            "Run only selected methods:\n"
            "  python run_experiments.py --config experiment_config.json --only N2V\n"
            "  python run_experiments.py --config experiment_config.json --only GAT GAT_GDC HGAT\n\n"
            "Run only selected datasets:\n"
            "  python run_experiments.py --config experiment_config.json --datasets Bash Chrome\n\n"
            "Clustering is controlled in experiment_config.json:\n"
            "  common \"clustering_algorithm\": \"agglomerative\"\n"
            "  N2V experiment \"clustering_algorithm\": \"kmeans\"\n"
            "  With evaluation enabled, each selected algorithm writes exact_k + silhouette search rows.\n\n"
            "GAT minibatching:\n"
            "  GAT and GAT_GDC use minibatching automatically when file count exceeds common.minibatch_threshold_files.\n\n"
            "GAT_GDC gdc_k:\n"
            "  Known winners are set per dataset in experiment_config.json.\n"
            "  Use \"gdc_k\": \"auto\" for Chrome or future datasets; the runner picks by graph size and average degree.\n\n"
            "Outputs are saved under Results/N2V, Results/M2V, Results/GAT, Results/GAT_GDC, and Results/HGAT.\n"
            "With common.evaluate=false, outputs are JSON file-to-cluster assignments."
        ),
    )
    parser.add_argument("--config", default="experiment_config.json", help="Path to the experiment JSON config.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional method names to run, for example: --only N2V GAT_GDC HGAT",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to run, for example: --datasets Bash Chrome",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    base_dir = config_path.parent
    common = config.setdefault("common", {})
    for key in ("data_dir", "out_dir", "embedding_cache_dir"):
        if common.get(key) and not Path(common[key]).is_absolute():
            common[key] = str(base_dir / common[key])

    run(config, only_experiments=args.only, only_datasets=args.datasets)


if __name__ == "__main__":
    main()
