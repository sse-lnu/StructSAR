from pathlib import Path
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "3")
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
    module="sklearn.cluster._kmeans",
)

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

from .evaluation import evaluate_clustering


DROP_RESULT_COLUMNS = {
    "dataset",
    "Method",
    "found_clusters",
    "clusters",
    "selected_k",
    "experiment",
    "model",
    "pipeline",
    "model_name",
    "random_state",
    "walk_length",
    "p",
    "q",
    "epochs",
    "dimensions",
    "embedding_dim",
    "num_walks",
    "walks_per_node",
    "window",
    "context_size",
    "negative",
    "num_negative_samples",
    "workers",
    "batch_size",
    "sparse",
    "lr",
    "lpe_dim",
    "lpe_is_undirected",
    "file_emb_dim",
    "folder_emb_dim",
    "hidden_channels",
    "out_channels",
    "num_layers",
    "heads",
    "variational",
    "beta_kl",
    "neg_ratio",
    "dropout",
    "use_gdc",
    "gdc_method",
    "gdc_alpha",
    "gdc_k",
    "file_feature_init",
    "data_build_seconds",
    "embedding_seconds",
    "clustering_seconds",
    "training_seconds",
    "inference_seconds",
    "gdc_seconds",
    "pipeline_seconds",
    "total_seconds",
}

RESULT_COLUMN_ORDER = [
    "Model",
    "Dataset",
    "run_id",
    "clustering",
    "algorithm",
    "n_clusters",
    "search_range",
    "mojofm",
    "a2a",
    "c2c_cvg_33",
    "c2c_cvg_50",
    "c2c_cvg_66",
    "c2c_cvg_80",
    "ari",
    "normalized_turbomq",
    "turbomq",
    "total_pipeline_seconds",
]


def clean_result_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=list(DROP_RESULT_COLUMNS), errors="ignore")
    ordered = [col for col in RESULT_COLUMN_ORDER if col in df.columns]
    extras = [col for col in df.columns if col not in ordered]
    return df.reindex(columns=ordered + extras)


def append_csv(df: pd.DataFrame, path: str | Path) -> None:
    if df.empty:
        return
    df = clean_result_frame(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        df.to_csv(path, index=False)
        return

    old = clean_result_frame(pd.read_csv(path))
    cols = list(old.columns)
    for col in df.columns:
        if col not in cols:
            cols.append(col)
    if list(old.columns) != cols:
        old.reindex(columns=cols).to_csv(path, index=False)
    df.reindex(columns=cols).to_csv(path, mode="a", header=False, index=False)


def clean_csv_schema(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    df = pd.read_csv(path)
    cleaned = clean_result_frame(df)
    if list(cleaned.columns) != list(df.columns):
        cleaned.to_csv(path, index=False)


class Clusterer:
    ALL_METHODS = ["kmeans_exact_k", "agglomerative_exact_k", "kmeans_search", "agglomerative_search"]

    def __init__(self, embeddings, y_true=None, k_range=range(5, 31), df_deps=None, node_names=None):
        self.x = np.asarray(embeddings, dtype=np.float32)
        self.y_true = None if y_true is None else np.asarray(y_true, dtype=int)
        self.k_range = list(k_range)
        self.df_deps = df_deps
        self.node_names = list(map(str, node_names)) if node_names is not None else None
        if self.node_names is not None and self.x.shape[0] != len(self.node_names):
            raise ValueError(f"embedding rows {self.x.shape[0]} != node_names length {len(self.node_names)}")

    def _metrics(self, labels):
        if self.y_true is None:
            raise ValueError("Ground-truth labels are required for evaluation metrics. Set evaluate=false to save file cluster assignments without GT labels.")
        return evaluate_clustering(
            self.y_true,
            labels,
            df_deps=self.df_deps,
            node_names=self.node_names,
        )

    def _silhouette_search_k(self, cluster_fn):
        sample = self.x[np.random.choice(self.x.shape[0], size=min(2000, self.x.shape[0]), replace=False)]
        best_k, best_score = self.k_range[0], -1.0
        for k in self.k_range:
            if int(k) >= sample.shape[0]:
                continue
            labels = cluster_fn(sample, int(k))
            if len(np.unique(labels)) <= 1:
                continue
            score = silhouette_score(sample, labels, metric="euclidean")
            if score > best_score:
                best_k, best_score = int(k), float(score)
        return int(best_k)

    def _kmeans_search_k(self):
        return self._silhouette_search_k(
            lambda sample, k: KMeans(n_clusters=k, n_init=10).fit_predict(sample)
        )

    def _agglomerative_search_k(self):
        return self._silhouette_search_k(
            lambda sample, k: AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(sample)
        )

    def kmean_exact_k(self, k):
        labels = KMeans(n_clusters=int(k), n_init=10).fit_predict(self.x)
        row = {"clustering": "exact_k", "algorithm": "kmeans", "n_clusters": int(k)}
        row.update(self._metrics(labels))
        return row

    def kmeans_exact_k(self, k):
        return self.kmean_exact_k(k)

    def exact_k(self, k):
        labels = AgglomerativeClustering(n_clusters=int(k), linkage="ward").fit_predict(self.x)
        row = {"clustering": "exact_k", "algorithm": "agglomerative", "n_clusters": int(k)}
        row.update(self._metrics(labels))
        return row

    def agglomerative_exact_k(self, k):
        return self.exact_k(k)

    def exact_k_labels(self, k, algorithm="agglomerative"):
        if algorithm == "kmeans":
            return KMeans(n_clusters=int(k), n_init=10).fit_predict(self.x)
        if algorithm == "agglomerative":
            return AgglomerativeClustering(n_clusters=int(k), linkage="ward").fit_predict(self.x)
        raise ValueError("algorithm must be 'kmeans' or 'agglomerative'")

    def kmean_exact_k_labels(self, k):
        return self.exact_k_labels(k, algorithm="kmeans")

    def kmeans_search(self):
        k = self._kmeans_search_k()
        labels = KMeans(n_clusters=int(k), n_init=10).fit_predict(self.x)
        row = {"clustering": "search", "algorithm": "kmeans", "n_clusters": int(k)}
        row.update(self._metrics(labels))
        return row

    def agglomerative_search(self):
        k = self._agglomerative_search_k()
        labels = AgglomerativeClustering(n_clusters=int(k), linkage="ward").fit_predict(self.x)
        row = {"clustering": "search", "algorithm": "agglomerative", "n_clusters": int(k)}
        row.update(self._metrics(labels))
        return row

    def run_all(self, exact_k, methods=None):
        methods = self.ALL_METHODS if methods is None else list(methods)
        rows = []
        for method in methods:
            if method in {"kmean_exact_k", "kmeans_exact_k"}:
                rows.append(self.kmean_exact_k(exact_k))
            elif method in {"exact_k", "agglomerative_exact_k"}:
                rows.append(self.exact_k(exact_k))
            elif method == "kmeans_search":
                rows.append(self.kmeans_search())
            elif method == "agglomerative_search":
                rows.append(self.agglomerative_search())
            else:
                raise ValueError(f"Unknown clustering method: {method}. Available: {self.ALL_METHODS}")
        return rows
