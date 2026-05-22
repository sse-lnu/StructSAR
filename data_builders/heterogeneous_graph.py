from collections import Counter, defaultdict
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .homogeneous_graph import build_file_graph, edge_index_from_adjacency, normalize_df


class HeterogeneousData(HeteroData):
    def __init__(self, df_nodes, df_deps, base_graph=None, folder_nodes=True, collapse_dependency_types=True):
        super().__init__()
        self.df_dep = normalize_df(df_deps)
        self.folder_nodes = bool(folder_nodes)
        self.collapse_dependency_types = bool(collapse_dependency_types)
        base = base_graph or build_file_graph(normalize_df(df_nodes), self.df_dep)
        self.file_names = base["file_names"]
        self.file_id = base["node_to_id"]
        self["file"].num_nodes = base["num_nodes"]
        if base["y_true"] is not None:
            self["file"].y = torch.tensor(base["y_true"], dtype=torch.long)
        self.num_classes = base["num_classes"]
        self.base_graph = base
        self.folder_id = {}
        self.folder_names = []
        if self.folder_nodes:
            self._add_folder_nodes()
        self._add_file_edges(self.df_dep)
        if self.folder_nodes:
            self._add_folder_edges()

    def _add_folder_nodes(self):
        folder_segs = {f: [p.lower() for p in str(f).split("/")[:-1] if p] for f in self.file_names}
        counts = Counter(tok for segs in folder_segs.values() for tok in segs)
        common = {tok for tok, c in counts.items() if c == len(self.file_names)}
        remove = common | {"src", "main", "java"}
        deepest = {}
        folders = set()
        for f, segs in folder_segs.items():
            toks = [s for s in segs if s not in remove]
            levels = []
            for i in range(1, len(toks) + 1):
                folder = "/".join(toks[:i])
                folders.add(folder)
                levels.append(folder)
            deepest[f] = levels[-1] if levels else None
        self.folder_names = sorted(folders)
        self.folder_id = {name: i for i, name in enumerate(self.folder_names)}
        self["folder"].num_nodes = len(self.folder_id)

        src, dst = [], []
        for f, folder in deepest.items():
            if folder in self.folder_id:
                src.append(self.folder_id[folder])
                dst.append(self.file_id[f])
        self["folder", "contains", "file"].edge_index = (
            torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
        )

    def _add_folder_edges(self):
        src, dst = [], []
        for folder in self.folder_id:
            parts = folder.split("/")
            if len(parts) <= 1:
                continue
            parent = "/".join(parts[:-1])
            if parent in self.folder_id:
                src.append(self.folder_id[parent])
                dst.append(self.folder_id[folder])
        self["folder", "parent_of", "folder"].edge_index = (
            torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
        )

    def _add_file_edges(self, df_deps):
        deps = normalize_df(df_deps)
        deps = deps[deps["Source_File"].isin(self.file_id) & deps["Target_File"].isin(self.file_id)]

        if self.collapse_dependency_types:
            grouped = deps.groupby(["Source_File", "Target_File"], as_index=False).size()
            grouped["Dependency_Type"] = "depends_on"
        else:
            grouped = deps.groupby(["Source_File", "Target_File", "Dependency_Type"], as_index=False).size()

        for rel, rel_group in grouped.groupby("Dependency_Type", sort=False):
            srcs = rel_group["Source_File"].map(self.file_id).to_numpy(dtype="int64", copy=False)
            dsts = rel_group["Target_File"].map(self.file_id).to_numpy(dtype="int64", copy=False)
            self["file", rel, "file"].edge_index = torch.tensor([srcs, dsts], dtype=torch.long)


StructuralGraph = HeterogeneousData
