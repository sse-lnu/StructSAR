from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .homogeneous_graph import normalize_df


class Node2VecGraph:
    def __init__(self, df: pd.DataFrame, df_dep: pd.DataFrame, use_majority_vote: bool = True):
        self.df = normalize_df(df)
        self.df_dep = normalize_df(df_dep)
        self.use_majority_vote = bool(use_majority_vote)

        self.file_to_label: pd.Series = pd.Series(dtype=str)
        self.G: nx.Graph = nx.Graph()
        self.node_list: list[str] = []
        self.nodes: pd.Series = pd.Series(dtype=str)
        self.y_true: Optional[np.ndarray] = None
        self.label_encoder = None
        self.num_classes = 0

        self._build_data()

    def _build_data(self) -> None:
        if "File" not in self.df.columns:
            raise ValueError("df must contain a 'File' column.")

        d = self.df.copy()
        file_order = pd.Index(pd.unique(d["File"]), name="File")

        if "Module" in d.columns:
            d["Module"] = (
                d["Module"]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": None, "none": None, "": None, "unmapped": None, "__none__": None})
            )
        else:
            d["Module"] = None

        vote_df = d.loc[d["Module"].notna(), ["File", "Module"]].copy()
        nodes_df = pd.DataFrame({"File": file_order})
        if not vote_df.empty:
            if self.use_majority_vote:
                c = vote_df.groupby(["File", "Module"], sort=False).size().reset_index(name="cnt")
                c = c.sort_values(["File", "cnt", "Module"], ascending=[True, False, True])
                primary_df = c.drop_duplicates("File", keep="first")[["File", "Module"]]
            else:
                primary_df = vote_df.drop_duplicates("File", keep="first")[["File", "Module"]]

            primary_map = dict(zip(primary_df["File"], primary_df["Module"]))
            nodes_df["Primary_Module"] = nodes_df["File"].map(primary_map)
            if nodes_df["Primary_Module"].isna().any():
                nodes_df["Primary_Module"] = None
            nodes_df["Module"] = nodes_df["Primary_Module"]
        else:
            nodes_df["Primary_Module"] = None
            nodes_df["Module"] = None

        self.df = nodes_df.reset_index(drop=True)

        if self.df["Primary_Module"].notna().all():
            self.label_encoder = LabelEncoder()
            self.df["Label"] = self.label_encoder.fit_transform(self.df["Primary_Module"].astype(str))
            self.num_classes = len(self.label_encoder.classes_)
            self.file_to_label = self.df.set_index("File")["Primary_Module"]
        else:
            self.df["Label"] = None
            self.num_classes = 0
            self.file_to_label = pd.Series(dtype=str)

        keep_list = self.df["File"].astype(str).tolist()
        keep_set = set(keep_list)

        dep = normalize_df(self.df_dep)
        if dep is None or dep.empty:
            dep = pd.DataFrame(columns=["Source_File", "Target_File"])

        if "Source_File" not in dep.columns or "Target_File" not in dep.columns:
            raise ValueError("df_dep must contain 'Source_File' and 'Target_File' columns.")

        dep = dep[dep["Source_File"] != dep["Target_File"]]
        dep = dep.drop_duplicates(subset=["Source_File", "Target_File"]).reset_index(drop=True)
        dep = dep[dep["Source_File"].isin(keep_set) & dep["Target_File"].isin(keep_set)].reset_index(drop=True)

        self.df_dep = dep

        G = nx.Graph()
        G.add_nodes_from(keep_list)
        if not dep.empty:
            G.add_edges_from(dep[["Source_File", "Target_File"]].itertuples(index=False, name=None))

        self.G = G
        self.node_list = keep_list
        self.nodes = pd.Series(self.node_list, name="File")
        self.y_true = self.df["Label"].to_numpy(dtype=int) if self.num_classes else None
