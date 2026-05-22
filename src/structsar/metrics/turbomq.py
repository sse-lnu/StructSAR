from __future__ import annotations

from collections import defaultdict
from typing import Dict, Hashable, List, Mapping, Optional, Tuple, Union

import pandas as pd

NodeId = Hashable
ClusterId = Hashable


class TurboMQ:
    """
    Normalized TurboMQ matching the Java implementation you found:

      - dependency(a,b) is treated as an undirected pair weight between items a and b.
        If dependencies are directed in df_deps, we sum both directions into one undirected pair weight.

      - For each cluster i:
          u_i     = sum dependency(e1,e2) over unordered pairs inside cluster i
          exdep_i = sum over clusters j != i of interClusterDSM[i][j]
          CF_i    = u_i / (u_i + 0.5 * exdep_i)
        TurboMQ = sum_i CF_i
        normalized TurboMQ = TurboMQ / k

    Inputs:
      - labels: dict {node_name -> cluster_id} OR (node_names, predicted_labels) tuple
      - df_deps: dependency DataFrame with file-level edges and Dependency_Count
    """

    def __init__(
        self,
        df_deps: pd.DataFrame,
        labels: Union[Mapping[NodeId, ClusterId], Tuple[List[NodeId], List[ClusterId]]],
        *,
        source_col: str = "Source_File",
        target_col: str = "Target_File",
        weight_col: str = "Dependency_Count",
        ignore_self_loops: bool = True,
        normalized=True,
    ):
        self.labels_input = labels
        self.source_col = source_col
        self.target_col = target_col
        self.weight_col = weight_col
        self.ignore_self_loops = ignore_self_loops
        self.normalized = normalized
        self.df_deps = df_deps.groupby([source_col, target_col], as_index=False)[weight_col].sum()

    def _labels_dict(self) -> Dict[NodeId, ClusterId]:
        if isinstance(self.labels_input, Mapping):
            return dict(self.labels_input)

        node_names, y = self.labels_input
        if len(node_names) != len(y):
            raise ValueError(
                f"node_names and labels must have same length, got {len(node_names)} vs {len(y)}"
            )
        return {node_names[i]: y[i] for i in range(len(node_names))}

    def score(self) -> float:
        lab = self._labels_dict()
        clusters = set(lab.values())
        k = len(clusters)
        if k == 0:
            return 0.0
        deps = self.df_deps

        # Build undirected pair weights: dependency(a,b)
        # Key point: sum all directed edges between a and b into one undirected weight.
        pair_w: Dict[Tuple[NodeId, NodeId], float] = defaultdict(float)

        # Use fast column lists
        src = deps[self.source_col].tolist()
        dst = deps[self.target_col].tolist()
        if self.weight_col in deps.columns:
            wts = deps[self.weight_col].fillna(1.0).astype(float).tolist()
        else:
            wts = [1.0] * len(deps)

        for u, v, w in zip(src, dst, wts):
            if self.ignore_self_loops and u == v:
                continue
            if u not in lab or v not in lab:
                continue

            a, b = (u, v) if str(u) <= str(v) else (v, u)
            pair_w[(a, b)] += float(w)

        # Aggregate intra and inter-touch per cluster
        intra_u: Dict[ClusterId, float] = defaultdict(float)     # μ_i
        inter_touch: Dict[ClusterId, float] = defaultdict(float) # exdep_i (touching i)

        for (a, b), w in pair_w.items():
            ca = lab[a]
            cb = lab[b]
            if ca == cb:
                intra_u[ca] += w
            else:
                inter_touch[ca] += w
                inter_touch[cb] += w

        # Compute sum(CF_i) and normalize by k
        total_cf = 0.0
        for c in clusters:
            u = intra_u.get(c, 0.0)
            exdep = inter_touch.get(c, 0.0)
            denom = u + 0.5 * exdep
            cf = (u / denom) if denom > 0 else 0.0
            if cf > 0:
                total_cf += cf
        if self.normalized:
            return (total_cf / k)*100 if k > 0 else 0.0
        return total_cf