from collections import defaultdict

import pandas as pd
import torch


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    for col in out.columns:
        if "file" in str(col).strip().lower():
            out[col] = out[col].astype(str).str.strip().str.replace("\\", "/", regex=False)
    if "Module" in out.columns:
        out["Module"] = out["Module"].astype(str).str.strip()
    if "Dependency_Type" in out.columns:
        out["Dependency_Type"] = out["Dependency_Type"].astype(str).str.strip()
    return out


def build_file_graph(df_nodes: pd.DataFrame, df_deps: pd.DataFrame):
    nodes = normalize_df(df_nodes)

    if "File" not in nodes.columns:
        raise ValueError("Node data must contain a 'File' column.")

    unique_nodes = nodes.drop_duplicates("File", keep="first").copy()

    file_names = unique_nodes["File"].tolist()
    node_to_id = {name: i for i, name in enumerate(file_names)}

    modules = []
    y_true = None
    if "Module" in unique_nodes.columns:
        labels = unique_nodes["Module"].astype(str).str.strip()
        labels = labels.replace({"": None, "nan": None, "none": None, "unmapped": None, "__none__": None})
        if labels.notna().all():
            modules = sorted(labels.unique().tolist())
            module_to_id = {module: i for i, module in enumerate(modules)}
            y_true = labels.map(module_to_id).astype(int).to_numpy()

    deps = normalize_df(df_deps)
    deps = deps.drop_duplicates(["Source_File", "Target_File", "Dependency_Type"]).reset_index(drop=True)
    deps = deps[deps["Source_File"].isin(node_to_id) & deps["Target_File"].isin(node_to_id)]

    adjacency = defaultdict(list)
    for _, row in deps.iterrows():
        s = node_to_id[row["Source_File"]]
        t = node_to_id[row["Target_File"]]
        if s != t:
            adjacency[s].append(t)
            adjacency[t].append(s)

    adjacency = {k: sorted(set(v)) for k, v in adjacency.items()}
    return {
        "node_to_id": node_to_id,
        "file_names": file_names,
        "y_true": y_true,
        "modules": modules,
        "adjacency": adjacency,
        "num_nodes": len(file_names),
        "num_classes": len(modules),
    }


def edge_index_from_adjacency(adjacency, num_nodes):
    src, dst = [], []
    for u in range(int(num_nodes)):
        for v in adjacency.get(u, []):
            src.append(int(u))
            dst.append(int(v))
    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)
