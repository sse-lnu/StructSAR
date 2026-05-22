from collections import Counter, defaultdict

import torch

from .homogeneous_graph import build_file_graph


M2V_ROOT_FOLDER = "__m2v_root__"


def build_metapath_graph(df_nodes, df_deps, base_graph=None):
    graph = base_graph or build_file_graph(df_nodes, df_deps)
    file_names = graph["file_names"]
    file_id = graph["node_to_id"]
    num_files = graph["num_nodes"]

    folder_segs = {}
    for f in file_names:
        segs = [p.lower() for p in str(f).replace("\\", "/").split("/")[:-1] if p]
        folder_segs[f] = segs
    counts = Counter(tok for segs in folder_segs.values() for tok in segs)
    common = {tok for tok, c in counts.items() if c == len(file_names)}
    remove = common | {"src", "main", "java"}

    folders = set()
    deepest = {}
    for f, segs in folder_segs.items():
        toks = [s for s in segs if s not in remove]
        levels = []
        for i in range(1, len(toks) + 1):
            node = "/".join(toks[:i])
            folders.add(node)
            levels.append(node)
        deepest[f] = levels[-1] if levels else M2V_ROOT_FOLDER
    if any(folder == M2V_ROOT_FOLDER for folder in deepest.values()):
        folders.add(M2V_ROOT_FOLDER)
    folder_local_id = {name: i for i, name in enumerate(sorted(folders))}
    folder_id = {name: num_files + i for name, i in folder_local_id.items()}
    file_to_folder = {file_id[f]: folder_local_id[d] for f, d in deepest.items() if d in folder_local_id}

    adjacency = defaultdict(list)
    for s, neigh in graph["adjacency"].items():
        adjacency[("file", s)].extend(("file", t) for t in neigh)
    for f, folder in file_to_folder.items():
        adjacency[("file", f)].append(("folder", folder + num_files))
        adjacency[("folder", folder + num_files)].append(("file", f))

    ff_src, ff_dst = [], []
    for s, neigh in graph["adjacency"].items():
        for t in neigh:
            ff_src.append(int(s))
            ff_dst.append(int(t))
    file_file = torch.tensor([ff_src, ff_dst], dtype=torch.long) if ff_src else torch.empty((2, 0), dtype=torch.long)

    f_src, folder_dst = [], []
    for file_idx, folder_idx in file_to_folder.items():
        f_src.append(int(file_idx))
        folder_dst.append(int(folder_idx))
    file_folder = torch.tensor([f_src, folder_dst], dtype=torch.long) if f_src else torch.empty((2, 0), dtype=torch.long)
    folder_file = torch.stack([file_folder[1], file_folder[0]], dim=0) if file_folder.numel() else torch.empty((2, 0), dtype=torch.long)

    return {
        **graph,
        "folder_id": folder_id,
        "folder_local_id": folder_local_id,
        "file_to_folder": file_to_folder,
        "hetero_adjacency": {k: list(dict.fromkeys(v)) for k, v in adjacency.items()},
        "total_nodes": num_files + len(folder_id),
        "edge_index_dict": {
            ("file", "depends_on", "file"): file_file,
            ("file", "in_folder", "folder"): file_folder,
            ("folder", "contains", "file"): folder_file,
        },
        "num_nodes_dict": {"file": num_files, "folder": len(folder_id)},
    }
