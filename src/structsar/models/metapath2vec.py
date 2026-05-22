import numpy as np
import torch
from torch_geometric.nn.models import MetaPath2Vec

from ..data_builders import build_metapath_graph


def simulate_metapath_walks(graph, metapath=None, walk_length=30, walks_per_node=30, random_state=None):
    rng = np.random.default_rng(random_state)
    metapath = metapath or ["file", "folder", "file", "file"]
    starts = np.arange(graph["num_nodes"])
    walks = []

    for _ in range(int(walks_per_node)):
        rng.shuffle(starts)
        for start in starts:
            walk = [int(start)]
            current = ("file", int(start))
            step = 0
            while len(walk) < int(walk_length):
                next_type = metapath[(step + 1) % len(metapath)]
                candidates = [n for n in graph["hetero_adjacency"].get(current, []) if n[0] == next_type]
                if not candidates:
                    break
                current = candidates[int(rng.integers(0, len(candidates)))]
                walk.append(int(current[1]))
                step += 1
            walks.append(walk)
    return walks


def train_metapath2vec_embeddings(graph, config):
    if config.get("training_backend", "pyg") == "pyg":
        return train_pyg_metapath2vec_embeddings(graph, config)

    raise ValueError("Metapath2Vec supports training_backend='pyg' in this project.")


def train_pyg_metapath2vec_embeddings(graph, config):
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    metapath = config.get("metapath_edges") or [
        ("file", "in_folder", "folder"),
        ("folder", "contains", "file"),
        ("file", "depends_on", "file"),
    ]
    metapath = [tuple(edge) for edge in metapath]
    edge_index_dict = {k: v for k, v in graph["edge_index_dict"].items() if v.numel() > 0}
    model = MetaPath2Vec(
        edge_index_dict=edge_index_dict,
        embedding_dim=int(config.get("embedding_dim", 128)),
        metapath=metapath,
        walk_length=int(config.get("walk_length", 30)),
        context_size=int(config.get("context_size", 10)),
        walks_per_node=int(config.get("walks_per_node", 100)),
        num_negative_samples=int(config.get("num_negative_samples", 5)),
        num_nodes_dict=graph["num_nodes_dict"],
        sparse=bool(config.get("sparse", True)),
    ).to(device)

    loader = model.loader(
        batch_size=int(config.get("batch_size", 32)),
        shuffle=bool(config.get("shuffle", False)),
        num_workers=int(config.get("num_workers", 0)),
    )
    lr = float(config.get("lr", 0.025))
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr) if bool(config.get("sparse", True)) else torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(int(config.get("epochs", 5))):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

    return model("file").detach().cpu().numpy()
