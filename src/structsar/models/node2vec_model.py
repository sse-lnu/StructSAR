from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from node2vec import Node2Vec


@dataclass
class Node2VecModel:
    dimensions: int = 128
    walk_length: int = 30
    num_walks: int = 100
    window: int = 15
    epochs: int = 5
    p: float = 1.0
    q: float = 1.0
    workers: int = 4
    negative: int = 5

    def fit_transform(self, data) -> np.ndarray:
        n2v = Node2Vec(
            data.G,
            dimensions=int(self.dimensions),
            walk_length=int(self.walk_length),
            num_walks=int(self.num_walks),
            p=float(self.p),
            q=float(self.q),
            workers=int(self.workers),
        )

        model = n2v.fit(
            window=int(self.window),
            min_count=1,
            sg=1,
            epochs=int(self.epochs),
            negative=int(self.negative),
            hs=0,
            workers=int(self.workers),
        )

        return np.vstack([model.wv[n] for n in data.node_list]).astype(np.float32)


def train_node2vec_embeddings(data, config):
    model = Node2VecModel(
        dimensions=int(config.get("dimensions", config.get("embedding_dim", 128))),
        walk_length=int(config.get("walk_length", 30)),
        num_walks=int(config.get("num_walks", config.get("walks_per_node", 100))),
        window=int(config.get("window", config.get("context_size", 15))),
        epochs=int(config.get("epochs", 5)),
        p=float(config.get("p", 1.0)),
        q=float(config.get("q", 1.0)),
        workers=int(config.get("workers", 4)),
        negative=int(config.get("negative", config.get("num_negative_samples", 5))),
    )
    return model.fit_transform(data)
