import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.utils import coalesce, negative_sampling, remove_self_loops, to_undirected


def file_edge_index(data, device=None):
    parts = [data[et].edge_index.to(device) for et in data.edge_types if et[0] == "file" and et[2] == "file"]
    edge_index = torch.cat(parts, dim=1)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=int(data["file"].num_nodes))
    edge_index, _ = coalesce(edge_index, None, int(data["file"].num_nodes), int(data["file"].num_nodes))
    return edge_index


def lpe_features(data, dim, is_undirected=False):
    graph = Data(edge_index=file_edge_index(data, "cpu"), num_nodes=int(data["file"].num_nodes))
    graph = AddLaplacianEigenvectorPE(
        k=int(dim),
        attr_name="x",
        is_undirected=bool(is_undirected),
    )(graph)
    return graph.x.float()


class HeterogeneousGAT(nn.Module):
    def __init__(self, data, config, file_features=None):
        super().__init__()
        self.config = config
        self.dropout = float(config.get("dropout", 0.00))
        self.variational = bool(config.get("variational", True))
        num_file_nodes = int(data["file"].num_nodes)

        # Static file features: external file_features or opt-in LPE.
        use_lpe = ("lpe_dim" in config) and (config.get("lpe_dim") is not None)
        if file_features is not None:
            self.file_x = nn.Parameter(torch.as_tensor(file_features, dtype=torch.float))
        elif use_lpe:
            self.file_x = nn.Parameter(
                lpe_features(
                    data,
                    int(config["lpe_dim"]),
                    is_undirected=bool(config.get("lpe_is_undirected", False)),
                )
            )
        else:
            self.file_x = None

        self.emb = nn.ModuleDict()
        file_emb_dim = int(config.get("file_emb_dim", config.get("embedding_dim", 128)))
        self.emb["file"] = nn.Embedding(num_file_nodes, file_emb_dim)
        nn.init.xavier_uniform_(self.emb["file"].weight)

        num_folders = int(data["folder"].num_nodes)
        folder_emb_dim = int(config.get("folder_emb_dim", 128))
        self.emb["folder"] = nn.Embedding(num_folders, folder_emb_dim)
        nn.init.xavier_uniform_(self.emb["folder"].weight)

        allowed = []
        for et in data.edge_types:
            if et[0] == "file" and et[2] == "file":
                allowed.append(et)
            elif "folder" in (et[0], et[2]):
                allowed.append(et)

        hidden = int(config.get("hidden_channels", 128))
        heads = int(config.get("heads", 2))
        self.convs = nn.ModuleList([
            HeteroConv({
                et: GATConv((-1, -1), hidden, heads=heads, concat=False, add_self_loops=False, dropout=self.dropout)
                for et in allowed
            }, aggr="sum")
            for _ in range(int(config.get("num_layers", 1)))
        ])
        if self.variational:
            self.lin_mu = nn.Linear(hidden, hidden)
            self.lin_logstd = nn.Linear(hidden, hidden)
        else:
            self.linear = nn.Linear(hidden, hidden)

    def _file_features(self, file_node_ids, device):
        feature_ids = file_node_ids.to(device)
        learnable = self.emb["file"](feature_ids)
        if self.file_x is None:
            return learnable
        return torch.cat([self.file_x[feature_ids], learnable], dim=-1)

    def _folder_features(self, folder_node_ids, device):
        folder_ids = folder_node_ids.to(device)
        return self.emb["folder"](folder_ids)

    def forward(self, edge_index_dict, node_id_dict=None):
        device = next(self.parameters()).device

        if node_id_dict is None:
            node_id_dict = {
                "file": torch.arange(self.emb["file"].num_embeddings, device=device),
                "folder": torch.arange(self.emb["folder"].num_embeddings, device=device),
            }

        file_feat = self._file_features(node_id_dict["file"], device)
        folder_feat = self._folder_features(node_id_dict["folder"], device)

        x = {
            "file": file_feat,
            "folder": folder_feat,
        }

        for conv in self.convs:
            x = conv(x, edge_index_dict)
            x = {k: F.dropout(F.elu(v), p=self.dropout, training=self.training) for k, v in x.items() if v is not None}
        h_file = x["file"]
        if self.variational:
            mu = self.lin_mu(h_file)
            logstd = self.lin_logstd(h_file).clamp(-10.0, 10.0)
            return mu, logstd
        return self.linear(h_file)


def kl_gaussian(mu, logstd):
    return 0.5 * torch.mean(torch.sum(torch.exp(2 * logstd) + mu**2 - 1 - 2 * logstd, dim=-1))


def hetero_neighbor_sizes(data, num_layers, fanout):
    return {edge_type: [int(fanout)] * int(num_layers) for edge_type in data.edge_types}


def train_heterogeneous_gat_minibatch_embeddings(
    data,
    config,
    file_features=None,
    batch_size=512,
    inference_batch_size=1024,
    neighbor_sizes=None,
    loader_workers=0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variational = bool(config.get("variational", True))
    verbose = bool(config.get("verbose", False))
    beta_kl = float(config.get("beta_kl", 0.2))
    neg_ratio = float(config.get("neg_ratio", 2.0))
    num_layers = int(config.get("num_layers", 1))
    fanout = int(config.get("neighbor_batch_size", 20))
    num_files = int(data["file"].num_nodes)

    if neighbor_sizes is None:
        neighbor_sizes = hetero_neighbor_sizes(data, num_layers, fanout)

    model = HeterogeneousGAT(data, config, file_features=file_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-3)))
    pin_memory = bool(torch.cuda.is_available())

    loader_kwargs = {
        "num_neighbors": neighbor_sizes,
        "batch_size": int(batch_size),
        "input_nodes": ("file", torch.arange(num_files)),
        "shuffle": True,
        "num_workers": int(loader_workers),
    }
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = True
    if pin_memory:
        loader_kwargs["pin_memory"] = True

    train_loader = NeighborLoader(data=data, **loader_kwargs)

    for _ in tqdm(range(int(config.get("epochs", 100))), desc="Epochs (hetero mini-batch)", disable=not verbose):
        model.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            pos = file_edge_index(batch, device)
            if pos.numel() == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            out = model(
                batch.edge_index_dict,
                node_id_dict={node_type: batch[node_type].n_id for node_type in batch.node_types},
            )
            if variational:
                mu, logstd = out
                z = mu + torch.randn_like(mu) * torch.exp(logstd)
            else:
                z = out

            neg = negative_sampling(
                pos,
                num_nodes=int(batch["file"].num_nodes),
                num_neg_samples=max(1, int(pos.size(1) * neg_ratio)),
                method="sparse",
            )
            edge_index = torch.cat([pos, neg], dim=1)
            edge_label = torch.cat(
                [
                    torch.ones(pos.size(1), device=device),
                    torch.zeros(neg.size(1), device=device),
                ]
            )
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
            recon = F.binary_cross_entropy_with_logits(logits, edge_label)
            if variational:
                loss = recon + beta_kl * (kl_gaussian(mu, logstd) / max(int(batch["file"].num_nodes), 1))
            else:
                loss = recon
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        edge_index_dict = {
            edge_type: data[edge_type].edge_index.to(device)
            for edge_type in data.edge_types
        }
        node_id_dict = {
            "file": torch.arange(num_files, device=device),
            "folder": torch.arange(int(data["folder"].num_nodes), device=device),
        }
        out = model(edge_index_dict, node_id_dict=node_id_dict)
        z = out[0] if variational else out
        embeddings = z.detach().cpu().numpy()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings

def train_heterogeneous_gat_embeddings(data, config, file_features=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = HeterogeneousGAT(data, config, file_features=file_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-3)))
    variational = bool(config.get("variational", True))
    beta_kl = float(config.get("beta_kl", 0.2))
    neg_ratio = float(config.get("neg_ratio", 2.0))
    pos = file_edge_index(data, device)
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
    n = int(data["file"].num_nodes)
    for _ in range(int(config.get("epochs", 100))):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(edge_index_dict)
        if variational:
            mu, logstd = out
            z = mu + torch.randn_like(mu) * torch.exp(logstd)
        else:
            z = out
        neg = negative_sampling(pos, num_nodes=n, num_neg_samples=int(pos.size(1) * neg_ratio), method="sparse")
        ei = torch.cat([pos, neg], dim=1)
        y = torch.cat([torch.ones(pos.size(1), device=device), torch.zeros(neg.size(1), device=device)])
        recon = F.binary_cross_entropy_with_logits((z[ei[0]] * z[ei[1]]).sum(dim=1), y)
        if variational:
            loss = recon + beta_kl * (kl_gaussian(mu, logstd) / max(n, 1))
        else:
            loss = recon
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        out = model(edge_index_dict)
        z = out[0] if variational else out
        return z.detach().cpu().numpy()


train_structural_gat_embeddings = train_heterogeneous_gat_embeddings
