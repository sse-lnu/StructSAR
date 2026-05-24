import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GATConv
import torch_geometric.transforms as T
from torch_geometric.utils import coalesce, negative_sampling, remove_self_loops, to_undirected


LPE_DIM = 32
NUM_LAYERS = 2
DROPOUT = 0.0
CONCAT = False
ADD_SELF_LOOPS = True
APPROX_GDC_EPS = 1e-4
DEFAULT_GDC_THRESHOLD = 1e-3


def normalize_file_df(df):
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


class HomogeneousFileGraphData(Data):
    def __init__(self, df_nodes, df_deps, dataset_name="dataset"):
        super().__init__()
        nodes = normalize_file_df(df_nodes)
        unique_nodes = nodes.drop_duplicates("File", keep="first")[["File", "Module"]].copy()
        self.dataset_name = str(dataset_name)
        self.file_names = unique_nodes["File"].tolist()
        self.file_id = {name: i for i, name in enumerate(self.file_names)}
        self.num_nodes = len(self.file_names)

        modules = sorted(unique_nodes["Module"].unique().tolist())
        module_to_id = {module: i for i, module in enumerate(modules)}
        self.y = torch.tensor(unique_nodes["Module"].map(module_to_id).astype(int).to_numpy(), dtype=torch.long)
        self.num_classes = len(modules)

        deps = normalize_file_df(df_deps)
        deps = deps.drop_duplicates(["Source_File", "Target_File", "Dependency_Type"]).reset_index(drop=True)
        deps = deps[deps["Source_File"].isin(self.file_id) & deps["Target_File"].isin(self.file_id)]
        self.df_dep = deps.copy()
        deps = deps[deps["Source_File"] != deps["Target_File"]]
        grouped = deps.groupby(["Source_File", "Target_File"], as_index=False).size()
        srcs = grouped["Source_File"].map(self.file_id).to_numpy(dtype="int64", copy=False)
        dsts = grouped["Target_File"].map(self.file_id).to_numpy(dtype="int64", copy=False)
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        self.edge_index, _ = coalesce(edge_index, None, self.num_nodes)


MINIBATCH_UNDIRECTED_CACHE = {}
MINIBATCH_LPE_CACHE = {}
MINIBATCH_GDC_CACHE = {}


def homogeneous_edge_index(data, device=None):
    deps = data.df_dep.copy()
    deps["Source_File"] = deps["Source_File"].astype(str).str.strip()
    deps["Target_File"] = deps["Target_File"].astype(str).str.strip()
    deps = deps[
        deps["Source_File"].isin(data.file_id)
        & deps["Target_File"].isin(data.file_id)
    ].copy()
    deps = deps.groupby(["Source_File", "Target_File"], as_index=False)["Dependency_Count"].sum()
    deps = deps[deps["Source_File"] != deps["Target_File"]]
    src = deps["Source_File"].map(data.file_id)
    dst = deps["Target_File"].map(data.file_id)
    valid = src.notna() & dst.notna()
    edge_index = torch.stack(
        [
            torch.tensor(src[valid].values, dtype=torch.long),
            torch.tensor(dst[valid].values, dtype=torch.long),
        ],
        dim=0,
    )
    edge_index, _ = coalesce(
        edge_index,
        None,
        num_nodes=int(data["file"].num_nodes),
        reduce="sum",
    )
    if device is not None:
        edge_index = edge_index.to(device)
    return edge_index


def undirected_homogeneous_edge_index(data, device=None):
    edge_index = homogeneous_edge_index(data, device=device)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=int(data["file"].num_nodes))
    edge_index, _ = coalesce(edge_index, None, int(data["file"].num_nodes), int(data["file"].num_nodes))
    return edge_index


def lpe_input(data, lpe_dim=LPE_DIM, is_undirected=True):
    num_nodes = int(data["file"].num_nodes)
    edge_index = undirected_homogeneous_edge_index(data, device="cpu")
    graph = Data(edge_index=edge_index, num_nodes=num_nodes)
    graph = T.AddLaplacianEigenvectorPE(
        k=int(lpe_dim),
        attr_name="lap_pe",
        is_undirected=bool(is_undirected),
    )(graph)
    x = torch.ones((num_nodes, 1), dtype=torch.float)
    if hasattr(graph, "lap_pe") and graph.lap_pe is not None:
        x = torch.cat([x, graph.lap_pe.float()], dim=-1)
    return x.float()


def gdc_edge_index(data, method="ppr", alpha=0.15, k=64, device=None):
    num_nodes = int(data["file"].num_nodes)
    edge_index = undirected_homogeneous_edge_index(data, device="cpu")
    graph = Data(edge_index=edge_index, num_nodes=num_nodes)
    sparsification = {"method": "topk", "k": int(k), "dim": 0}

    if method == "ppr":
        transform = T.GDC(
            self_loop_weight=1,
            normalization_in="sym",
            normalization_out="col",
            diffusion_kwargs={"method": "ppr", "alpha": float(alpha)},
            sparsification_kwargs=sparsification,
            exact=True,
        )
    elif method == "heat":
        transform = T.GDC(
            self_loop_weight=1,
            normalization_in="sym",
            normalization_out="col",
            diffusion_kwargs={"method": "heat", "t": float(alpha)},
            sparsification_kwargs=sparsification,
            exact=True,
        )
    else:
        raise ValueError(f"Unknown GDC method: {method!r}. Use 'ppr' or 'heat'.")

    graph = transform(graph)
    edge_attr = graph.edge_attr
    if edge_attr is None:
        edge_attr = torch.ones(graph.edge_index.size(1), dtype=torch.float)
    if device is not None:
        return graph.edge_index.to(device), edge_attr.to(device)
    return graph.edge_index, edge_attr


def minibatch_dataset_key(data):
    return getattr(data, "dataset_name", "dataset")


def minibatch_undirected_edge_index(data):
    cache_key = minibatch_dataset_key(data)
    if cache_key in MINIBATCH_UNDIRECTED_CACHE:
        return MINIBATCH_UNDIRECTED_CACHE[cache_key]
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
    edge_index, _ = coalesce(edge_index, None, data.num_nodes)
    MINIBATCH_UNDIRECTED_CACHE[cache_key] = edge_index
    return edge_index


def minibatch_lpe_input(data, lpe_dim=16, is_undirected=True):
    cache_key = (minibatch_dataset_key(data), int(lpe_dim), bool(is_undirected))
    if cache_key in MINIBATCH_LPE_CACHE:
        return MINIBATCH_LPE_CACHE[cache_key]
    edge_index = minibatch_undirected_edge_index(data)
    graph = Data(edge_index=edge_index, num_nodes=data.num_nodes)
    graph = T.AddLaplacianEigenvectorPE(
        k=int(lpe_dim),
        attr_name="lap_pe",
        is_undirected=bool(is_undirected),
    )(graph)
    x = torch.ones((data.num_nodes, 1), dtype=torch.float)
    if hasattr(graph, "lap_pe") and graph.lap_pe is not None:
        x = torch.cat([x, graph.lap_pe.float()], dim=-1)
    MINIBATCH_LPE_CACHE[cache_key] = x.float()
    return MINIBATCH_LPE_CACHE[cache_key]


def gdc_edge_index_approx(data, alpha=0.15, threshold=None, topk=None, device=None, use_cache=True):
    threshold = float(DEFAULT_GDC_THRESHOLD if threshold is None else threshold)
    cache_key = (minibatch_dataset_key(data), float(alpha), float(threshold), None if topk is None else int(topk))
    if use_cache and cache_key in MINIBATCH_GDC_CACHE:
        out_ei, out_attr = MINIBATCH_GDC_CACHE[cache_key]
        if device is not None:
            return out_ei.to(device), out_attr.to(device)
        return out_ei, out_attr

    edge_index = minibatch_undirected_edge_index(data).cpu()
    graph = Data(edge_index=edge_index, num_nodes=int(data.num_nodes))
    graph = T.GDC(
        self_loop_weight=1,
        normalization_in="sym",
        normalization_out="col",
        diffusion_kwargs=dict(method="ppr", alpha=float(alpha), eps=float(APPROX_GDC_EPS)),
        sparsification_kwargs=dict(method="threshold", eps=float(threshold)),
        exact=False,
    )(graph)

    out_ei = graph.edge_index.cpu()
    out_attr = graph.edge_attr
    if out_attr is None:
        out_attr = torch.ones(out_ei.size(1), dtype=torch.float32)
    else:
        out_attr = out_attr.view(-1).cpu().float()

    if topk is not None:
        src = out_ei[0]
        keep_parts = []
        for node in range(int(data.num_nodes)):
            idx = (src == node).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            if idx.numel() > int(topk):
                idx = idx[torch.topk(out_attr[idx], k=int(topk), largest=True).indices]
            keep_parts.append(idx)
        if keep_parts:
            keep = torch.cat(keep_parts)
            out_ei = out_ei[:, keep]
            out_attr = out_attr[keep]
        else:
            out_ei = torch.empty((2, 0), dtype=torch.long)
            out_attr = torch.empty((0,), dtype=torch.float32)

    out_ei, out_attr = coalesce(out_ei, out_attr, int(data.num_nodes), int(data.num_nodes))
    MINIBATCH_GDC_CACHE[cache_key] = (out_ei, out_attr)
    if device is not None:
        return out_ei.to(device), out_attr.to(device)
    return out_ei, out_attr


class HomogeneousGAT(nn.Module):
    def __init__(self, data, config):
        super().__init__()
        self.variational = bool(config.get("variational", True))
        self.dropout = float(config.get("dropout", DROPOUT))
        num_nodes = int(data["file"].num_nodes)

        in_channels = 0
        file_x = None
        if config.get("file_features") is not None:
            file_x = torch.as_tensor(config["file_features"], dtype=torch.float)
        elif getattr(data["file"], "x", None) is not None:
            file_x = data["file"].x.float()
        elif ("lpe_dim" in config) and (config.get("lpe_dim") is not None):
            file_x = lpe_input(
                data,
                lpe_dim=int(config["lpe_dim"]),
                is_undirected=bool(config.get("lpe_is_undirected", True)),
            )

        if file_x is not None:
            self.register_buffer("x", file_x)
            in_channels += int(file_x.size(1))
        else:
            self.x = None

        file_emb_dim = int(config.get("file_emb_dim", config.get("embedding_dim", 128)))
        self.node_embedding = nn.Embedding(num_nodes, file_emb_dim)
        nn.init.xavier_uniform_(self.node_embedding.weight)
        in_channels += file_emb_dim

        hidden = int(config.get("hidden_channels", 256))
        heads = int(config.get("heads", 2))
        num_layers = int(config.get("num_layers", NUM_LAYERS))
        self.use_gdc = bool(config.get("use_gdc", False))
        edge_dim = 1 if self.use_gdc else None
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    in_channels,
                    hidden,
                    heads=heads,
                    concat=CONCAT,
                    dropout=self.dropout,
                    add_self_loops=ADD_SELF_LOOPS,
                    edge_dim=edge_dim,
                )
            )
            in_channels = hidden

        out_channels = int(config.get("out_channels", hidden))
        if self.variational:
            self.lin_mu = nn.Linear(in_channels, out_channels)
            self.lin_logstd = nn.Linear(in_channels, out_channels)
        else:
            self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, edge_index, edge_attr=None):
        x = self.x
        if self.node_embedding is not None:
            num_nodes = x.size(0) if x is not None else self.node_embedding.num_embeddings
            node_ids = torch.arange(num_nodes, device=edge_index.device)
            emb = self.node_embedding(node_ids)
            x = emb if x is None else torch.cat([x, emb], dim=-1)
        edge_attr = edge_attr.unsqueeze(-1) if self.use_gdc and edge_attr is not None else None
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.variational:
            return self.lin_mu(x), self.lin_logstd(x).clamp(-10.0, 10.0)
        return self.linear(x)


def kl_gaussian(mu, logstd):
    return 0.5 * torch.mean(torch.sum(torch.exp(2 * logstd) + mu**2 - 1 - 2 * logstd, dim=-1))


def train_homogeneous_gat_embeddings(data, config, precomputed_pos=None, precomputed_attr=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = HomogeneousGAT(data, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-3)))
    variational = bool(config.get("variational", True))
    beta_kl = float(config.get("beta_kl", 0.2))
    neg_ratio = float(config.get("neg_ratio", 2.0))
    orig = undirected_homogeneous_edge_index(data, device=device)
    pos = precomputed_pos.to(device) if precomputed_pos is not None else orig
    pos_attr = precomputed_attr.to(device) if precomputed_attr is not None else None
    n = int(data["file"].num_nodes)

    for _ in range(int(config.get("epochs", 100))):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(pos, edge_attr=pos_attr)
        if variational:
            mu, logstd = out
            z = mu + torch.randn_like(mu) * torch.exp(logstd)
        else:
            z = out
        neg = negative_sampling(orig, num_nodes=n, num_neg_samples=int(orig.size(1) * neg_ratio), method="sparse")
        ei = torch.cat([orig, neg], dim=1)
        y = torch.cat([torch.ones(orig.size(1), device=device), torch.zeros(neg.size(1), device=device)])
        recon = F.binary_cross_entropy_with_logits((z[ei[0]] * z[ei[1]]).sum(dim=1), y)
        loss = recon + beta_kl * (kl_gaussian(mu, logstd) / max(n, 1)) if variational else recon
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(pos, edge_attr=pos_attr)
        z = out[0] if variational else out
        return z.detach().cpu().numpy()


class MiniBatchGAT(nn.Module):
    def __init__(self, data, config):
        super().__init__()
        self.variational = bool(config.get("variational", True))
        self.dropout = float(config.get("dropout", DROPOUT))
        self.use_gdc = bool(config.get("use_gdc", False))
        num_nodes = int(data.num_nodes)

        in_channels = 0
        file_x = None
        if config.get("lpe_dim") is not None:
            file_x = minibatch_lpe_input(
                data,
                lpe_dim=int(config["lpe_dim"]),
                is_undirected=bool(config.get("lpe_is_undirected", True)),
            )
        if file_x is not None:
            self.register_buffer("x", file_x)
            in_channels += int(file_x.size(1))
        else:
            self.x = None

        file_emb_dim = int(config.get("file_emb_dim", config.get("embedding_dim", 128)))
        self.node_embedding = nn.Embedding(num_nodes, file_emb_dim)
        nn.init.xavier_uniform_(self.node_embedding.weight)
        in_channels += file_emb_dim

        hidden = int(config.get("hidden_channels", 256))
        heads = int(config.get("heads", 2))
        edge_dim = 1 if self.use_gdc else None
        self.convs = nn.ModuleList()
        for _ in range(int(config.get("num_layers", NUM_LAYERS))):
            self.convs.append(
                GATConv(
                    in_channels,
                    hidden,
                    heads=heads,
                    concat=CONCAT,
                    dropout=self.dropout,
                    add_self_loops=ADD_SELF_LOOPS,
                    edge_dim=edge_dim,
                )
            )
            in_channels = hidden

        out_channels = int(config.get("out_channels", hidden))
        if self.variational:
            self.lin_mu = nn.Linear(in_channels, out_channels)
            self.lin_logstd = nn.Linear(in_channels, out_channels)
        else:
            self.linear = nn.Linear(in_channels, out_channels)

    def _batch_features(self, node_ids, batch_device):
        x = None
        if self.x is not None:
            x = self.x[node_ids.to(self.x.device)]
            if x.device != batch_device:
                x = x.to(batch_device, non_blocking=True)
        emb = self.node_embedding(node_ids.to(self.node_embedding.weight.device))
        if emb.device != batch_device:
            emb = emb.to(batch_device, non_blocking=True)
        return emb if x is None else torch.cat([x, emb], dim=-1)

    def forward(self, batch):
        node_ids = batch.n_id if hasattr(batch, "n_id") else torch.arange(batch.num_nodes, device=batch.edge_index.device)
        x = self._batch_features(node_ids, batch.edge_index.device)
        edge_attr = getattr(batch, "edge_attr", None)
        if self.use_gdc and edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if not self.use_gdc:
            edge_attr = None
        for conv in self.convs:
            x = F.elu(conv(x, batch.edge_index, edge_attr=edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.variational:
            return self.lin_mu(x), self.lin_logstd(x).clamp(-10.0, 10.0)
        return self.linear(x)


def train_homogeneous_gat_minibatch_embeddings(
    data,
    config,
    precomputed_pos=None,
    precomputed_attr=None,
    run_seed=None,
):
    if run_seed is not None:
        torch.manual_seed(int(run_seed))
        torch.cuda.manual_seed_all(int(run_seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variational = bool(config.get("variational", True))
    beta_kl = float(config.get("beta_kl", 0.1))
    neg_ratio = float(config.get("neg_ratio", 1.0))
    num_layers = int(config.get("num_layers", 1))
    epochs = int(config.get("epochs", 50))
    num_nodes = int(data.num_nodes)

    orig_cpu = minibatch_undirected_edge_index(data).cpu()
    pos_cpu = precomputed_pos.cpu() if precomputed_pos is not None else orig_cpu
    pos_attr_cpu = precomputed_attr.cpu() if precomputed_attr is not None else None

    fanout = int(config.get("neighbor_batch_size", 10))
    neighbor_sizes = [fanout] * num_layers
    batch_size = int(config.get("batch_size", 1024))
    loader_workers = int(config.get("loader_workers", 0))

    model = MiniBatchGAT(data, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-3)))
    pin_memory = bool(torch.cuda.is_available())

    train_data = Data(edge_index=pos_cpu, num_nodes=num_nodes)
    if pos_attr_cpu is not None:
        train_data.edge_attr = pos_attr_cpu
    loader_kwargs = dict(
        num_neighbors=neighbor_sizes,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_workers,
    )
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = True
    if pin_memory:
        loader_kwargs["pin_memory"] = True
    train_loader = LinkNeighborLoader(
        data=train_data,
        edge_label_index=orig_cpu,
        neg_sampling_ratio=neg_ratio,
        **loader_kwargs,
    )

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            if variational:
                mu, logstd = out
                z = mu + torch.randn_like(mu) * torch.exp(logstd)
            else:
                z = out
            edge_label_index = batch.edge_label_index
            edge_label = batch.edge_label.float()
            logits = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=1)
            recon = F.binary_cross_entropy_with_logits(logits, edge_label)
            loss = recon + beta_kl * (kl_gaussian(mu, logstd) / max(batch.num_nodes, 1)) if variational else recon
            loss.backward()
            optimizer.step()

    full_graph = Data(edge_index=pos_cpu, num_nodes=num_nodes)
    if pos_attr_cpu is not None:
        full_graph.edge_attr = pos_attr_cpu

    model.eval()
    with torch.no_grad():
        full_graph = full_graph.to(device, non_blocking=True)
        out = model(full_graph)
        z_out = (out[0] if variational else out).detach().cpu()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return z_out.numpy()
