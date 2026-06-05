import numpy as np
from scipy.optimize import linear_sum_assignment


class ReadOnlyCluster:
    def __init__(self, name, entites=None):
        self.name = name
        self.entities = set()
        if entites:
            for e in entites:
                self.entities.add(e)

    def add(self, e):
        self.entities.add(e)

    def __len__(self):
        return len(self.entities)


class ReadOnlyArchitecture:
    def __init__(self):
        self.entities = {}
        self.entity_location_map = {}

    def get_ents(self):
        res = set()
        for _, v in self.entities.items():
            res.update(v.entities)
        return res

    def count_ents(self):
        return sum(len(v) for v in self.entities.values())

    def __len__(self):
        return len(self.entities)

    @staticmethod
    def read_rsf(fn) -> 'ReadOnlyArchitecture':
        result = ReadOnlyArchitecture()
        with open(fn) as fp:
            for row in fp:
                tmp = row.split()
                if tmp[1] not in result.entities:
                    result.entities[tmp[1]] = ReadOnlyCluster(tmp[1])
                result.entities[tmp[1]].add(tmp[2])
                result.entity_location_map[tmp[2]] = result.entities[tmp[1]]
        return result

    @staticmethod
    def from_array(labels: list[int]) -> 'ReadOnlyArchitecture':
        result = ReadOnlyArchitecture()
        for idx, cluster_id in enumerate(labels):
            cid = str(cluster_id)
            if cid not in result.entities:
                result.entities[cid] = ReadOnlyCluster(cid)
            entity_id = str(idx)
            result.entities[cid].add(entity_id)
            result.entity_location_map[entity_id] = result.entities[cid]
        return result


class A2ACalculator:
    """Architecture-to-architecture distance.

    The moved-entity term is the minimum-cost matching between the two
    architectures' clusters, where the cost of pairing clusters is the size of
    the symmetric difference of their entity sets. This is a balanced linear
    assignment, solved here with scipy.optimize.linear_sum_assignment.

    The cluster intersections needed for the costs are read off an O(N)
    contingency table via |S_i ^ T_j| = |S_i| + |T_j| - 2|S_i n T_j|, which
    replaces the previous networkx min-cost-flow formulation. Results are
    numerically identical; this implementation is dramatically faster.
    """

    def __init__(self, src, tgt, mode='file'):
        assert mode in ['file', 'array']
        self.mode = mode
        if mode == 'file':
            self.source = ReadOnlyArchitecture.read_rsf(src)
            self.target = ReadOnlyArchitecture.read_rsf(tgt)
        elif mode == 'array':
            self.source = ReadOnlyArchitecture.from_array(src)
            self.target = ReadOnlyArchitecture.from_array(tgt)

    def a2a(self):
        return (1 - self._numerator() / self._denominator()) * 100

    def _numerator(self):
        src, tgt = self.source, self.target
        p_clusters = len(src)
        g_clusters = len(tgt)
        num_cluster_diff = abs(p_clusters - g_clusters)

        src_ents = src.get_ents()
        tgt_ents = tgt.get_ents()
        added_ents = tgt_ents - src_ents
        removed_ents = src_ents - tgt_ents
        common = src_ents & tgt_ents

        # Cluster index maps (stable order of the dict keys).
        s_idx = {k: i for i, k in enumerate(src.entities.keys())}
        t_idx = {k: i for i, k in enumerate(tgt.entities.keys())}

        # entity -> list of cluster indices it belongs to (entities may be
        # duplicated across clusters in RSF input).
        ent_src = {}
        for k, cluster in src.entities.items():
            for e in cluster.entities:
                ent_src.setdefault(e, []).append(s_idx[k])
        ent_tgt = {}
        for k, cluster in tgt.entities.items():
            for e in cluster.entities:
                ent_tgt.setdefault(e, []).append(t_idx[k])

        # Costs are over the trimmed architectures (entities present in both),
        # so accumulate sizes and intersections over the common entities only.
        size_p = np.zeros(p_clusters, dtype=np.int64)
        size_t = np.zeros(g_clusters, dtype=np.int64)
        contingency = np.zeros((p_clusters, g_clusters), dtype=np.int64)
        for e in common:
            si = ent_src.get(e, ())
            tj = ent_tgt.get(e, ())
            for i in si:
                size_p[i] += 1
            for j in tj:
                size_t[j] += 1
            for i in si:
                for j in tj:
                    contingency[i, j] += 1

        num_moved = self._min_move_cost(size_p, size_t, contingency) / 2.0
        return num_cluster_diff + 2 * len(added_ents) + 2 * len(removed_ents) + num_moved

    @staticmethod
    def _min_move_cost(size_p, size_t, contingency):
        p, g = len(size_p), len(size_t)
        k = max(p, g)
        if k == 0:
            return 0.0
        # Balance with dummy (empty) clusters on the smaller side, then solve.
        sp = np.zeros(k, dtype=np.int64); sp[:p] = size_p
        st = np.zeros(k, dtype=np.int64); st[:g] = size_t
        padded = np.zeros((k, k), dtype=np.int64); padded[:p, :g] = contingency
        cost = sp[:, None] + st[None, :] - 2 * padded   # |S_i ^ T_j|
        rows, cols = linear_sum_assignment(cost)
        return float(cost[rows, cols].sum())

    def _denominator(self):
        return (len(self.source) + 2 * self.source.count_ents()
                + len(self.target) + 2 * self.target.count_ents())
