import networkx as nx
import copy

class ReadOnlyCluster:
    def __init__(self, name, entites = None):
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

    def difference(self, e: set) -> 'ReadOnlyArchitecture':
        result = copy.deepcopy(self)
        for k in result.entities.keys():
            result.entities[k].entities -= e
        return result

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

                
class MCFP:
    def __init__(self, src, tgt):
        self._balance(src, tgt)
        self.match_set = self._solve(src, tgt)

    def _balance(self, src, tgt):
        smaller = src if len(src) < len(tgt) else tgt
        for i in range(abs(len(src) - len(tgt))):
            smaller.entities[f'dummy_{i}'] = ReadOnlyCluster(f'dummy_{i}')

    def _solve(self, src, tgt):
        graph = self._make_graph(src, tgt)

        graph.nodes['source']['demand'] = -len(src)
        graph.nodes['sink']['demand'] = len(tgt)
        nx.set_edge_attributes(graph, 1, 'capacity')

        fcost, fdict = nx.capacity_scaling(graph)
        self.cost = fcost
        
    def _make_graph(self, src, tgt) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_node('source')
        graph.add_node('sink')
        first_pass = True
        for k1, v1 in src.entities.items():
            vert1 = f'source_202311111800_{k1}'
            graph.add_node(vert1)
            graph.add_edge('source', vert1, weight=0)
            
            for k2, v2 in tgt.entities.items():
                cost = len(v1.entities ^ v2.entities)
                if first_pass:
                    vert2 = f'target_202311111800_{k2}'
                    graph.add_node(vert2)
                    graph.add_edge(vert2, 'sink', weight=0)
                graph.add_edge(vert1, vert2, weight=cost)

        first_pass = False

        return graph

class A2ACalculator:
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
        num_cluster_diff = abs(len(self.source) - len(self.target))
        src_ents = self.source.get_ents()
        tgt_ents = self.target.get_ents()

        added_ents = tgt_ents - src_ents
        removed_ents = src_ents - tgt_ents

        src_trimmed = self.source.difference(removed_ents)
        tgt_trimmed = self.target.difference(added_ents)

        mcfp = MCFP(src_trimmed, tgt_trimmed)
        num_moved = mcfp.cost / 2

        return num_cluster_diff + 2 * len(added_ents) + 2 * len(removed_ents) + num_moved

    def _denominator(self):
        return len(self.source) + 2 * self.source.count_ents() + len(self.target) + 2 * self.target.count_ents()
