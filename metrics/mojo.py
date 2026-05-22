# Direct translation from MoJoCalulator.java
# Should be cleaned up before release
import numpy as np
class Vertex:
    def __init__(self):
        self.matched = False
        self.is_left = False
        self.outdeg = 0
        self.indeg = 0


class BipartiteGraph:
    def __init__(self, pts, l_pts, r_pts):
        self.left_points = l_pts
        self.right_points = r_pts
        self.points = pts

        self.adj_list = [ [] for _ in range(pts) ]
        self.vertex = []
        self.augment_path = []
        for i in range(pts):
            self.vertex.append(Vertex())
            if i < l_pts:
                self.vertex[i].is_left = True

    def matching(self):
        s = ''
        while self.findAugmentPath():
            s += self.XOR()
        return s

    def findAugmentPath(self):
        self.augment_path.clear()
        for i in range(self.left_points):
            if not self.vertex[i].matched:
                if self.findPath(i):
                    return True
                else:
                    self.augment_path.clear()
        return False

    def findPath(self, start):
        if self.vertex[start].outdeg == 0:
            return False
        self.augment_path.append(start)
        for i in range(len(self.adj_list[start])):
            np = self.adj_list[start][i]
            if np in self.augment_path:
                continue
            if not self.vertex[np].matched:
                self.augment_path.append(np)
                return True
            elif self.findPath(np):
                return True

        self.augment_path.remove(start)
        return False
   
    def XOR(self):
        s = ''
        start = self.augment_path[0]
        for i in range(1, len(self.augment_path)):
            end = self.augment_path[i]
            self.reverse_edge(start, end)
            start = end
        return s
    
    def add_edge(self, sp, ep):
        self.adj_list[sp].append(ep)
        self.vertex[sp].outdeg += 1;
        self.vertex[ep].indeg += 1;

        if self.is_right(sp) and self.is_left(ep):
            self.vertex[sp].matched = True
            self.vertex[ep].matched = True

    def remove_edge(self, sp, ep):
        try:
            self.adj_list[sp].remove(ep)
            self.vertex[sp].outdeg -= 1
            self.vertex[ep].indeg -= 1

            if self.is_right(sp) and self.vertex[sp].outdeg == 0:
                self.vertex[sp].matched = False

            if self.is_left(ep) and self.vertex[ep].indeg == 0:
                self.vertex[ep].matched = False
        except:
            pass
              
    def reverse_edge(self, sp, ep):
        self.remove_edge(sp, ep)
        self.add_edge(ep, sp)
    
    def is_left(self, pt):
        return pt < self.left_points

    def is_right(self, pt):
        return pt > self.left_points - 1

class Cluster:
    def __init__(self, no=0, l=0, m=0):
        self.no = no
        self.l = l
        self.m = m
        self.maxtag = 0
        self.totaltags = 0
        self.groupNo = 0
        self.group = 0
        self.misplacedOmnipresentObjects = 0
        self.tags = [ 0 for _ in range(m) ]
        self.objList = [ [] for _ in range(m) ]
        self.groupList = []
        self.isempty = False;
    
    def addobject(self, tag, obj, mode):
        if mode == 'MoJo':
            return self.addobject_mojo(tag, obj)
        #else:
        #    return self.addobject_mojoplus(tag, obj)        

    def addobject_mojo(self, tag, obj):
        if tag >= 0 and tag < self.m:
            self.tags[tag] += 1
            self.totaltags += 1
            self.objList[tag].append(obj)

            if self.tags[tag] > self.maxtag:
                self.maxtag = self.tags[tag]
                self.group = tag
                self.groupNo = 1
                self.groupList.clear()
                self.groupList.append(tag)
            elif self.tags[tag] == self.maxtag:
                self.groupNo += 1
                self.groupList.append(tag)

        return self.group

    def __str__(self):
        s = ''
        s = s + 'A' + str(self.no + 1) + ' is in group G' + str(self.group + 1) + '\n' 
        for i in range(self.m):
            if len(self.objList[i]) > 0:
                s = s + 'Group ' + str(i + 1) + ':' + ' have ' + str(len(self.objList[i])) + ' objects, they are ' + str(self.objList[i]) + '\n'    
        
        return s

class MoJoCalculator:
    def __init__(self, src, tgt, mode='array'):
        self.source = src
        self.target = tgt
        self.mode = mode
        
        self.mapObjectClusterInB = {}
        self.mapClusterTagA = {}
        self.mapClusterTagB = {}
        
        self.clusterNamesInA = []
        self.partitionA = []
        self.cardinalitiesInB = []
        
    def mojofm(self):

        self.commonPrep()
        self.tagAssignment('MoJo')            
        self.maxbipartiteMatching()
        
        return self.mojofmValue(self.cardinalitiesInB, self.numberOfObjectsInA, self.calc_cost());

    def mojofmValue(self, num_of_B, obj_num, tot_cost):
        mojofm_val = 0
        max_dist = self.max_distance_to(num_of_B, obj_num)
        if tot_cost == 0:
            if obj_num > 1:
                mojofm_val = round((1 - tot_cost / max_dist ) * 10_000) / 100
        else:
            mojofm_val = round((1 - tot_cost / max_dist ) * 10_000) / 100

        return mojofm_val

    def max_distance_to(self, num_of_B, obj_num):
        group_num = 0
        B = sorted([ num_of_B[i] for i in range(len(num_of_B))])
        for i in range(len(B)):
            if group_num < B[i]:
                group_num += 1

        return obj_num - group_num

    def calc_cost(self):
        moves = 0
        no_of_nonempty_group = 0

        for i in range(self.l):
            if self.groupscount[self.A[i].group] == 0:
                no_of_nonempty_group += 1
            if self.grouptags[self.A[i].group] is None:
                self.grouptags[self.A[i].group] = self.A[i]
            self.groupscount[self.A[i].group] += 1
            moves += self.A[i].totaltags - self.A[i].maxtag
            
        return moves + self.l - no_of_nonempty_group
        
    def commonPrep(self):
        self.numberOfObjectsInA = 0

        # fix to be more stable!
        assert self.mode in ['file', 'array'], "wrong mode"
        match self.mode:
            case 'file':
                self.readTargetRSFFile();
                self.readSourceRSFFile();
            case 'array':
                self.readTargetFromArray()
                self.readSourceFromArray()

        self.l = len(self.mapClusterTagA) # number of clusters in A
        self.m = len(self.mapClusterTagB) # number of clusters in B
        
        self.A = [ Cluster(i, self.l, self.m) for i in range(self.l) ]
        
        self.groupscount = [ 0 for _ in range(self.m) ] # the count of each group, 0 if empty
        self.grouptags = [ None for _ in range(self.m) ] 

    

    def tagAssignment(self, mode):
        for i in range(self.l):
            for objName in self.partitionA[i]:
                clusterName = self.mapObjectClusterInB.get(objName, '')
                tag = self.mapClusterTagB.get(clusterName, -1)
                self.A[i].addobject(tag, objName, mode)                
    

    def maxbipartiteMatching(self):
        bgraph = BipartiteGraph(self.l + self.m, self.l, self.m);

        for i in range(self.l):
            for j in range(len(self.A[i].groupList)):
                bgraph.add_edge(i, self.l + self.A[i].groupList[j])

        bgraph.matching()

        for i in range(self.l, self.l + self.m):
            if bgraph.vertex[i].matched:
                index = bgraph.adj_list[i][0]
                self.A[index].group = i - self.l
    
    def readSourceRSFFile(self):
        extraInA = 0
        with open(self.source) as fp:
            for row in fp:
                tmp = row.split()
                assert len(tmp) == 3
                if tmp[0].lower() != 'contain':
                    continue
                index = -1
                clusterName = tmp[1]
                objectName = tmp[2]

                if objectName in self.mapObjectClusterInB:
                    self.numberOfObjectsInA += 1
                    objectIndex = self.mapClusterTagA.get(clusterName, None)
                    if objectIndex is None:
                        index = len(self.mapClusterTagA)
                        self.clusterNamesInA.append(clusterName)
                        self.mapClusterTagA[clusterName] = index
                        self.partitionA.append([])
                    else:
                        index = objectIndex
                    self.partitionA[index].append(objectName)
                else:
                    extraInA += 1

    def readSourceFromArray(self):
        extraInA = 0
        for ix, v in enumerate(self.source):
            clusterName = v
            objectName = ix
            
            if objectName in self.mapObjectClusterInB:
                self.numberOfObjectsInA += 1
                objectIndex = self.mapClusterTagA.get(clusterName, None)
                if objectIndex is None:
                    index = len(self.mapClusterTagA)
                    self.clusterNamesInA.append(clusterName)
                    self.mapClusterTagA[clusterName] = index
                    self.partitionA.append([])
                else:
                    index = objectIndex
                self.partitionA[index].append(objectName)
            else:
                extraInA += 1
                    
    def readTargetFromArray(self):
        for ix, v in enumerate(self.target):
            clusterName = v
            objectName = ix
            index = -1
            objectIndex = self.mapClusterTagB.get(clusterName, None)
            if objectIndex is None:
                index = len(self.mapClusterTagB)
                self.cardinalitiesInB.append(1)
                self.mapClusterTagB[clusterName] = index
            else:
                index = objectIndex
                newCardinality = 1 + self.cardinalitiesInB[index]
                self.cardinalitiesInB[index] = newCardinality
            self.mapObjectClusterInB[objectName] = clusterName
                        
    def readTargetRSFFile(self):
        with open(self.target) as fp:
            for row in fp:
                tmp = row.split()
                assert len(tmp) == 3
                if tmp[0].lower() != 'contain':
                    continue

                clusterName = tmp[1].strip()
                objectName = tmp[2].strip()
                index = -1
                objectIndex = self.mapClusterTagB.get(clusterName, None)
                if objectIndex is None:
                    index = len(self.mapClusterTagB)
                    self.cardinalitiesInB.append(1)
                    self.mapClusterTagB[clusterName] = index
                else:
                    index = objectIndex
                    newCardinality = 1 + self.cardinalitiesInB[index]
                    self.cardinalitiesInB[index] = newCardinality
                self.mapObjectClusterInB[objectName] = clusterName


if __name__ == '__main__':
    src_a = []
    with open('src.rsf') as fp:
        for row in fp:
            tmp = row.split()
            src_a.append(tmp[1].strip())

    tgt_a = []
    with open('tgt.rsf') as fp:
        for row in fp:
            tmp = row.split()
            tgt_a.append(tmp[1].strip())


    mjo = MoJoCalculator('tgt.rsf', 'src.rsf', mode='file')
    print(mjo.mojofm())
    mjo = MoJoCalculator('src.rsf', 'tgt.rsf', mode='file')
    print(mjo.mojofm())
    mjo = MoJoCalculator(tgt_a, src_a, mode='array')
    print(mjo.mojofm())
    mjo = MoJoCalculator(src_a, tgt_a, mode='array')
    print(mjo.mojofm())

    