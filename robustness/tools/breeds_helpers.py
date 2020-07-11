import sys, os
import json
import numpy as np
from collections import OrderedDict, Counter
import operator
import networkx as nx
from ..datasets import DATASETS

class ClassHierarchy():
    '''
    Class representing a general ImageNet-style hierarchy.
    '''
    def __init__(self, info_dir):
        """
        Args:
            info_dir (str) : Path to hierarchy information files. Contains a 
                "class_hierarchy.txt" file with one edge per line, a
                "node_names.txt" mapping nodes to names, and "class_info.json".
        """
        
        with open(f'{info_dir}/class_info.json', 'r') as f:
            class_info = json.load(f)
            
        self.IN_WNIDS = [c['wnid'] for c in class_info]
        self.WORDNET_TO_NUM = {k['wnid']: k['cid'] for k in class_info}

        with open((f'{info_dir}/class_hierarchy.txt')) as f:
            edges = [l.strip().split() for l in f.readlines()]
            
        self.graph = self._make_parent_graph(self.IN_WNIDS, edges)

        with open((f'{info_dir}/node_names.txt')) as f:
            mapping = [l.strip().split('\t') for l in f.readlines()]
        self.NODE_NAME = {w[0]: w[1] for w in mapping}
            
        self.level_dict = self._make_level_dict(self.graph, root='n00001740')

    @staticmethod
    def _make_parent_graph(nodes, edges):
        """
        Create a graph for the predecessors of the nodes given.

        Args:
            nodes [str] : List of node names to traverse upwards.
            edges [(str, str)] : Tuples of parent-child pairs.

        Return:
            networkx representation of the graph.
        """

        # create full graph
        full_graph_dir = {}
        for p, c in edges:
            if p not in full_graph_dir:
                full_graph_dir[p] = {c: 1}
            else:
                full_graph_dir[p].update({c: 1})
                    
        FG = nx.DiGraph(full_graph_dir)

        # perform backward BFS to get the relevant graph
        graph_dir = {}
        todo = [n for n in nodes if n in FG.nodes()] # skip nodes not in graph
        while todo:
            curr = todo
            todo = []
            for w in curr:
                for p in FG.predecessors(w):
                    if p not in graph_dir:
                        graph_dir[p] = {w: 1}
                    else:
                        graph_dir[p].update({w: 1})
                    todo.append(p)
            todo = set(todo)
        
        return nx.DiGraph(graph_dir)

    @staticmethod
    def _make_level_dict(graph, root):
        """
        Map nodes to their level within the hierarchy (top-down).

        Args:
            graph (networkx graph( : Graph representation of the hierarchy
            root (str) : Hierarchy root.

        Return:
            Dictionary mapping node names to integer level.
        """    

        level_dict = {} 
        todo = [(root, 0)] # (node, depth)
        while todo:
            curr = todo
            todo = []
            for n, d in curr:
                if n not in level_dict:
                    level_dict[n] = d
                else:
                    level_dict[n] = max(d, level_dict[n]) # keep longest path
                for c in graph.successors(n):
                    todo.append((c, d + 1))

        return level_dict

    def leaves_reachable(self, n):
        """
        Determine the leaves (ImageNet classes) reachable for a give node.

        Args:
            n (str) : WordNet ID of node

        Returns:
            leaves (list): List of WordNet IDs of the ImageNet descendants
        """    
        leaves = set()
        todo = [n]
        while todo:
            curr = todo
            todo = []
            for w in curr:
                for p in self.graph.successors(w):
                    if p in self.IN_WNIDS:
                        leaves.add(p)
                    else:
                        todo.append(p)
            todo = set(todo)

        # If the node itself is an ImageNet node
        if n in self.IN_WNIDS: leaves = leaves.union([n])
        return leaves

    def node_name(self, n):
        """
        Name of a node.
        """    
        if n in self.NODE_NAME:
            return self.NODE_NAME[n]
        else:
            return n

    def print_node_info(self, nodes):
        """
        Prints basic information (name, number of ImageNet descendants) 
        about a given set of nodes.

        Args:
            nodes (list) : List of WordNet IDs of relevant nodes
        """    

        for n in nodes:
            if n in self.NODE_NAME:
                print_str = f"{n}: {self.NODE_NAME[n]}"
            else:
                print_str = n

            print_str += f" ({len(self.leaves_reachable(n))})"
            print(print_str)

    def traverse(self, nodes, direction='down', depth=20):
        """
        Find all nodes accessible from a set of nodes.

        Args:
            nodes (list) : List of WordNet IDs of relevant nodes
            direction ("up"/"down"): Allowed traversal directions
            depth (int): Maximum depth to traverse (from nodes)

        Returns:
            Set of nodes reachable within the desired depth, in the
            desired direction.
        """    

        if not nodes or depth == 0:
            return nodes

        todo = []
        for n in nodes:
            if direction == 'down':
                todo.extend(self.graph.successors(n))
            else: 
                todo.extend(self.graph.predecessors(n))
        return nodes + self.traverse(todo, direction=direction, depth=depth-1)

    def get_nodes_at_level(self, L, ancestor=None):
        """
        Find all superclasses at a specified depth.

        Args:
            L (int): Depth in hierarchy (from root node)
            ancestor (str): (optional) WordNet ID that can be used to
                            restrict the subtree in which superclasses
                            are found

        Returns:
            nodes (list): Set of superclasses at that depth in 
                                   the hierarchy
        """            
        if ancestor is not None:
            valid = set(self.traverse([ancestor], direction="down"))

        nodes= set()
        for k, v in self.level_dict.items() :
            if v == L and (ancestor is None or k in valid):
                nodes.add(k)
                
        return nodes

class BreedsDatasetGenerator():
    '''
    Class for generating datasets from ImageNet superclasses.
    '''
    def __init__(self, info_dir):    
        self.hierarchy = ClassHierarchy(info_dir)

    def split_superclass(self, superclass_wnid, Nsubclasses, balanced,
                            split_type, rng=np.random.RandomState(2)):

        """
        Split superclass into two disjoint sets of subclasses.

        Args:
            superclass_wnid (str): WordNet ID of superclass node
            Nsubclasses (int): Number of subclasses per superclass
                               (not used when balanced is True)
            balanced (bool): Whether or not the dataset should be
                             balanced over superclasses
            split_type ("good"/"bad"/"rand"): Whether the subclass
                             partitioning should be more or less 
                             adversarial or random
            rng (RandomState): Random number generator

        Returns:
            class_ranges (tuple): Tuple of lists of subclasses
        """ 

        # Find a descendant of the superclass that has at least two children        
        G = self.hierarchy.graph
        node, desc = superclass_wnid, sorted(list(G.successors(superclass_wnid)))
        while len(desc) == 1:
            node = desc[0]
            desc = sorted(list(G.successors(node)))
        
        # Map each descendant to its ImageNet subclasses
        desc_map = {}
        for d in desc:
            dcurr = sorted(list(self.hierarchy.leaves_reachable(d)))
            desc_map[d] = dcurr

        # Map sorted by nodes that have the maximum number of children
        desc_sorted = sorted(desc_map.items(), key=lambda x: -len(x[1]))

        # If not balanced, we will pick as many subclasses as possible
        # from this superclass (ignoring Nsubclasses)
        if not balanced:
            S = sum([len(d) for d in desc_map.values()])
            assert S >= Nsubclasses
            Nsubclasses = S
            if Nsubclasses % 2 != 0:
                Nsubclasses = max(Nsubclasses - 1, 2)

        # Split superclasses into two disjoint sets
        assert Nsubclasses % 2 == 0
        Nh = Nsubclasses // 2

        if split_type == "rand":
            # If split is random, aggregate all subclasses, subsample
            # the desired number, and then partition into two
            desc_node_list = []
            for d in desc_map.values():
                desc_node_list.extend(d)
            sel = rng.choice(sorted(desc_node_list), Nh * 2, replace=False)
            split = (sel[:Nh], sel[Nh:])
        else:
            # If split is good, we will partition similar subclasses across
            # both domains. If it is bad, similar subclasses will feature in
            # one or the other

            split, spare = ([], []), []
            
            for k, v in desc_sorted:
                l = [len(s) + 0 for s in split]
                if split_type == "bad":            
                    if l[0] <= l[1] and l[0] < Nh:
                        if len(v) > Nh - l[0]: spare.extend(v[Nh-l[0]:])
                        split[0].extend(v[:Nh-l[0]])
                    elif l[1] < Nh:
                        if len(v) > Nh - l[1]: spare.extend(v[Nh-l[1]:])
                        split[1].extend(v[:Nh-l[1]])
                else:
                    if len(v) == 1:
                        i1 = 1 if l[0] < Nh else 0
                    else:
                        i1 = min(len(v) // 2, Nh - l[0])
                        
                    if l[0] < Nh:
                        split[0].extend(v[:i1])
                    if l[1] < Nh:
                        split[1].extend(v[i1:i1 + Nh-l[1]])
            
            if split_type == "bad":
                l = [len(s) + 0 for s in split]
                assert max(l) == Nh
                if l[0] < Nh:
                    split[0].extend(spare[:Nh - l[0]])
                if l[1] < Nh:
                    split[1].extend(spare[:Nh - l[1]])
                
        assert len(split[0]) == len(split[1]) and not set(split[0]).intersection(split[1])
        class_ranges = ([self.hierarchy.WORDNET_TO_NUM[s] for s in split[0]],
                        [self.hierarchy.WORDNET_TO_NUM[s] for s in split[1]])
        
        return class_ranges

    def get_superclasses(self, level, Nsubclasses=None,
                         split=None, ancestor=None, balanced=True, 
                         random_seed=2, verbose=False):
        """
        Obtain a dataset composed of ImageNet superclasses with a desired
        set of properties. 
        (Optional) By specifying a split, one can parition the subclasses
        into two disjoint datasets (with the same superclasses).

        Args:
            level (int): Depth in hierarchy (from root node)
            Nsubclasses (int): Number of subclasses per superclass
                               (not used when balanced is True)
            balanced (bool): Whether or not the dataset should be
                             balanced over superclasses
            split ("good"/"bad"/"rand"/None): If None, superclasses are
                             not partitioned into two disjoint datasets.
                             If not None, determines whether the subclass
                             partitioning should be more or less 
                             adversarial or random
            rng (RandomState): Random number generator

        Returns:
            subclass_ranges (list): Each entry is a list of subclasses 
                                    for a given superclass in the dataset
            label_map (dict): Map from class number to superclass name 
            subclass_tuple (tuple): Tuple of lists capturing the split.
                                    If split is None, lists are empty.
                                    Otherwise, each list entry is 
                                    a list of subclasses for a given 
                                    superclass in the dataset
            superclasses (list): WordNet IDs of superclasses
            all_subclasses (list): List of all possible subclasses included in 
                                    superclass

        """ 

        rng = np.random.RandomState(random_seed)
        hierarchy = self.hierarchy

        # Identify superclasses at this level
        rel_nodes = sorted(list(hierarchy.get_nodes_at_level(
                                            level, ancestor=ancestor)))
        if verbose: hierarchy.print_node_info(rel_nodes)

        # Count number of subclasses
        in_desc = []
        for n in rel_nodes:
            dcurr = self.hierarchy.leaves_reachable(n)
            in_desc.append(sorted(list(dcurr)))
        min_desc = np.min([len(d) for d in in_desc])
        assert min_desc > 0

        # Determine number of subclasses to include per superclass
        if Nsubclasses is None:
            if balanced:
                Nsubclasses = min_desc
                if Nsubclasses % 2 != 0: Nsubclasses = max(2, Nsubclasses - 1)
            else:
                Nsubclasses = 1 if split is None else 2

        # Find superclasses that have sufficient subclasses
        superclass_idx = [i for i in range(len(rel_nodes)) 
                          if len(in_desc[i]) >= Nsubclasses]
        superclasses, all_subclasses = [rel_nodes[i] for i in superclass_idx], \
                                        [in_desc[i] for i in superclass_idx]

        # Superclass names
        label_map = {}
        for ri, r in enumerate(superclasses):
            label_map[ri] = self.hierarchy.node_name(r)

        subclass_ranges, subclass_tuple = [], ([], [])
        
        if split is None:

            if balanced:
                Ns = [Nsubclasses] * len(all_subclasses)
            else:
                Ns = [len(d) for d in all_subclasses]
            wnids = [list(rng.choice(d, n, replace=False))
                                   for n, d in zip(Ns, all_subclasses)] 
            subclass_ranges = [[self.hierarchy.WORDNET_TO_NUM[w] for w in c] for c in wnids]
        else:
            for sci, sc in enumerate(sorted(superclasses)):
                class_tup = self.split_superclass(sc, Nsubclasses=Nsubclasses, 
                                                     balanced=balanced,
                                                     split_type=split, rng=rng)
                subclass_tuple[0].append(class_tup[0])
                subclass_tuple[1].append(class_tup[1])
                subclass_ranges.append(class_tup[0] + class_tup[1])

        return subclass_ranges, label_map, subclass_tuple, superclasses, all_subclasses


# Some standard datasets from the BREEDS paper.

def Entity13(info_dir, split=None):
    """
    ENTITY-13 Dataset
    Args:
        info_dir (str) : Path to ImageNet information files
        split ("good"/"bad"/"rand"/None): Nature of subclass
    Returns:
        subclass_ranges (list): Each entry is a list of subclasses 
                                for a given superclass in the dataset
        label_map (dict): Map from class number to superclass name 
        subclass_tuple (tuple): Tuple of lists capturing the split; 
                                each list entry is 
                                a list of subclasses for a given 
                                superclass in the dataset
        superclasses (list): WordNet IDs of superclasses
        all_subclasses (list): List of all possible subclasses included in 
                                superclass

    """ 

    DG = BreedsDatasetGenerator(info_dir)
    return DG.get_superclasses(level=3, 
                       ancestor=None,
                       Nsubclasses=20, 
                       split=split, 
                       balanced=True, 
                       random_seed=2,
                       verbose=False)

def Entity30(info_dir, split=None):
    """
    ENTITY-30 Dataset
    Args:
        info_dir (str) : Path to ImageNet information files
        split ("good"/"bad"/"rand"/None): Nature of subclass
    Returns:
        subclass_ranges (list): Each entry is a list of subclasses 
                                for a given superclass in the dataset
        label_map (dict): Map from class number to superclass name 
        subclass_tuple (tuple): Tuple of lists capturing the split; 
                                each list entry is 
                                a list of subclasses for a given 
                                superclass in the dataset
        superclasses (list): WordNet IDs of superclasses
        all_subclasses (list): List of all possible subclasses included in 
                                superclass

    """ 
    DG = BreedsDatasetGenerator(info_dir)
    return DG.get_superclasses(level=4, 
                       ancestor=None,
                       Nsubclasses=8, 
                       split=split, 
                       balanced=True, 
                       random_seed=2,
                       verbose=False)

def Living17(info_dir, split=None):
    """
    LIVING-17 Dataset
    Args:
        info_dir (str) : Path to ImageNet information files
        split ("good"/"bad"/"rand"/None): Nature of subclass
    Returns:
        subclass_ranges (list): Each entry is a list of subclasses 
                                for a given superclass in the dataset
        label_map (dict): Map from class number to superclass name 
        subclass_tuple (tuple): Tuple of lists capturing the split; 
                                each list entry is 
                                a list of subclasses for a given 
                                superclass in the dataset
        superclasses (list): WordNet IDs of superclasses
        all_subclasses (list): List of all possible subclasses included in 
                                superclass

    """ 
    DG = BreedsDatasetGenerator(info_dir)
    return DG.get_superclasses(level=5, 
                       ancestor="n00004258",
                       Nsubclasses=4, 
                       split=split, 
                       balanced=True, 
                       random_seed=2,
                       verbose=False)

def Nonliving26(info_dir, split=None):
    """
    NONLIVING-26 Dataset
    Args:
        info_dir (str) : Path to ImageNet information files
        split ("good"/"bad"/"rand"/None): Nature of subclass
    Returns:
        subclass_ranges (list): Each entry is a list of subclasses 
                                for a given superclass in the dataset
        label_map (dict): Map from class number to superclass name 
        subclass_tuple (tuple): Tuple of lists capturing the split; 
                                each list entry is 
                                a list of subclasses for a given 
                                superclass in the dataset
        superclasses (list): WordNet IDs of superclasses
        all_subclasses (list): List of all possible subclasses included in 
                                superclass

    """ 
    DG = BreedsDatasetGenerator(info_dir)
    return DG.get_superclasses(level=5, 
                       ancestor="n00021939",
                       Nsubclasses=4, 
                       split=split, 
                       balanced=True, 
                       random_seed=2,
                       verbose=False)