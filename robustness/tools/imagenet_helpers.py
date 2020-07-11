import os
import numpy as np
import json
from itertools import product

class Node():
    '''
    Class for representing a node in the ImageNet/WordNet hierarchy. 
    '''
    def __init__(self, wnid, parent_wnid=None, name=""):
        """
        Args:
            wnid (str) : WordNet ID for synset represented by node
            parent_wnid (str) : WordNet ID for synset of node's parent
            name (str) : word/human-interpretable description of synset 
        """

        self.wnid = wnid
        self.name = name
        self.class_num = -1
        self.parent_wnid = parent_wnid
        self.descendant_count_in = 0
        self.descendants_all = set()
    
    def add_child(self, child):
        """
        Add child to given node.

        Args:
            child (Node) : Node object for child
        """
        child.parent_wnid = self.wnid
    
    def __str__(self):
        return f'Name: ({self.name}), ImageNet Class: ({self.class_num}), Descendants: ({self.descendant_count_in})'
    
    def __repr__(self):
        return f'Name: ({self.name}), ImageNet Class: ({self.class_num}), Descendants: ({self.descendant_count_in})'

class ImageNetHierarchy():
    '''
    Class for representing ImageNet/WordNet hierarchy. 
    '''
    def __init__(self, ds_path, ds_info_path):
        """
        Args:
            ds_path (str) : Path to ImageNet dataset
            ds_info_path (str) : Path to supplementary files for the ImageNet dataset 
                                 ('wordnet.is_a.txt', 'words.txt' and 'imagenet_class_index.json')
                                 which can be obtained from http://image-net.org/download-API.

        """
        self.tree = {}

        ret = self.load_imagenet_info(ds_path, ds_info_path)
        self.in_wnids, self.wnid_to_name, self.wnid_to_num, self.num_to_name = ret
            
        with open(os.path.join(ds_info_path, 'wordnet.is_a.txt'), 'r') as f:
            for line in f.readlines():
                parent_wnid, child_wnid = line.strip('\n').split(' ')
                parentNode = self.get_node(parent_wnid)
                childNode = self.get_node(child_wnid)
                parentNode.add_child(childNode)
                
        for wnid in self.in_wnids:
            self.tree[wnid].descendant_count_in = 0
            self.tree[wnid].class_num = self.wnid_to_num[wnid]
            
        for wnid in self.in_wnids:
            node = self.tree[wnid]
            while node.parent_wnid is not None:
                self.tree[node.parent_wnid].descendant_count_in += 1
                self.tree[node.parent_wnid].descendants_all.update(node.descendants_all)
                self.tree[node.parent_wnid].descendants_all.add(node.wnid)
                node = self.tree[node.parent_wnid]
        
        del_nodes = [wnid for wnid in self.tree \
                     if (self.tree[wnid].descendant_count_in == 0 and self.tree[wnid].class_num == -1)]
        for d in del_nodes:
            self.tree.pop(d, None)
                        
        assert all([k.descendant_count_in > 0 or k.class_num != -1 for k in self.tree.values()])

        self.wnid_sorted = sorted(sorted([(k, v.descendant_count_in, len(v.descendants_all)) \
                                        for k, v in self.tree.items()
                                        ],
                                        key=lambda x: x[2], 
                                        reverse=True
                                        ),
                                key=lambda x: x[1], 
                                reverse=True
                                )

    @staticmethod
    def load_imagenet_info(ds_path, ds_info_path):
        """
        Get information about mapping between ImageNet wnids/class numbers/class names.

        Args:
            ds_path (str) : Path to ImageNet dataset
            ds_info_path (str) : Path to supplementary files for the ImageNet dataset 
                                 ('wordnet.is_a.txt', 'words.txt', 'imagenet_class_index.json')
                                 which can be obtained from http://image-net.org/download-API.

        """
        files = os.listdir(os.path.join(ds_path, 'train'))
        in_wnids = [f for f in files if f[0]=='n'] 

        f = open(os.path.join(ds_info_path, 'words.txt'))
        wnid_to_name = [l.strip() for l in f.readlines()]
        wnid_to_name = {l.split('\t')[0]: l.split('\t')[1] \
                             for l in wnid_to_name}

        with open(os.path.join(ds_info_path, 'imagenet_class_index.json'), 'r') as f:
            base_map = json.load(f)
            wnid_to_num = {v[0]: int(k) for k, v in base_map.items()}
            num_to_name = {int(k): v[1] for k, v in base_map.items()}

        return in_wnids, wnid_to_name, wnid_to_num, num_to_name

    def get_node(self, wnid):
        """
        Add node to tree.

        Args:
            wnid (str) : WordNet ID for synset represented by node

        Returns:
            A node object representing the specified wnid.
        """
        if wnid not in self.tree:
            self.tree[wnid] = Node(wnid, name=self.wnid_to_name[wnid])
        return self.tree[wnid]


    def is_ancestor(self, ancestor_wnid, child_wnid):
        """
        Check if a node is an ancestor of another.

        Args:
            ancestor_wnid (str) : WordNet ID for synset represented by ancestor node
            child_wnid (str) : WordNet ID for synset represented by child node

        Returns:
            A boolean variable indicating whether or not the node is an ancestor
        """
        return (child_wnid in self.tree[ancestor_wnid].descendants_all)

    
    def get_descendants(self, node_wnid, in_imagenet=False):
        """
        Get all descendants of a given node.

        Args:
            node_wnid (str) : WordNet ID for synset for node
            in_imagenet (bool) : If True, only considers descendants among 
                                ImageNet synsets, else considers all possible
                                descendants in the WordNet hierarchy

        Returns:
            A set of wnids corresponding to all the descendants
        """        
        if in_imagenet:
            return set([self.wnid_to_num[ww] for ww in self.tree[node_wnid].descendants_all
                        if ww in set(self.in_wnids)])
        else:
            return self.tree[node_wnid].descendants_all
    
    def get_superclasses(self, n_superclasses, 
                         ancestor_wnid=None, superclass_lowest=None, 
                         balanced=True):
        """
        Get superclasses by grouping together classes from the ImageNet dataset.

        Args:
            n_superclasses (int) : Number of superclasses desired
            ancestor_wnid (str) : (optional) WordNet ID that can be used to specify
                                common ancestor for the selected superclasses
            superclass_lowest (set of str) : (optional) Set of WordNet IDs of nodes
                                that shouldn't be further sub-classes
            balanced (bool) : If True, all the superclasses will have the same number
                            of ImageNet subclasses

        Returns:
            superclass_wnid (list): List of WordNet IDs of superclasses
            class_ranges (list of sets): List of ImageNet subclasses per superclass
            label_map (dict): Mapping from class number to human-interpretable description
                            for each superclass
        """             
        
        assert superclass_lowest is None or \
               not any([self.is_ancestor(s1, s2) for s1, s2 in product(superclass_lowest, superclass_lowest)])
         
        superclass_info = []
        for (wnid, ndesc_in, ndesc_all) in self.wnid_sorted:
            
            if len(superclass_info) == n_superclasses:
                break
                
            if ancestor_wnid is None or self.is_ancestor(ancestor_wnid, wnid):
                keep_wnid = [True] * (len(superclass_info) + 1)
                superclass_info.append((wnid, ndesc_in))
                
                for ii, (w, d) in enumerate(superclass_info):
                    if self.is_ancestor(w, wnid):
                        if superclass_lowest and w in superclass_lowest:
                            keep_wnid[-1] = False
                        else:
                            keep_wnid[ii] = False
                
                for ii in range(len(superclass_info) - 1, -1, -1):
                    if not keep_wnid[ii]:
                        superclass_info.pop(ii)
            
        superclass_wnid = [w for w, _ in superclass_info]
        class_ranges, label_map = self.get_subclasses(superclass_wnid, 
                                    balanced=balanced)
                
        return superclass_wnid, class_ranges, label_map


    def get_subclasses(self, superclass_wnid, balanced=True):
        """
        Get ImageNet subclasses for a given set of superclasses from the WordNet 
        hierarchy. 

        Args:
            superclass_wnid (list): List of WordNet IDs of superclasses
            balanced (bool) : If True, all the superclasses will have the same number
                            of ImageNet subclasses

        Returns:
            class_ranges (list of sets): List of ImageNet subclasses per superclass
            label_map (dict): Mapping from class number to human-interpretable description
                            for each superclass
        """      
        ndesc_min = min([self.tree[w].descendant_count_in for w in superclass_wnid]) 
        class_ranges, label_map = [], {}
        for ii, w in enumerate(superclass_wnid):
            descendants = self.get_descendants(w, in_imagenet=True)
            if balanced and len(descendants) > ndesc_min:
                descendants = set([dd for ii, dd in enumerate(sorted(list(descendants))) if ii < ndesc_min])
            class_ranges.append(descendants)
            label_map[ii] = self.tree[w].name
            
        for i in range(len(class_ranges)):
            for j in range(i + 1, len(class_ranges)):
                assert(len(class_ranges[i].intersection(class_ranges[j])) == 0)
                
        return class_ranges, label_map

def common_superclass_wnid(group_name):
    """
        Get WordNet IDs of common superclasses. 

        Args:
            group_name (str): Name of group

        Returns:
            superclass_wnid (list): List of WordNet IDs of superclasses
        """    
    common_groups = {

        # ancestor_wnid = 'n00004258'
        'living_9': ['n02084071', #dog, domestic dog, Canis familiaris
                    'n01503061', # bird
                    'n01767661', # arthropod
                    'n01661091', # reptile, reptilian
                    'n02469914', # primate
                    'n02512053', # fish
                    'n02120997', # feline, felid
                    'n02401031', # bovid
                    'n01627424', # amphibian
                    ],

        'mixed_10': [
                     'n02084071', #dog,
                     'n01503061', #bird 
                     'n02159955', #insect 
                     'n02484322', #monkey 
                     'n02958343', #car 
                     'n02120997', #feline 
                     'n04490091', #truck 
                     'n13134947', #fruit 
                     'n12992868', #fungus 
                     'n02858304', #boat 
                     ],

        'mixed_13': ['n02084071', #dog,
                     'n01503061', #bird (52)
                     'n02159955', #insect (27)
                     'n03405725', #furniture (21)
                     'n02512053', #fish (16),
                     'n02484322', #monkey (13)
                     'n02958343', #car (10)
                     'n02120997', #feline (8),
                     'n04490091', #truck (7)
                     'n13134947', #fruit (7)
                     'n12992868', #fungus (7)
                     'n02858304', #boat (6)  
                     'n03082979', #computer(6)
                    ],

        # Dataset from Geirhos et al., 2018: arXiv:1811.12231
        'geirhos_16': ['n02686568', #aircraft (3)
                       'n02131653', #bear (3)
                       'n02834778', #bicycle (2)
                       'n01503061', #bird (52)
                       'n02858304', #boat (6)
                       'n02876657', #bottle (7)
                       'n02958343', #car (10)
                       'n02121808', #cat (5)
                       'n03001627', #char (4)
                       'n03046257', #clock (3)
                       'n02084071', #dog (116)
                       'n02503517', #elephant (2)
                       'n03614532', #keyboard (3)
                       'n03623556', #knife (2)
                       'n03862676', #oven (2)
                       'n04490091', #truck (7)
                      ],
        'big_12':  ['n02084071', #dog (100+)
                     'n04341686', #structure (55)
                     'n01503061', #bird (52)
                     'n03051540', #clothing (48)
                     'n04576211', #wheeled vehicle
                     'n01661091', #reptile, reptilian (36)
                     'n02075296', #carnivore
                     'n02159955', #insect (27)
                     'n03800933', #musical instrument (26)
                     'n07555863', #food (24)
                     'n03405725', #furniture (21)
                     'n02469914', #primate (20)
                   ],
        'mid_12':  ['n02084071', #dog (100+)
                      'n01503061', #bird (52)
                      'n04576211', #wheeled vehicle
                      'n01661091', #reptile, reptilian (36)
                      'n02075296', #carnivore
                      'n02159955', #insect (27)
                      'n03800933', #musical instrument (26)
                      'n07555863', #food (24)
                      'n03419014', #garment (24)
                      'n03405725', #furniture (21)
                      'n02469914', #primate (20)
                      'n02512053', #fish (16)
                    ]
    }

    if group_name in common_groups:
        superclass_wnid = common_groups[group_name]
        return superclass_wnid
    else:
        raise ValueError("Custom group does not exist")
