import os
import numpy as np
import json
from collections import OrderedDict

node_map = {}

class Node():
	def __init__(self, wnid, parent=None, matters=0):
		self.wnid = wnid
		self.parent = parent
		self.matters = matters
	
	def addChild(self, child):
		child.parent = self.wnid

	@staticmethod
	def getNode(wnid):
		global node_map
		if wnid not in node_map:
			node_map[wnid] = Node(wnid, None, 0)
		return node_map[wnid]
	
	def __str__(self):
		return str(self.wnid) + ' ' + str(self.matters)

	def __repr__(self):
		return str(self.wnid) + ' ' + str(self.matters)
	
def is_ancestor(ans, child):
	parent = node_map[child].parent
	while parent is not None:
		if parent == ans:
			return True
		parent = node_map[parent].parent
	return False

def select_wnids(hierarchy, N, fix_ancestor=None):
	N_wnids = []

	for (count, name, wnid) in hierarchy:

		if len(N_wnids) == N:
			break
		
		if fix_ancestor is None or is_ancestor(fix_ancestor, wnid):
			N_wnids_replace = []
			for c, n, w in N_wnids:
				if is_ancestor(w, wnid):
					pass
				else:
					N_wnids_replace.append((c, n, w))
			N_wnids_replace.append((count, name, wnid))
			N_wnids = N_wnids_replace
	return N_wnids

def get_class_ranges(g, balanced=True):
	class_ranges = []
	for k in g:
		class_ranges.append(g[k]['numbers'])
	
	if balanced:
		min_count = min([len(c) for c in class_ranges])
		class_ranges = [set([c[i] for i in range(min_count)]) for c in class_ranges]
	return class_ranges

def load_imagenet_info(ds_path, ds_info_path):

	# Get list of wnids in ImageNet dataset
	files = os.listdir(os.path.join(ds_path, 'train'))
	imagenet_wnids = [f for f in files if f[0]=='n'] 

	# Load map from wnid to class name
	f = open(os.path.join(ds_info_path, 'words.txt'))
	wnid_to_name = [l.strip() for l in f.readlines()]
	wnid_to_name = {l.split('\t')[0]: l.split('\t')[1] for l in wnid_to_name}

	# Get map between class number, labels and class names
	with open(os.path.join(ds_info_path, 'imagenet_class_index.json'), 'r') as f:
		base_map = json.load(f)
		wnid_to_class = {v[0]: int(k) for k, v in base_map.items()}
		name_to_class = {int(k): v[1] for k, v in base_map.items()}

	return imagenet_wnids, wnid_to_name, wnid_to_class, name_to_class


def get_imagenet_group(ds_path, 
					   ds_info_path,
					   n_classes,
					   fix_ancestor=None,
					   balanced=True):
	
	# Recover all nodes in wordnet along with parent-child relations
	with open(os.path.join(ds_info_path, 'wordnet.is_a.txt'), 'r') as f:
			for line in f.readlines():
				parent, child = line.strip('\n').split(' ')
				parentNode, childNode = Node.getNode(parent), Node.getNode(child)
				if parentNode.matters != 1:
					parentNode.addChild(childNode)

	# Load information about ImageNet such as wnids 
	imagenet_wnids, wnid_to_name, wnid_to_class, name_to_class = load_imagenet_info(ds_path,
																	ds_info_path)

	# Calculate node importance based on number of descendents in ImageNet
	for wnid in imagenet_wnids:
		node_map[wnid].matters = 1

	for wnid in imagenet_wnids:
		node = node_map[wnid]
		while node.parent is not None:
			node_map[node.parent].matters += 1
			node = node_map[node.parent]

	# Filter part of WordNet hierarchy that is relevant
	working_node_map = [(node_map[wnid].matters, wnid_to_name[wnid], wnid) \
					for wnid in node_map if node_map[wnid].matters > 0]

	hierarchy = sorted(working_node_map)[::-1]

	# Chose n_classes # of classes, possibly with a fixed ancestor
	chosen_classes = select_wnids(hierarchy, 
								  n_classes, 
								  fix_ancestor=fix_ancestor)

	# Get all classes in ImageNet that are descendents of chosen classes 
	imagenet_wnid_subgraph = OrderedDict({})
	for (c, n, w) in chosen_classes:
		subset, meanings, numbers = [], [], []
		for iw in imagenet_wnids:
			if is_ancestor(w, iw):
				subset.append(iw)
				meanings.append(wnid_to_name[iw])
				numbers.append(wnid_to_class[iw])
		imagenet_wnid_subgraph[(w, c, n)] = {'wnids': subset, 'names': meanings, 'numbers': numbers}

	label_map = {i: k[2] for i, k in enumerate(imagenet_wnid_subgraph)}
	class_ranges = get_class_ranges(imagenet_wnid_subgraph, balanced=balanced)
	return label_map, class_ranges




