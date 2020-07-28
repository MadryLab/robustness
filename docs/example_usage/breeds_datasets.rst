Creating BREEDS subpopulation shift benchmarks
===============================================

In this document, we will discuss how to create BREEDS datasets, used in 
the `code <https://github.com/MadryLab/BREEDS_benchmarks>`_ `release
<https://github.com/MadryLab/BREEDS_benchmarks>`_ based on our paper [STM20]_. 
These datasets can be used to study model robustness to subpopulation 
shift---a specific form of distribution shift wherein the subpopulations of data 
(e.g., animal breeds) present in the training set (source domain) are 
entirely different from the test set (target domain). 
You can create BREEDS datasets from any existing dataset that has 
an explicit class hierarchy describing the semantic relationship 
between pairs of classes (e.g., ImageNet, OpenImages). 
In this walkthrough, we will do so using ImageNet and the corresponding
class hierarchy from [STM20]_ (modification of WordNet to be better-suited
for the object recognition task).

At a high-level, the procedure to construct BREEDS datasets is as follows:

1. Use the dataset's class hierarchy to define `superclasses` by grouping
   together semantically similar dataset classes (e.g., define the `dog`` superclass 
   by aggregating all the dog breed classes). 
2. Design the classification task to be between different superclasses, so that
   we have easy access to the individual subpopulations (i.e., the original 
   dataset classes). Then, for each superclass, we will partition the subpopulations 
   into the source (used for model training) and target domain without overlap
   (used for evaluation). 


.. raw:: html

   <i class="fa fa-play"></i> &nbsp;&nbsp; <a
   href="https://github.com/MadryLab/robustness/blob/master/notebooks/Breeds%20Datasets.ipynb">Download
   a Jupyter notebook</a> containing all the code from this walkthrough! <br />
   <br />

Requirements/Setup
''''''''''''''''''
To create BREEDS datasets using ImageNet, we need to create a: 

- ``data_dir`` which contains the dataset (in this case, ImageNet) 
  in PyTorch-readable format.
- ``info_dir`` which contains information about the class hierarchy.
  For ImageNet, you can find the relevant files in the 
  `imagenet_class_hierarchy/modified` subfolder of our
  `release <https://github.com/MadryLab/BREEDS_benchmarks>`_. 

Part 1: Browsing through the Class Hierarchy
''''''''''''''''''''''''''''''''''''''''''''
The :class:`~robustness.tools.breeds_helpers.ClassHierarchy` class allows
us to probe the class hierarchy. The class hierarchy has a graph structure,
where a child node is a subclass (an instance) of the parent node and the
original dataset classes are the leaves. We use a top-down approach to obtain 
superclasses. Specifically, by defining all nodes as a fixed (user-specified) 
distance from the root node as superclasses, and treating all the descendant 
leaves (dataset classes) as subpopulations. 

To browse through the ImageNet class hierarchy, let's create an instance 
of the ``ClassHierarchy`` class:

.. code-block:: python

   from robustness.tools.breeds_helpers import ClassHierarchy
   hier = ClassHierarchy(info_dir)


Here, :samp:`info_dir` should be the path to a folder which contains
information about the class hierarchy 
(e.g., ``imagenet_class_hierarchy/modified``). In general, :samp:`info_dir`
must contain the following files:

- ``dataset_class_info.json``: A list which contains, for every class
  in the dataset, a triplet of class number, ID (e.g., WordNet ID) and
  name. 
- ``class_hierarchy.txt``: Each line should contain
  information about an edge in the class hierarchy, represented as
  parent node ID followed    by child node ID (space separated). 
- ``node_names.txt``: Each line should contain a node ID
  followed by it's name (tab separated).


The :samp:`hier` object contains a ``graph`` attribute that captures the
hierarchy as a ``networkx`` graph. We can explore some high-level properties
of the graph, for instance:

.. code-block:: python

  import numpy as np
  print(f"# Levels in hierarchy: {np.max(list(hier.level_to_nodes.keys()))}")
  print(f"# Nodes/level:",
      [f"Level {k}: {len(v)}" for k, v in hier.level_to_nodes.items()])

We can now use top-down clustering to find superclasses---by selecting all
nodes at a certain depth from the root node (or any other desired ancestor
node):

.. code-block:: python

  level = 2 # Could be any number smaller than max level
  superclasses = hier.get_nodes_at_level(level)
  print(f"Superclasses at level {level}:\n")
  print(", ".join([f"{hier.HIER_NODE_NAME[s]}" for s in superclasses]))


For a specific superclass, we can also inspect all the leaves reachable
from the superclass (which correspond to classes in the dataset):

.. code-block:: python

  idx = np.random.randint(0, len(superclasses), 1)[0]
  superclass = list(superclasses)[idx]
  leaves = hier.leaves_reachable(superclass)
  print(f"Superclass: {hier.HIER_NODE_NAME[superclass]}\n")

  print(f"Leaves ({len(leaves)}):")
  print([f"{hier.LEAF_ID_TO_NAME[l]}" for l in list(leaves)])


We can also visualize subtrees of the graph with the help of
the `networkx` and `pygraphviz` packages. For instance, we can
taks a look at the subtree of the class hierarchy rooted at a
particular superclass:

.. code-block:: python

  import networkx as nx
  from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
  import pygraphviz as pgv
  from IPython.display import Image

  subtree = nx.ego_graph(hier.graph, superclass, radius=10)
  mapping = {n: hier.HIER_NODE_NAME[n] for n in subtree.nodes()}
  subtree = to_agraph(nx.relabel_nodes(subtree, mapping))
  subtree.delete_edge(subtree.edges()[0])
  subtree.layout('dot')
  subtree.node_attr['color']='blue'
  subtree.draw('graph.png', format='png')
  Image('graph.png')
  
For instance, visualizing tree rooted at the ``fungus`` superclass yields:

.. image:: ../figures/graph.png
  :width: 600
  :alt: Visulization of subtree rooted at a specific superclass.

Part 2: Creating BREEDS Datasets
'''''''''''''''''''''''''''''''''

To create a dataset composed of superclasses, we use the 
:class:`~robustness.tools.breeds_helpers.BreedsDatasetGenerator`.
Internally, this class instantiates an object of 
:class:`~robustness.tools.breeds_helpers.ClassHierarchy` and uses it
to define the superclasses.

.. code-block:: python

  from robustness.tools.breeds_helpers import BreedsDatasetGenerator
  DG = BreedsDatasetGenerator(info_dir)

Specifically, we will use  
py:meth:`~robustness.tools.breeds_helpers.BreedsDatasetGenerator.get_superclasses`.
This function takes in the following arguments (see :meth:`this docstring
<robustness.tools.breeds_helpers.BreedsDatasetGenerator.get_superclasses>` for more details):

- :samp:`level`: Level in the hierarchy (in terms of distance from the
  root node) at which to define superclasses.
- :samp:`Nsubclasses`: Controls the minimum number of subclasses/superclass
  in the dataset. If None, it is automatically set to be the size (in terms
  of subclasses) of the smallest superclass. 
- :samp:`split`: If ``None``, subclasses of a superclass are returned 
  as is, without partitioning them into the source and target domains. 
  Else, can be ``rand/good/bad`` depending on whether the subclass split should be
  random or less/more adversarially chosen [STM20]_.
- :samp:`ancestor`: If a node ID is specified, superclasses are chosen from 
  subtree of class hierarchy rooted at this node. Else, if None, :samp:`ancestor`
  is set to be the root node.
- :samp:`balanced`: If True, subclasses/superclass is fixed over superclasses.

For instance, we could create a balanced dataset, with the subclass partition 
being less adversarial as follows:

.. code-block:: python

   ret = DG.get_superclasses(level=2, 
                          Nsubclasses=None, 
                          split="rand", 
                          ancestor=None, 
                          balanced=True)
  subclass_ranges, label_map, subclass_tuple, superclasses, _ = ret                                    

This method returns:

- :samp:`superclasses` is a list containing the IDs of all the
  superclasses.
- :samp:`label_map` is a dictionary mapping a superclass
  number (label) to name. 
- :samp:`subclass_ranges` is a list, which for
  each superclass, contains a list of subclasses included (in both
  domains). 
- :samp:`subclass_tuple` is a tuple of subclass ranges for
  the source and target domains. For instance,
  :samp:`subclass_tuple[0]` is a list, which for each superclass,
  contains a list of subclasses present in the source domain.

You can experiment with these parameters to create datasets of different
granularity. For instance, you could specify the :samp:`Nsubclasses` to
restrict the size of every superclass in the dataset,
set the :samp:`ancestor` to be a specific node (e.g., ``n00004258`` 
to focus on living things), or set :samp:`balanced` to ``False`` 
to get an imbalanced dataset.

we can take a closer look at the composition of the dataset---what
superclasses/subclasses it contains---using:

.. code-block:: python
  from robustness.tools.breeds_helpers import print_dataset_info

  print_dataset_info(subclass_ranges, 
                    label_map, 
                    subclass_tuple, 
                    superclasses, 
                    hier.LEAF_NUM_TO_NAME)

Finally, for the source and target domains, we can create datasets
and their corresponding loaders:

.. code-block:: python
  from robustness.datasets import DATASETS
  # For the source domain
  dataset_source = DATASETS['custom_imagenet'](data_dir, subclass_tuple[0])
  train_loader_source, val_loader_source = dataset_source.make_loaders(num_workers, 
                                                                     batch_size)
  # For the target domain                                                                     
  dataset_target = DATASETS['custom_imagenet'](data_dir, subclass_tuple[1])
  train_loader_target, val_loader_target = dataset_source.make_loaders(num_workers, 
                                                                     batch_size)

You're all set! You can then use this :samp:`custom_dataset` and loaders
just as you would any other existing/custom dataset in the robustness 
library. For instance, you can visualize validation set samples from
both domains and their labels using:

.. code-block:: python

  from robustness.tools.vis_tools import show_image_row
  for domain, loader in zip(["Source", "Target"],
                            [val_loader_source, val_loader_target]):
      im, lab = next(iter(loader))
      show_image_row([im], 
                     tlist=[[label_map[int(k)].split(",")[0] for k in lab]],
                     ylist=[domain],
                     fontsize=20)

You can also create superclass tasks where subclasses are not 
partitioned across domains: 

.. code-block:: python

  ret = DG.get_superclasses(level=level, 
                            Nsubclasses=Nsubclasses, 
                            split=None, 
                            ancestor=ancestor, 
                            balanced=balanced)
  subclass_ranges, label_map, subclass_tuple, superclasses, _ = ret
  dataset = DATASETS['custom_imagenet'](data_dir, subclass_ranges)

  print_dataset_info(subclass_ranges, 
                     label_map, 
                     subclass_tuple, 
                     superclasses, 
                     hier.LEAF_NUM_TO_NAME)

Part 3: Loading in-built BREEDS Datasets
''''''''''''''''''''''''''''''''''''''''

Alternatively, we can directly use one of the datasets from our paper 
[STM20]_---namely ``Entity13``, ``Entity30``, ``Living17`` 
and ``Nonliving26``. Loading any of these datasets is relatively simple:

.. code-block:: python

  from robustness.tools.breeds_helpers import Living17
  ret = Living17(info_dir, split="rand")
  subclass_ranges, label_map, subclass_tuple, superclasses, _ = ret

You can then use a similar methodology to Part 2 above to probe
dataset information and create datasets and loaders.

