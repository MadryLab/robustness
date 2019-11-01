Creating a custom dataset by superclassing ImageNet 
====================================================
Often in both adversarial robustness research and otherwise, datasets with the
richness of ImageNet are desired, but without the added complexity of the 1000-way 
ILSVRC classification task. A common workaround is to "superclass" ImageNet,
that is, to define a new dataset that contains broad classes which each subsume
several of the original ImageNet classes.

In this document, we will discuss how to (a) load pre-packaged ImageNet-based
datasets that we've created, and (b) create new custom N-class subset of
ImageNet data by leveraging the WordNet hierarchy to build superclasses. The
robustness library provides functionality to do this via the
:class:`~robustness.datasets.CustomImageNet` and
:class:`~robustness.tools.imagenet_helpers.ImageNetHierarchy` classes. In this
walkthrough, we'll see how to use these classes to browse and use the WordNet
hierarchy to create custom ImageNet-based datasets.

.. raw:: html

   <i class="fa fa-play"></i> &nbsp;&nbsp; <a
   href="https://github.com/MadryLab/robustness/blob/master/notebooks/Superclassing%20ImageNet.ipynb">Download
   a Jupyter notebook</a> containing all the code from this walkthrough! <br />
   <br />

Requirements/Setup
''''''''''''''''''
To create custom ImageNet datasets, we need (a) the ImageNet dataset to be
downloaded and available in PyTorch-readable format, and (b) the files
``wordnet.is_a.txt``, ``words.txt`` and ``imagenet_class_index.json``, all
contained within the same directory (all of these files can be obtained from
`the ImageNet website <http://image-net.org/download-API>`_. 

Basic Usage: Loading Pre-Packaged ImageNet-based Datasets
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
To make things as easy as possible, we've compiled a list of large, but less
complex ImageNet-based datasets. These datasets can be loaded in their
unbalanced or balanced forms, where in the latter we truncate each class to have
the same number of images as the smallest class. We enumerate these datasets
below:

=============== ==================================================
  Dataset Name      Classes                                       
=============== ==================================================
living_9            | Dog (n02084071), Bird (n01503061),         
                    | Arthropod (n01767661), Reptile (n01661091), 
                    | Primate (n02469914), Fish (n02512053), 
                    | Feline (n02120997), Bovid (n02401031), 
                    | Amphibian (n01627424) 
mixed_10            | Dog (n02084071), Bird (n01503061),
                    | Insect (n02159955), Monkey (n02484322),
                    | Car (n02958343), Cat (n02120997),
                    | Truck (n04490091), Fruit (n13134947),
                    | Fungus (n12992868), Boat (n02858304)
mixed_13            | Dog (n02084071), Bird (n01503061),
                    | Insect (n02159955), Furniture (n03405725),
                    | Fish (n02512053), Monkey (n02484322),
                    | Car (n02958343), Cat (n02120997),
                    | Truck (n04490091), Fruit (n13134947),
                    | Fungus (n12992868), Boat (n02858304),
                    | Computer (n03082979)
geirhos_16          | Aircraft (n02686568), Bear (n02131653), 
                    | Bicycle (n02834778), Bird (n01503061), 
                    | Boat (n02858304), Bottle (n02876657), 
                    | Car (n02958343), Cat (n02121808), 
                    | Char (n03001627), Clock (n03046257), 
                    | Dog (n02084071), Elephant (n02503517), 
                    | Keyboard (n03614532), Knife (n03623556), 
                    | Oven (n03862676), Truck (n04490091),
big_12              | Dog (n02084071), Structure(n04341686),
                    | Bird (n01503061), Clothing (n03051540),
                    | Vehicle(n04576211), Reptile (n01661091),
                    | Carnivore (n02075296), Insect (n02159955),
                    | Instrument (n03800933), Food (n07555863),
                    | Furniture (n03405725), Primate (n02469914),
=============== ==================================================

Loading any of these datasets (for example, ``mixed_10``) is relatively simple:

.. code-block:: python

  from robustness import datasets
  from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
 
  in_hier = ImageNetHierarchy(in_path, in_info_path)
  superclass_wnid = common_superclass_wnid('mixed_10')
  class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)       

In the above, :samp:`in_path` should point to a folder with the ImageNet
dataset in ``train`` and ``val`` sub-folders; :samp:`in_info_path` should be the
path to the directory containing the aforementioned files (``wordnet.is_a.txt``,
``words.txt``, ``imagenet_class_index.json``).

We can then create a dataset and the corresponding data loader using:

.. code-block:: python

  custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
  train_loader, test_loader = custom_dataset.make_loaders(workers=num_workers, 
                                                          batch_size=batch_size)

You're all set! You can then use this :samp:`custom_dataset` and loaders
just as you would any other existing/custom dataset in the robustness 
library. For instance, you can visualize training set samples and their 
labels using:

.. code-block:: python

  from robustness.tools.vis_tools import show_image_row
  im, lab = next(iter(train_loader))
  show_image_row([im], tlist=[[label_map[int(k)] for k in lab]])

Advanced Usage (Making Custom Datasets) Part 1: Browsing the WordNet Hierarchy
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
The :class:`~robustness.tools.imagenet_helpers.ImageNetHierarchy` class allows
us to probe the WordNet hierarchy and create custom datasets with the desired
number of superclasses.  We first create an instance of the
``ImageNetHierarchy`` class:

.. code-block:: python

   from robustness.tools.imagenet_helpers import ImageNetHierarchy
   in_hier = ImageNetHierarchy(in_path, in_info_path)


Again, :samp:`in_path` should point to a folder with the ImageNet
dataset in ``train`` and ``val`` sub-folders; :samp:`in_info_path` should be the
path to the directory containing the aforementioned files (``wordnet.is_a.txt``,
``words.txt``, ``imagenet_class_index.json``).

We can now use the :samp:`in_hier` object to probe the ImageNet hierarchy.  The
``wnid_sorted`` attribute, for example, is an iterator over the WordNet IDs,
sorted by the number of descendents they have which are ImageNet classes:

.. code-block:: python

  for cnt, (wnid, ndesc_in, ndesc_total) in enumerate(in_hier.wnid_sorted):
      print(f"WordNet ID: {wnid}, Name: {in_hier.wnid_to_name[wnid]}, #ImageNet descendants: {ndesc_in}")

Given any WordNet ID, we can also enumerate all of its subclasses of a given
superclass using the ``in_hier.tree`` object and its related methods/attributes:

.. code-block:: python

  ancestor_wnid = 'n02120997'
  print(f"Superclass | WordNet ID: {ancestor_wnid}, Name: {in_hier.wnid_to_name[ancestor_wnid]}")

  for cnt, wnid in enumerate(in_hier.tree['n02120997'].descendants_all):
      print(f"Subclass | WordNet ID: {wnid}, Name: {in_hier.wnid_to_name[wnid]}")

We can filter these subclasses based on whether they correspond to ImageNet
classes using the ``in_wnids`` attribute:

.. code-block:: python

  ancestor_wnid = 'n02120997'
  print(f"Superclass | WordNet ID: {ancestor_wnid}, Name: {in_hier.wnid_to_name[ancestor_wnid]}")
  for cnt, wnid in enumerate(in_hier.tree[ancestor_wnid].descendants_all):
      if wnid in in_hier.in_wnids:
          print(f"ImageNet subclass | WordNet ID: {wnid}, Name: {in_hier.wnid_to_name[wnid]}")


Advanced Usage (Making Custom Datasets) Part 2: Making the Datasets
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
To create a dataset with the desired number of superclasses we use 
the :meth:`~robustness.tools.imagenet_helpers.ImageNetHierarchy.get_superclasses` function, 
which takes in a desired number of superclasses :samp:`n_classes`, an
(optional) WordNet ID :samp:`ancestor_wnid` that allows us to fix a common
WordNet ancestor for all the classes in our new dataset, and an optional boolean 
:samp:`balanced` to get a balanced dataset (where each superclass 
has the same number of ImageNet subclasses).
(see :py:meth:`the docstring 
<robustness.tools.imagenet_helpers.ImageNetHierarchy.get_superclasses>` for
more details). 

.. code-block:: python

   superclass_wnid, class_ranges, label_map = in_hier.get_superclasses(n_classes, 
                                                ancestor_wnid=ancestor_wnid,
                                                balanced=balanced)                                      

This method returns WordNet IDs of chosen superclasses 
:samp:`superclass_wnid`, sets of ImageNet subclasses to group together
for each of the superclasses :samp:`class_ranges`, and a mapping from 
superclass number to its human-interpretable description :samp:`label_map`.

You can also directly provide a list of superclass WordNet IDs :samp:`ancestor_wnid`
that you would like to use to build a custom dataset. For instance, some sample superclass 
groupings can be found in 
py:meth:`~robustness.tools.imagenet_helpers.ImageNetHierarchy.common_superclass_wnid`.

Once a list of WordNet IDs has been acquired (whether through the method
described here or just manually), we can use the method presented at the
beginning of this article to load the corresponding dataset:

.. code-block:: python

  custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
  train_loader, test_loader = custom_dataset.make_loaders(workers=num_workers, 
                                                          batch_size=batch_size)
