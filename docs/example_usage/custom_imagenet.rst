Creating a custom dataset by superclassing ImageNet 
====================================================

.. raw:: html

In this document, we will discuss how to create a custom N class 
subset of ImageNet data by leveraging the WordNet hierarchy to 
build superclasses. The robustness library provides functionality
to do this via :py:class:`~robustness.datasets.CustomImageNet`
and :py:class:`~robustness.tools.imagenet_helpers.ImageNetHierarchy`.
For example:

1. To probe the WordNet hierarchy and create the desired
number of superclasses, you could do:

.. code-block:: python

   from robustness.tools.imagenet_helpers import ImageNetHierarchy

   in_hier = ImageNetHierarchy(in_path, 
                               in_info_path)


Here, :samp:`in_path` should point to a folder with the ImageNet
dataset in ``train`` and ``val`` sub-folders. :samp:`in_info_path`
should contain the files "wordnet.is_a.txt", "words.txt" and 
"imagenet_class_index.json" which can be obtained from 
`here <http://image-net.org/download-API>`_. 

You could use the object :samp:`in_hier` to probe the ImageNet hierarchy.
For example, to see the top 10 synsets in WordNet that have the highest
number of ImageNet synsets as descendants, you could do:

.. code-block:: python

  for cnt, (wnid, ndesc_in, ndesc_total) in enumerate(in_hier.wnid_sorted):
      if cnt < 10: 
          print(f"WordNet ID: {wnid}, Name: {in_hier.wnid_to_name[wnid]}, #ImageNet descendants: {ndesc_in}")

To enumerate all subclasses of a given superclass:

.. code-block:: python

  ancestor_wnid = 'n02120997'
  print(f"Superclass | WordNet ID: {ancestor_wnid}, Name: {in_hier.wnid_to_name[ancestor_wnid]}")

  for cnt, wnid in enumerate(in_hier.tree['n02120997'].descendants_all):
      print(f"Subclass | WordNet ID: {wnid}, Name: {in_hier.wnid_to_name[wnid]}")

To enumerate subclasses of a superclass that are a part of ImageNet:

.. code-block:: python

  ancestor_wnid = 'n02120997'
  print(f"Superclass | WordNet ID: {ancestor_wnid}, Name: {in_hier.wnid_to_name[ancestor_wnid]}")
  for cnt, wnid in enumerate(in_hier.tree[ancestor_wnid].descendants_all):
      if wnid in in_hier.in_wnids:
          print(f"ImageNet subclass | WordNet ID: {wnid}, Name: {in_hier.wnid_to_name[wnid]}")


2. To create the desired number of superclass we use 
py:meth:`~robustness.tools.imagenet_helpers.ImageNetHierarchy.get_superclasses`, 
which takes in the desired number of superclasses :samp:`n_classes`, an
optional WordNet ID :samp:`ancestor_wnid` to pick superclasses that share a 
common ancestor in the WordNet hierarchy, an optional set of WordNet IDs of 
superclasses which should not be further sub-classes :samp:`superclass_lowest`
(if encountered), and an optional boolean 
:samp:`balanced` to get a balanced dataset (where each superclass 
has the same number of ImageNet subclasses).
(see :py:meth:`the docstring 
<robustness.tools.imagenet_helpers.ImageNetHierarchy.get_superclasses>` for
more details). 

.. code-block:: python

   superclass_wnid, class_ranges, label_map = in_hier.get_superclasses(n_classes, 
                                                ancestor_wnid=ancestor_wnid,
                                                superclass_lowest=superclass_lowest,
                                                balanced=balanced)                                      

This method returns WordNet IDs of chosen superclasses 
:samp:`superclass_wnid`, sets of ImageNet subclasses to group together
for each of the superclasses :samp:`class_ranges`, and a mapping from 
superclass number to its human-interpretable description :samp:`label_map`.


You could also directly provide a list of superclass WordNet IDs :samp:`ancestor_wnid`
that you would like to use to build a custom dataset. For instance, some sample superclass 
groupings can be found in 
py:meth:`~robustness.tools.imagenet_helpers.ImageNetHierarchy.common_superclass_wnid`.


.. code-block:: python

  from robustness.tools.imagenet_helpers import common_superclass_wnid

  superclass_wnid = common_superclass_wnid('mixed_10')
  class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, 
                                                   balanced=balanced)       



3. We can then create a dataset and the corresponding data loader
using:

.. code-block:: python

  from robustness import datasets

  custom_dataset = datasets.CustomImageNet(in_path, 
                                           class_ranges)

  train_loader, test_loader = custom_dataset.make_loaders(workers=num_workers, 
                                                          batch_size=batch_size)

You're all set! You can then use this :samp:`custom_dataset` and loaders
just as you would any other existing/custom dataset in the robustness 
library. For instance, you could visualize training set samples and their 
labels using:

.. code-block:: python

  from robustness.tools.vis_tools import show_image_row

  iterator = enumerate(train_loader)

  _, (im, lab) = next(iterator)

  show_image_row([im], 
                 tlist=[[label_map[int(k)] for k in lab]])
