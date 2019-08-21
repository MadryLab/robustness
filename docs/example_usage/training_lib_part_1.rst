Using robustness as a general training library (Part 1: Getting started)
========================================================================
In the other walkthroughs, we've demonstrated how to use :samp:`robustness` as a
:doc:`command line tool for training and evaluating models <cli_usage>`, and how
to use it as a library for :doc:`input manipulation <input_space_manipulation>`. 
Here, we'll demonstrate how :samp:`robustness` can be used a general library for
experimenting with neural network training. We've found the library has saved us
a tremendous amount of time both writing boilerplate code and making custom
modifications to the training process (one of the primary goals of the library
is to make such modifications simple).

This walkthrough will be split into two parts: in the first part (this one),
we'll show how to get started with the :samp:`robustness` library, and go
through the process of making a ``main.py`` file for training neural networks.
In the :doc:`second part <../example_usage/training_lib_part_2>`, we'll show how to customize the training
process through custom loss functions, architectures, datasets, logging, and
more.

.. raw:: html 
   
   <i class="fa fa-file"></i> &nbsp; You can follow along using <a
   href="https://github.com/MadryLab/robustness/blob/master/robustness/main.py">the
   source</a> of robustness.main <br /> <br />

   <i class="fa fa-play"></i> &nbsp;&nbsp; You can also <a
   href="https://github.com/MadryLab/robustness/blob/master/notebooks/Using%20robustness%20as%20a%20library.ipynb">download
   a Jupyter notebook</a> containing code from this walkthrough and the next! <br />
   <br />

Step 1: Imports
----------------
Our goal in this tutorial will be to make a python file that works nearly
identically to the robustness :doc:`Command-line tool
<../example_usage/cli_usage>`. That is, a user
will be able to call ``python main.py [--arg value ...]`` to train a standard or
robust model. We'll start by importing the necessary modules from the package:

.. code-block:: python
   
   from robustness.datasets import DATASETS 
   from robustness.model_utils import make_and_restore_model
   from robustness.train import train_model
   from robustness.defaults import check_and_fill_args
   from robustness.tools import constants, helpers
   from robustness import defaults

To make life easier, we use `cox <https://github.com/MadryLab/cox>`_ (a super
lightweight python logging library) for logging:

.. code-block:: python
   
   from cox import utils 
   from cox import store

Finally, we'll also need the following external imports:

.. code-block:: python

   import torch as ch
   from argparse import ArgumentParser
   import os

Step 2: Dealing with arguments
-------------------------------
In this first step, we'll set up an ``args`` object which has all the parameters
we need to train our model. In Step 2.1 we'll show how to use ``argparse`` to
accept user input for specifying parameters via command line; in Step 2.2 we
show how to sanity-check the ``args`` object and fill in reasonable defaults.

Step 2.1: Setting up command-line args
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The first real step in making our main file is setting up an
``argparse.ArgumentParser`` object to parse command-line arguments for us. (If
you are not familiar with the python `argparses
<https://docs.python.org/3/library/argparse.html>`_ module, we recommend looking
there first). Note that if you're not interested in accepting command-line input
for arguments via argparse, you can skip to Step 2.2.

The ``robustness`` package provides the :mod:`robustness.defaults` module to
make dealing with arguments less painful. In particular, the properties
:attr:`robustness.defaults.TRAINING_ARGS`, :attr:`robustness.defaults.PGD_ARGS`,
and :attr:`robustness.defaults.MODEL_LOADER_ARGS`, contain all of the arguments
(along with default values) needed for training models:

- :attr:`~robustness.defaults.TRAINING_ARGS` has all of the model training
  arguments, like learning rate, momentum, weight decay, learning rate schedule,
  etc.
- :attr:`~robustness.defaults.PGD_ARGS` has all of the arguments needed only for
  adversarial training, like number of PGD steps, perturbation budget, type of
  norm constraint, etc.
- :attr:`~robustness.defaults.MODEL_LOADER_ARGS` has all of the arguments for
  instantiating the model and data loaders: dataset, path to dataset, batch
  size, number of workers, etc.

You can take a look at the documentation of :mod:`robustness.defaults` to
learn more about how these attributes are set up and see exactly which arguments
they contain and with what defaults, as well as which arguments are required. The important thing is that the
``robustness`` package provides the function
:meth:`robustness.defaults.add_args_to_parser` which takes in an arguments
object like the above, and an ``argparse`` parser, and adds the arguments to the
parser:

.. code-block:: python

   parser = ArgumentParser()
   parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
   parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
   parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)
   # Note that we can add whatever extra arguments we want to the parser here
   args = parser.parse_args()

**Important note:** Even though the arguments objects do specify defaults for
the arguments, these defaults are **not** given to the parser directly. More on
this in Step 2.2.

If you don't want to use ``argparse`` and already know the values you want to
use for the parameters, you can look at the :mod:`robustness.defaults`
documentation, and hard-code the desired arguments as follows:

.. code-block:: python

   # Hard-coded base parameters
   train_kwargs = {
       'out_dir': "train_out",
       'adv_train': 1,
       'constraint': '2',
       'eps': 0.5,
       'attack_lr': 1.5,
       'attack_steps': 20
   }

   # utils.Parameters is just an object wrapper for dicts implementing
   # getattr and settattr 
   train_args = utils.Parameters(train_kwargs)

Step 2.2: Sanity checks and defaults
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We generally found the ``ArgumentParser`` defaults to be too restrictive for our
applications, and we also wanted to be able to fill in argument defaults even
when we were not using ``ArgumentParser``. Thus, we fill in default arguments
separately via the :meth:`robustness.defaults.check_and_fill_args` function.
This function takes in the ``args`` Namespace object (basically anything
exposing ``setattr`` and ``getattr`` functions), the same ``ARGS`` attributes
discussed above, and a dataset class (used for filling in per-dataset defaults).
The function fills in the defaults it has, and then throws an error if a
required argument is missing:

.. code-block:: python

   assert args.dataset is not None, "Must provide a dataset"
   ds_class = DATASETS[args.dataset]

   args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)
   if args.adv_train or args.adv_eval:
     args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)
   args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)

Note that the :meth:`~robustness.defaults.check_and_fill_args` function is
totally independent of ``argparse``, and can be used even when you don't want to
support command-line arguments. It's a really useful way of sanity checking the
``args`` object to make sure that there aren't any missing or misspecified arguments.

Step 3: Creating the model, dataset, and loader
------------------------------------------------
The next step is to create the model, dataset, and data loader. We start by
creating the dataset and loaders as follows:

.. code-block:: python

   # Load up the dataset
   data_path = os.path.expandvars(args.data)
   dataset = DATASETS[args.dataset](data_path)

   # Make the data loaders
   train_loader, val_loader = dataset.make_loaders(args.workers,
                 args.batch_size, data_aug=bool(args.data_aug))

   # Prefetches data to improve performance
   train_loader = helpers.DataPrefetcher(train_loader)
   val_loader = helpers.DataPrefetcher(val_loader)

We can now create the model by using the
:meth:`robustness.model_utils.make_and_restore_model` function. This function is
used for both creating new models, or (if a ``resume_path`` if passed) loading
previously saved models.

.. code-block:: python

    model, _ = make_and_restore_model(arch=args.arch, dataset=dataset)
   
Step 4: Training the model
---------------------------
Finally, we create a `cox Store <https://cox.readthedocs.io/en/latest/cox.store.html>`_ for saving the results of the
training, and call :meth:`robustness.train.train_model` to begin training:

.. code-block:: python

    # Create the cox store, and save the arguments in a table
    store = store.Store(args.out_dir, args.exp_name)
    args_dict = args.as_dict() if isinstance(args, utils.Parameters) else vars(args)
    schema = store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)

    model = train_model(args, model, (train_loader, val_loader), store=store)

