Using robustness as a general training library (Part 2: Customizing training)
==============================================================================

.. raw:: html

   <i class="fa fa-play"></i> &nbsp;&nbsp; <a
   href="https://github.com/MadryLab/robustness/blob/master/notebooks/Using%20robustness%20as%20a%20library.ipynb">Download
   a Jupyter notebook</a> containing all the code from this walkthrough! <br />
   <br />

In this document, we'll continue our walk through using robustness as a library.
In the :doc:`first part <../example_usage/training_lib_part_1>`, we made a ``main.py`` file that trains a model
given user-specified parameters. For this part of the walkthrough, we'll
continue from that ``main.py`` file. You can also start with a copy the `source
<https://github.com/MadryLab/robustness/robustness/main.py>`_ of
:mod:`robustness.main`, or (if you don't want the full flexibility of all of
those arguments) the following bare-bones :samp:`main.py` file suffices for
training an adversarially robust CIFAR classifier with a fixed set of
parameters:

.. code-block:: python

   from robustness import model_utils, datasets, train, defaults
   from robustness.datasets import CIFAR
   import torch as ch

   # We use cox (http://github.com/MadryLab/cox) to log, store and analyze 
   # results. Read more at https//cox.readthedocs.io.
   from cox.utils import Parameters
   import cox.store

   # Hard-coded dataset, architecture, batch size, workers
   ds = CIFAR('/tmp/')
   m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds)
   train_loader, val_loader = ds.make_loaders(batch_size=BATCH_SIZE, workers=NUM_WORKERS)

   # Create a cox store for logging
   out_store = cox.store.Store(OUT_DIR)

   # Hard-coded base parameters
   train_kwargs = {
       'out_dir': "train_out",
       'adv_train': 1,
       'constraint': '2',
       'eps': 0.5,
       'attack_lr': 1.5,
       'attack_steps': 20
   }
   train_args = Parameters(train_kwargs)

   # Fill whatever parameters are missing from the defaults
   train_args = defaults.check_and_fill_args(train_args,
                           defaults.TRAINING_ARGS, CIFAR)
   train_args = defaults.check_and_fill_args(train_args,
                           defaults.PGD_ARGS, CIFAR)

   # Train a model
   train.train_model(train_args, m, (train_loader, val_loader), store=out_store)

The following sections will demonstrate how to customize training in a variety
of ways.

Training networks with custom loss functions
--------------------------------------------
By default, training uses the cross-entropy loss; however, we can easily change
this by specifying a custom training loss and a custom adversary loss. For
example, suppose that instead of just computing the cross-entropy loss, we're
going to try an experimental new training loss that multiplies a random 50%
of the logits by 10. (*Note that this is just for illustrative purposes---in
practice this is a terrible idea*.)

We can implement this crazy loss function as a training criterion and a
corresponding adversary loss. Recall that as discussed in the
:meth:`robustness.train.train_model` docstring, the train loss takes in
:samp:`logits,targets` and returns a scalar, whereas the adversary loss takes in
:samp:`model,inputs,targets` and returns a vector (not averaged along the
batch) as well as the output.

.. code-block:: python

   train_crit = ch.nn.CrossEntropyLoss()
   def custom_train_loss(logits, targ):
       probs = ch.ones_like(logits) * 0.5
       logits_to_multiply = ch.bernoulli(probs) * 9 + 1
       return train_crit(logits_to_multiply * logits, targ)
       
   adv_crit = ch.nn.CrossEntropyLoss(reduction='none').cuda()
   def custom_adv_loss(model, inp, targ):
       logits = model(inp)
       probs = ch.ones_like(logits) * 0.5
       logits_to_multiply = ch.bernoulli(probs) * 9 + 1
       new_logits = logits_to_multiply * logits
       return adv_crit(new_logits, targ), new_logits

   train_args.custom_train_loss = custom_train_loss
   train_args.custom_adv_loss = custom_adv_loss

Adding these few lines right before calling of
:meth:`~robustness.train.train_model`
suffices for training our network robustly with this custom loss.

As of the latest version of ``robustness``, you can now also supply a custom
function for computing accuracy using the ``custom_accuracy`` flag. This should
be a function that takes in the model output and the target labels, and returns
a tuple of ``(top1, top5)`` accuracies (feel free to make the second element
``float('nan')`` if there's only one accuracy metric you want to display). Here
is an example:

.. code-block:: python

    def custom_acc_func(out, targ):
        # Calculate top1 and top5 accuracy for this batch here
        return 100., float('nan') # Return (top1, top5)
    
    train_args.custom_accuracy = custom_acc_func


.. _using-custom-loaders:

Training networks with custom data loaders
-------------------------------------------
Another aspect of the training we can customize is data loading, through two
utilities for modifying dataloaders called
:meth:`robustness.loaders.TransformedLoader` and
:class:`robustness.loaders.LambdaLoader`. To see how they work, we're going to
consider two variations on our training: (a) training with label noise, and (b)
training with random labels.

Using LambdaLoader to train with label noise
""""""""""""""""""""""""""""""""""""""""""""
:class:`~robustness.laoders.LambdaLoader` works by modifying the output of a
data loader *in real-time*, i.e. it applies a fixed function to the output of a
loader. This makes it well-suited to, e.g., custom data augmentation,
input/label noise, or other applications where randomness across batches is
needed. To demonstrate its usage, we're going to add label noise to our training
setup. To do this, all we need to do is define a function which takes in a batch
of inputs and labels, and returns the same batch but with label noise added in.
For example:

.. code-block:: python

   from robustness.loaders import LambdaLoader

   def label_noiser(ims, labels):
       label_noise = ch.randint_like(labels, high=9)
       probs = ch.ones_like(label_noise) * 0.1
       labels_to_noise = ch.bernoulli(probs.float()).long()
       new_labels = (labels + label_noise * labels_to_noise) % 10
       return ims, new_labels

   train_loader = LambdaLoader(train_loader, label_noiser)

Note that LamdaLoader is quite general---any function that takes in :samp:`ims,
labels` and outputs :samp:`ims, labels` of the same shape can be put in place of
:samp:`label_noiser` above.

Using TransformedLoader to train with random labels
"""""""""""""""""""""""""""""""""""""""""""""""""""
In contrast to :class:`~robustness.loaders.LambdaLoader`,
:meth:`~robustness.loaders.TransformedLoader` is a data loader transformation
that is applied *once* at the beginning of training (this makes it better suited
to deterministic transformations to inputs or labels). Unfortunately, the
implementation of TransformedLoader currently loads the entire dataset into
memory, so it only reliably works on small datasets (e.g. CIFAR). This will be 
fixed in a future version of the library. To demonstrate its usage, we will use 
it to randomize labels for the training set. (Recall that when we usually train
using random labels, we perform the label assignment only once, prior to 
training.) To do this, all we need to do is define a function which takes in a
batch of inputs and labels, and returns the same batch, but with random labels
instead. For example:

.. code-block:: python

   from robustness.loaders import TransformedLoader
   from robustness.data_augmentation import TRAIN_TRANSFORMS_DEFAULT

   def make_rand_labels(ims, targs):
       new_targs = ch.randint(0, high=10, size=targs.shape).long()      
       return ims, new_targs

   train_loader_transformed = TransformedLoader(train_loader,
                                              make_rand_labels,
                                              TRAIN_TRANSFORMS_DEFAULT(32), 
                                              workers=NUM_WORKERS, 
                                              batch_size=BATCH_SIZE,
                                              do_tqdm=True)

Here, we start with a :samp:`train_loader` without data augmentation, to get access 
to the actual image-label pairs from the training set. We then transform each input
by assigning an image a random label instead. Moreover, we also support applying other
transforms in *real-time* (such as data augmentation) during the creation of the 
transformed dataset using :samp:`train_loader_transformed` (e.g., 
:samp:`TRAIN_TRANSFORMS(32)` here).

Note that TransformedLoader is quite general---any function that takes in :samp:`ims,
labels` and outputs :samp:`ims, labels` of the same shape can be put in place of
:samp:`rand_label_transform` above. 

Training networks with custom logging
-------------------------------------
The :samp:`robustness` library also supports training with custom logging
functionality. When calling :meth:`~robustness.train.train_model`, the user can
specify "hooks," functions that will be called by the training process every
iteration or every epoch. Here, we'll demonstrate this functionality using a
logging function that measures the norm of the network parameters (by treating
them as a single vector). We will modify/augment the :samp:`main.py` code
described above:

.. code-block:: python

   from torch.nn.utils import parameters_to_vector as flatten

   def log_norm(mod, log_info):
      curr_params = flatten(mod.parameters())
      log_info_custom = { 'epoch': log_info['epoch'],
                           'weight_norm': ch.norm(curr_params).detach().cpu().numpy() }
      out_store['custom'].append_row(log_info_custom)

We now create a custom `cox <http://github.com/MadryLab/cox>`_ store that we'll
hold our results in (:samp:`cox` is our super-lightweight library for storing
and analyzing experimental results, you can read the docs `here
<https://cox.readthedocs.io>`_).

.. code-block:: python

    CUSTOM_SCHEMA = {'epoch': int, 'weight_norm': float }
    out_store.add_table('custom', CUSTOM_SCHEMA)

We will then modify the :samp:`train_args` to incorporate this function into 
the logging done per epoch/iteration. If we want to log the norm of the weights
every epoch, we can do:

.. code-block:: python

   train_args.epoch_hook = log_norm

If we want to perform the logging every iteration, we need to make the
following modifications:

.. code-block:: python

  CUSTOM_SCHEMA = {'iteration': int, 'weight_norm': float}
  out_store.add_table('custom', CUSTOM_SCHEMA)

   def log_norm(mod, it, loop_type, inp, targ):
      if loop_type == 'train':
         curr_params = flatten(mod.parameters())
         log_info_custom = { 'iteration': it,
                        'weight_norm': ch.norm(curr_params).detach().cpu().numpy() }
         out_store['custom'].append_row(log_info_custom)

   train_args.iteration_hook = log_norm

The arguments taken by the iteration hook differ from those taken by
the epoch hook: the former takes a model, iteration number, loop_type, current
input batch, and current target batch. The latter takes only the model and a
dictionary called log_info containing all of the normally logged statistics as
in train.py.

Note that the custom logging functionality provided by the robustness library is
quite general---any function that takes the appropriate input arguments can be
used in place of :samp:`log_norm` above.

.. _using-custom-datasets:

Training on custom datasets
---------------------------
The robustness library by default includes most common datasets: ImageNet,
Restricted-ImageNet, CIFAR, CINIC, and A2B. That said, it is rather
straightforward to add your own dataset. 

1. Subclass the :py:class:`~robustness.datasets.DataSet` class from
   :mod:`robustness.datasets`. This means implementing 
   :py:meth:`~robustness.datasets.DataSet.__init__`
   and :py:meth:`~robustness.datasets.DataSet.get_model` functions.
2. In :samp:`__init__()`, all that is required is to call
   :samp:`super(NewClass, self).__init__` with the appropriate arguments,
   found in :py:class:`the docstring <robustness.datasets.DataSet>` and
   duplicated below:
   
   Arguments:
      - Dataset name (e.g. :samp:`imagenet`).
      - Dataset path (if your desired dataset is in the list of already
        implemented datasets in torchvision.datasets, pass the appropriate
        location, otherwise make this an argument of your subclassed
        :samp:`__init__` function.

     Named arguments (all required):
      - :samp:`num_classes`, the number of classes in the dataset
      - :samp:`mean`, the mean to normalize the dataset with
      - :samp:`std`, the standard deviation to normalize the dataset with
      - :samp:`custom_class`, the `torchvision.models` class corresponding
        to the dataset, if it exists (otherwise :samp:`None`)
      - :samp:`label_mapping`, a dictionary mapping from class numbers to
        human-interpretable class names (can be :samp:`None`)
      - :samp:`transform_train`, instance of :samp:`torchvision.transforms`
        to apply to the training images from the dataset
      - :samp:`transform_test`, instance of :samp:`torchvision.transforms`
        to apply to the validation images from the dataset
3. In :py:meth:`~robustness.datasets.DataSet.get_model`, implement a
   function which takes in an architecture name :samp:`arch` and boolean
   :samp:`pretrained`, and returns a PyTorch model (nn.Module) (see
   :py:meth:`the docstring <robustness.datasets.DataSet.get_model>` for
   more details). This will probably entail just using something like::
      
      from robustness import imagenet_models # or cifar_models
      assert not pretrained, "pretrained only available for ImageNet"
      return imagenet_models.__dict__[arch](num_classes=self.num_classes)
      # replace "models" with "cifar_models" in the above if the 
      # image size is less than [224, 224, 3]

You're all set! You can create an instance of your dataset and a
corresponding model with:::

   from robustness.datasets import MyNewDataSet
   from robustness.model_utils import make_and_restore_model
   ds = MyNewDataSet('/path/to/dataset/')
   model, _ = make_and_restore_model(arch='resnet50', dataset=ds)

**Note: if you also want to be able to use your dataset with the**
:doc:`command-line tool <../example_usage/cli_usage>`, **you'll need to clone the repository and pip
install it locally, after also following these extra steps**: 

4. Add an entry to :attr:`robustness.datasets.DATASETS` dictionary for your
   dataset.
5. If you want to be able to train a robust model on your dataset, add it
   to the :attr:`~robustness.main.DATASET_TO_CONFIG` dictionary in `main.py` and
   create a config file in the same manner as for the other datasets.

.. _using-custom-archs:

Training with custom architectures
----------------------------------
Currently the robustness library supports a few common architectures. The
models are split between two folders: :samp:`cifar_models` for
architectures that handle CIFAR-size (i.e. 32x32x3) images, and
:samp:`imagenet_models` for models that require larger images (e.g.
224x224x3). It is possible to add architectures to either of these
folders, but to make them fully compatible with the :samp:`robustness`
library requires a few extra steps. 

We'll go through an example of how to add a simple one-hidden-layer MLP
architecture for CIFAR:

0. Let's set up our imports and instantiate the dataset:

   .. code-block:: python
      
      from torch import nn
      from robustness.model_utils import make_and_restore_model
      from robustness.datasets import CIFAR

      ds = CIFAR('/path/to/cifar')

1. Implement and create an instance of your model:
   
   .. code-block:: python

      class MLP(nn.Module):
         # Must implement the num_classes argument
         def __init__(self, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(32*32*3, 1000)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(1000, num_classes)

         def forward(self, x, *args, **kwargs):
            out = x.view(x.shape[0], -1)
            out = self.fc1(out)
            out = self.relu1(out)
            return self.fc2(out)

      model = MLP(num_classes=10)

2. Call :meth:`robustness.model_utils.make_and_restore_model`, but this time
   feed in ``model`` instead of a string with the architecture name:

   .. code-block:: python

      model, _ = make_and_restore_model(arch=model, dataset=ds)

3. (If all you want to do with this architecture is training a
   model, **you can skip this step**). In order to make it fully compatible
   with the robustness library, the :`forward` function of our architecture
   must support the following three (boolean) arguments:

   - :samp:`with_latent` : If this option is given, :samp:`forward` should
     return the output of the second-last layer along with the logits.
   - :samp:`fake_relu` :  If this option is given, replace the ReLU just
     after the second-last layer with a :samp:`custom_modules.FakeReLUM`,
     which is a ReLU on the forwards pass and identity on the backwards
     pass.
   - :samp:`no_relu` : If this option is given, then :samp:`with_latent`
     should return the *pre-ReLU* activations of the second-last layer.

   These options are usually actually quite simple to implement:

   .. code-block:: python

      from robustness.imagenet_models import custom_modules

      class MLP(nn.Module):
         # Must implement the num_classes argument
         def __init__(self, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(32*32*3, 1000)
            self.relu1 = nn.ReLU()
            self.fake_relu1 = custom_modules.FakeReLUM()
            self.fc2 = nn.Linear(1000, num_classes)

         def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
            out = x.view(x.shape[0], -1)
            pre_relu = self.fc1(out)
            out = self.fake_relu1(pre_relu) if fake_relu else self.relu1(pre_relu)
            final = self.fc2(out)
            if with_latent:
               return (final, pre_relu) if no_relu else (final, out)
            return final

That's it! Now, just like for custom datasets, if you want these architectures
to be available via the :doc:`command line tool <../example_usage/cli_usage>`,
you'll have to clone the ``robustness`` repository and pip install it locally.
You'll also have to do the following:

4. Put the declaration of the ``MLP`` class into its own ``mlp.py`` file, and
   add this file to the :samp:`cifar_models` folder

3. In :samp:`cifar_models/__init__.py`, add the line::
   
      from .mlp import MLP

4. The new architecture is now available as::

      from robustness.model_utils import make_and_restore_model
      from robustness.datasets import CIFAR
      ds = CIFAR('/path/to/cifar')
      model, _ = make_and_restore_model(arch='MLP', dataset=ds)

   and via the command line option ``--arch MLP``.
