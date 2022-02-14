robustness package
==================
Install via ``pip``: ``pip install robustness``

Read the docs: https://robustness.readthedocs.io/en/latest/index.html

``robustness`` is a package we (students in the `MadryLab <http://madry-lab.ml>`_) created
to make training, evaluating, and exploring neural networks flexible and easy.
We use it in almost all of our projects (whether they involve
adversarial training or not!) and it will be a dependency in many of our
upcoming code releases. A few projects using the library include:

- `Code for "Learning Perceptually-Aligned Representations via Adversarial Robustness" <https://github.com/MadryLab/robust_representations>`_ (https://arxiv.org/abs/1906.00945) 
- `Code for
  "Image Synthesis with a Single (Robust) Classifier" <https://github.com/MadryLab/robustness_applications>`_ (https://arxiv.org/abs/1906.09453)
- `Code for
  "Do Adversarially Robust ImageNet Models Transfer Better?" <https://github.com/microsoft/robust-models-transfer>`_ (https://arxiv.org/abs/2007.08489)
- `Code for
  "BREEDS: Benchmarks for Subpopulation Shift"
  <https://github.com/MadryLab/BREEDS-Benchmarks>`_ (https://arxiv.org/abs/2008.04859)
- `Code for
  "Certified Patch Robustness via Smoothed Vision Transformers." <https://github.com/MadryLab/smoothed-vit>`_ (https://arxiv.org/abs/2110.07719)
- `Code for
  "Unadversarial Examples: Designing Objects for Robust Vision." <https://github.com/microsoft/unadversarial>`_ (https://arxiv.org/abs/2012.12235)

We
demonstrate how to use the library in a set of walkthroughs and our API
reference. Functionality provided by the library includes:

- Training and evaluating standard and robust models for a variety of
  datasets/architectures using a `CLI interface
  <https://robustness.readthedocs.io/en/latest/example_usage/cli_usage.html>`_. The library also provides support for adding
  `custom datasets <https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html#training-on-custom-datasets>`_ and `model architectures <https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html#training-with-custom-architectures>`_.

.. code-block:: bash

   python -m robustness.main --dataset cifar --data /path/to/cifar \
      --adv-train 0 --arch resnet18 --out-dir /logs/checkpoints/dir/

- Performing `input manipulation
  <https://robustness.readthedocs.io/en/latest/example_usage/input_space_manipulation.html>`_ using robust (or standard)
  models---this includes making adversarial examples, inverting representations,
  feature visualization, etc. The library offers a variety of optimization
  options (e.g. choice between real/estimated gradients, Fourier/pixel basis,
  custom loss functions etc.), and is easily extendable.

.. code-block:: python
   
   import torch as ch
   from robustness.datasets import CIFAR
   from robustness.model_utils import make_and_restore_model

   ds = CIFAR('/path/to/cifar')
   model, _ = make_and_restore_model(arch='resnet50', dataset=ds, 
                resume_path='/path/to/model', state_dict_path='model')
   model.eval()
   attack_kwargs = {
      'constraint': 'inf', # L-inf PGD 
      'eps': 0.05, # Epsilon constraint (L-inf norm)
      'step_size': 0.01, # Learning rate for PGD
      'iterations': 100, # Number of PGD steps
      'targeted': True # Targeted attack
      'custom_loss': None # Use default cross-entropy loss
   }

   _, test_loader = ds.make_loaders(workers=0, batch_size=10)
   im, label = next(iter(test_loader))
   target_label = (label + ch.randint_like(label, high=9)) % 10
   adv_out, adv_im = model(im, target_label, make_adv, **attack_kwargs)

- Importing ``robustness`` as a package, which allows for easy training of
  neural networks with support for custom loss functions, logging, data loading,
  and more! A good introduction can be found in our two-part walkthrough
  (`Part 1 <https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_1.html>`_, 
  `Part 2 <https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html>`_).

.. code-block:: python

   from robustness import model_utils, datasets, train, defaults
   from robustness.datasets import CIFAR

   # We use cox (http://github.com/MadryLab/cox) to log, store and analyze 
   # results. Read more at https//cox.readthedocs.io.
   from cox.utils import Parameters
   import cox.store

   # Hard-coded dataset, architecture, batch size, workers
   ds = CIFAR('/path/to/cifar')
   m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds)
   train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)

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

**Note**: ``robustness`` requires PyTorch to be installed with CUDA support.

Pretrained models
-----------------

Along with the training code, we release a number of pretrained models for
different datasets, norms and ε-train values. This list will be updated as
we release more or improved models. *Please cite this library (see bibtex
entry below) if you use these models in your research.* 

For each (model, ε-test) combination we evaluate 20-step and 100-step PGD with a
step size of `2.5 * ε-test / num_steps`. Since these two accuracies are quite 
close to each other, we do not consider more steps of PGD.
For each value of ε-test, we highlight the best robust accuracy achieved over
different ε-train in bold.

**Note #1**: We did not perform any hyperparameter tuning and simply used the same
hyperparameters as standard training. It is likely that exploring different 
training hyperparameters will increasse these robust accuracies by a few percent
points.

**Note #2**: The pytorch checkpoint (``.pt``) files below were saved with the following versions of PyTorch and Dill:

.. code-block::

  torch==1.1.0
  dill==0.2.9


CIFAR10 L2-norm (ResNet50):

- `ε = 0.0 <https://www.dropbox.com/s/yhpp4yws7sgi6lj/cifar_nat.pt?dl=0>`_ (standard training)
- `ε = 0.25 <https://www.dropbox.com/s/2qsp7pt6t7uo71w/cifar_l2_0_25.pt?dl=0>`_
- `ε = 0.5 <https://www.dropbox.com/s/1zazwjfzee7c8i4/cifar_l2_0_5.pt?dl=0>`_
- `ε = 1.0 <https://www.dropbox.com/s/s2x7thisiqxz095/cifar_l2_1_0.pt?dl=0>`_

+--------------+----------------+-----------------+---------------------+---------------------+
| CIFAR10 L2-robust accuracy                                                                  |
+--------------+----------------+-----------------+---------------------+---------------------+
|              | ε-train                                                                      |
+--------------+----------------+-----------------+---------------------+---------------------+
| ε-test       | 0.0            | 0.25            | 0.5                 | 1.0                 |
+==============+================+=================+=====================+=====================+
| 0.0          | **95.25% / -** | 92.77%  / -     | 90.83% / -          | 81.62% / -          |
+--------------+----------------+-----------------+---------------------+---------------------+
| 0.25         |  8.66% / 7.34% | 81.21% / 81.19% | **82.34% / 82.31%** | 75.53% / 75.53%     |
+--------------+----------------+-----------------+---------------------+---------------------+
| 0.5          |  0.28% / 0.14% | 62.30% / 62.13% | **70.17% / 70.11%** | 68.63% / 68.61%     |
+--------------+----------------+-----------------+---------------------+---------------------+
| 1.0          |  0.00% / 0.00% | 21.18% / 20.66% | 40.47% / 40.22%     | **52.72% / 52.61%** |
+--------------+----------------+-----------------+---------------------+---------------------+
| 2.0          |  0.00% / 0.00% |  0.58% /  0.46% |  5.23% /  4.97%     | **18.59% / 18.05%** |
+--------------+----------------+-----------------+---------------------+---------------------+

CIFAR10 Linf-norm (ResNet50):

- ε = 0.0 (PyTorch pre-trained)
- `ε = 8/255 <https://www.dropbox.com/s/c9qlt1lbdnu9tlo/cifar_linf_8.pt?dl=0>`_

+--------------+-----------------+---------------------+
| CIFAR10 Linf-robust accuracy                         |
+--------------+-----------------+---------------------+
|              | ε-train                               |
+--------------+-----------------+---------------------+
| ε-test       | 0 / 255         | 8 / 255             |
+==============+=================+=====================+
|  0 / 255     | **95.25% / -**  | 87.03%  / -         |
+--------------+-----------------+---------------------+
|  8 / 255     |  0.00% /  0.00% | **53.49% / 53.29%** |
+--------------+-----------------+---------------------+
| 16 / 255     |  0.00% /  0.00% | **18.13% / 17.62%** |
+--------------+-----------------+---------------------+

ImageNet L2-norm (ResNet50):

- ε = 0.0 (PyTorch pre-trained)
- `ε = 3.0 <https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0>`_

+--------------+-----------------+---------------------+
| ImageNet L2-robust accuracy                          |
+--------------+-----------------+---------------------+
|              | ε-train                               |
+--------------+-----------------+---------------------+
| ε-test       | 0.0             | 3.0                 |
+==============+=================+=====================+
| 0.0          | **76.13% / -**  | 57.90%  / -         |
+--------------+-----------------+---------------------+
| 0.5          |  3.35% /  2.98% | **54.42% / 54.42%** |
+--------------+-----------------+---------------------+
| 1.0          |  0.44% /  0.37% | **50.67% / 50.67%** |
+--------------+-----------------+---------------------+
| 2.0          |  0.16% /  0.14% | **43.04% / 43.02%** |
+--------------+-----------------+---------------------+
| 3.0          |  0.13% /  0.12% | **35.16% / 35.09%** |
+--------------+-----------------+---------------------+

ImageNet Linf-norm (ResNet50):

- ε = 0.0 (PyTorch pre-trained)
- `ε = 4 / 255 <https://www.dropbox.com/s/axfuary2w1cnyrg/imagenet_linf_4.pt?dl=0>`_
- `ε = 8 / 255 <https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=0>`_

+--------------+-----------------+---------------------+---------------------+
| ImageNet Linf-robust accuracy                                              |
+--------------+-----------------+---------------------+---------------------+
|              | ε-train                                                     |
+--------------+-----------------+---------------------+---------------------+
| ε-test       | 0.0             | 4 / 255             | 8 / 255             |
+==============+=================+=====================+=====================+
|  0 / 255     | **76.13% / -**  | 62.42%  / -         | 47.91%  / -         |
+--------------+-----------------+---------------------+---------------------+
|  4 / 255     | 0.04% / 0.03%   | **33.58% / 33.38%** |   33.06% / 33.03%   |
+--------------+-----------------+---------------------+---------------------+
|  8 / 255     | 0.01% / 0.01%   |   13.13% / 12.73%   | **19.63% / 19.52%** |
+--------------+-----------------+---------------------+---------------------+
| 16 / 255     | 0.01% / 0.01%   |    1.53% /  1.37%   |  **5.00% /  4.82%** |
+--------------+-----------------+---------------------+---------------------+

Citation
--------
If you use this library in your research, cite it as
follows:

.. code-block:: bibtex
   
   @misc{robustness,
      title={Robustness (Python Library)},
      author={Logan Engstrom and Andrew Ilyas and Hadi Salman and Shibani Santurkar and Dimitris Tsipras},
      year={2019},
      url={https://github.com/MadryLab/robustness}
   }

*(Have you used the package and found it useful? Let us know!)*.

Maintainers
-------------
- `Andrew Ilyas <https://twitter.com/andrew_ilyas>`_
- `Logan Engstrom <https://twitter.com/logan_engstrom>`_
- `Shibani Santurkar <https://twitter.com/ShibaniSan>`_
- `Dimitris Tsipras <https://twitter.com/tsiprasd>`_
- `Hadi Salman <https://twitter.com/hadisalmanX>`_

Contributors/Commiters
'''''''''''''''''''''''
- See `here <https://github.com/MadryLab/robustness/pulse>`_ 
