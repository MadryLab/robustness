robustness package
==================

.. raw:: html
   
   <i class="fa fa-github"></i> View on <a
   href="https://github.com/MadryLab/robustness">GitHub</a> <br /> <br />

Install via ``pip``: ``pip install robustness``

:samp:`robustness` is a package we (students in the `MadryLab <http://madry-lab.ml>`_) created
to make training, evaluating, and exploring neural networks flexible and easy.
We use it in almost all of our projects (whether they involve
adversarial training or not!) and it will be a dependency in many of our
upcoming code releases. A few projects using the library include:

- `Code <https://github.com/MadryLab/robust_representations>`_  for
  "Learning Perceptually-Aligned Representations via Adversarial Robustness"
  [EIS+19]_ 
- `Code <https://github.com/MadryLab/robustness_applications>`_ for
  "Image Synthesis with a Single (Robust) Classifier" [STE+19]_
- `Code <https://github.com/microsoft/robust-models-transfer>`_ for
  "Do Adversarially Robust ImageNet Models Transfer Better?" [SIE+20]_
- `Code <https://github.com/MadryLab/BREEDS-Benchmarks>`_ for
  "BREEDS: Benchmarks for Subpopulation Shift" [STM20]_
- `Code <https://github.com/microsoft/unadversarial>`_ for
  "Unadversarial Examples: Designing Objects for Robust Vision" [SIE+21]_
- `Code <https://github.com/MadryLab/smoothed-vit>`_ for
  "Certified Patch Robustness via Smoothed Vision Transformers" [SJW+21]_

We demonstrate how to use the library in a set of walkthroughs and our API
reference. Functionality provided by the library includes:

- Training and evaluating standard and robust models for a variety of
  datasets/architectures using a :doc:`CLI interface
  <example_usage/cli_usage>`. The library also provides support for adding
  :ref:`custom datasets <using-custom-datasets>` and :ref:`model architectures
  <using-custom-archs>`.

.. code-block:: bash

   python -m robustness.main --dataset cifar --data /path/to/cifar \
      --adv-train 0 --arch resnet18 --out-dir /logs/checkpoints/dir/

- Performing :doc:`input manipulation
  <example_usage/input_space_manipulation>` using robust (or standard)
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

- Importing :samp:`robustness` as a package, which allows for easy training of
  neural networks with support for custom loss functions, logging, data loading,
  and more! A good introduction can be found in our two-part walkthrough
  (:doc:`Part 1 <example_usage/training_lib_part_1>`, :doc:`Part 2
  <example_usage/training_lib_part_2>`).

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

Walkthroughs
------------

.. toctree::
   example_usage/cli_usage
   example_usage/input_space_manipulation
   example_usage/training_lib_part_1
   example_usage/training_lib_part_2
   example_usage/custom_imagenet
   example_usage/breeds_datasets
   example_usage/changelog

API Reference
-------------

We provide an API reference where we discuss the role of each module and
provide extensive documentation.

.. toctree::
   api


Contributors
-------------
- `Andrew Ilyas <https://twitter.com/andrew_ilyas>`_
- `Logan Engstrom <https://twitter.com/logan_engstrom>`_
- `Shibani Santurkar <https://twitter.com/ShibaniSan>`_
- `Dimitris Tsipras <https://twitter.com/tsiprasd>`_
- `Hadi Salman <https://twitter.com/hadisalmanX>`_

.. [EIS+19] Engstrom L., Ilyas A., Santurkar S., Tsipras D., Tran B., Madry A. (2019). Learning Perceptually-Aligned Representations via Adversarial Robustness. arXiv, arXiv:1906.00945 

.. [STE+19] Santurkar S., Tsipras D., Tran B., Ilyas A., Engstrom L., Madry A. (2019). Image Synthesis with a Single (Robust) Classifier. arXiv, arXiv:1906.09453

.. [SIE+20] Salman H., Ilyas A., Engstrom L., Kapoor A., Madry A. (2020). Do Adversarially Robust ImageNet Models Transfer Better? arXiv, arXiv:2007.08489

.. [STM20] Santurkar S., Tsipras D., Madry A. (2020). BREEDS: Benchmarks for Subpopulation Shift. arXiv, arXiv:2008.04859

.. [SIE+21] Salman H., Ilyas A., Engstrom L., Vemprala S., Kapoor A., Madry A. (2021). Unadversarial Examples: Designing Objects for Robust Vision. arXiv, arXiv:2012.12235

.. [SJW+21] Salman H., Jain S., Wong E., Madry A. (2021). : Certified Patch Robustness via Smoothed Vision Transformers. arXiv, arXiv:2110.07719
