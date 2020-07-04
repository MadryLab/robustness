Training and evaluating networks via command line
==================================================
In this walkthrough, we'll go over how to train and evaluate networks via the
:mod:`robustness.main` command-line tool.

Training a standard (nonrobust) model
--------------------------------------
We'll start by training a standard (non-robust) model. This is accomplished through the following command:

.. code-block:: bash

   python -m robustness.main --dataset DATASET --data /path/to/dataset \
      --adv-train 0 --arch ARCH --out-dir /logs/checkpoints/dir/

In the above, :samp:`DATASET` can be any supported dataset (i.e. in
:attr:`robustness.datasets.DATASETS`). For a demonstration of how to add a
supported dataset, see :ref:`here <using-custom-datasets>`.

With the above command, you should start seeing progress bars indicating that
the training has begun! Note that there are a whole host of arguments that you
can customize in training, including optimizer parameters (e.g. :samp:`--lr`,
:samp:`--weight-decay`, :samp:`--momentum`), logging parameters (e.g.
:samp:`--log-iters`, :samp:`--save-ckpt-iters`), and learning rate schedule. To
see more about these arguments, we run:

.. code-block:: bash

   python -m robustness --help

For completeness, the full list of parameters related to *non-robust* training
are below:

.. code-block:: bash

     --out-dir OUT_DIR     where to save training logs and checkpoints (default:
                        required)
                           config path for loading in parameters (default: None)
     --exp-name EXP_NAME   where to save in (inside out_dir) (default: None)
     --dataset {imagenet,restricted_imagenet,cifar,cinic,a2b}
                           (choices: {arg_type}, default: required)
     --data DATA           path to the dataset (default: /tmp/)
     --arch ARCH           architecture (see {cifar,imagenet}_models/ (default:
                           required)
     --batch-size BATCH_SIZE
                           batch size for data loading (default: by dataset)
     --workers WORKERS     data loading workers (default: 30)
     --resume RESUME       path to checkpoint to resume from (default: None)
     --data-aug {0,1}      whether to use data augmentation (choices: {arg_type},
                           default: 1)
     --epochs EPOCHS       number of epochs to train for (default: by dataset)
     --lr LR               initial learning rate for training (default: 0.1)
     --weight_decay WEIGHT_DECAY
                           SGD weight decay parameter (default: by dataset)
     --momentum MOMENTUM   SGD momentum parameter (default: 0.9)
     --step-lr STEP_LR     number of steps between 10x LR drops (default: by
                           dataset)
     --step-lr-gamma GAMMA multiplier for each LR drop (default: 0.1, i.e., 10x drops)
     --custom-lr-multiplier CUSTOM_SCHEDULE
                           LR sched (format: [(epoch, LR),...]) (default: None)
     --lr-interpolation {linear, step} 
                           How to interpolate between learning rates (default: step)
     --log-iters LOG_ITERS
                           how frequently (in epochs) to log (default: 5)
     --save-ckpt-iters SAVE_CKPT_ITERS
                           how frequently (epochs) to save (-1 for bash, only
                           saves best and last) (default: -1)
     --mixed-precision {0, 1}
                           Whether to use mixed-precision training (needs
                           to be compiled with NVIDIA AMP support)

Finally, there is one additional argument, :samp:`--adv-eval {0,1}`, that enables
adversarial evaluation of the non-robust model as it is being trained (i.e.
instead of reporting just standard accuracy every few epochs, we'll also report
robust accuracy if :samp:`--adv-eval 1` is added). However, adding this argument
also necessitates the addition of hyperparameters for adversarial attack, which
we cover in the following section.

Training a robust model (adversarial training)
--------------------------------------------------
To train a robust model we proceed in the exact same way as for a standard
model, but with a few changes. First, we change :samp:`--adv-train 0` to
:samp:`--adv-train 1` in the training command. Then, we need to make sure to
supply all the necessary hyperparameters for the attack:

.. code-block:: bash

     --attack-steps ATTACK_STEPS
                        number of steps for adversarial attack (default: 7)
     --constraint {inf,2,unconstrained}
                           adv constraint (choices: {arg_type}, default:
                           required)
     --eps EPS             adversarial perturbation budget (default: required)
     --attack-lr ATTACK_LR
                           step size for PGD (default: required)
     --use-best {0,1}      if 1 (0) use best (final) PGD step as example
                           (choices: {arg_type}, default: 1)
     --random-restarts RANDOM_RESTARTS
                           number of random PGD restarts for eval (default: 0)
     --custom-eps-multiplier EPS_SCHEDULE
                           epsilon multiplier sched (same format as LR schedule)


Evaluating trained models
-------------------------
To evaluate a trained model, we use the ``--eval-only`` flag when calling
:mod:`robustness.main`. To evaluate the model for just standard
(not adversarial) accuracy, only the following arguments are required:

.. code-block:: bash

   python -m robustness.main --dataset DATASET --data /path/to/dataset \
      --eval-only 1 --out-dir OUT_DIR --arch arch --adv-eval 0 \
      --resume PATH_TO_TRAINED_MODEL_CHECKPOINT

We can also evaluate adversarial accuracy by changing ``--adv-eval 0`` to
``--adv-eval 1`` and also adding the arguments from the previous section used
for adversarial attacks.

Examples
--------
Training a non-robust ResNet-18 for the CIFAR dataset:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m robustness.main --dataset cifar --data /path/to/cifar \
      --adv-train 0 --arch resnet18 --out-dir /logs/checkpoints/dir/

Training a robust ResNet-50 for the Restricted-ImageNet dataset:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python -m robustness.main --dataset restricted_imagenet --data \
      $IMAGENET_PATH --adv-train 1 --arch resnet50 \
      --out-dir /tmp/logs/checkpoints/dir/ --eps 3.0 --attack-lr 0.5 \
      --attack-steps 7 --constraint 2

Testing the standard and adversarial accuracy of a trained CIFAR-10 model with
L2 norm constraint of 0.5 and 100 L2-PGD steps:

.. code-block:: bash

   python -m robustness.main --dataset cifar --eval-only 1 --out-dir /tmp/ \
   --arch resnet50 --adv-eval 1 --constraint 2 --eps 0.5 --attack-lr 0.1 \
   --attack-steps 100 --resume path/to/ckpt/checkpoint.pt.best

Reading and analyzing training results
--------------------------------------

By default, the above command will store all the data generated from the
training process above in a subdirectory inside of
:samp:`/logs/checkpoints/dir/`, the path supplied to the :samp:`--out-dir`
argument. The subdirectory will be named by default via a 36 character, randomly
generated unique identifier, but it can be named manually via the
:samp:`--exp-name` argument. By the end of training, the folder structure will
look something like like:

.. code-block:: bash

   /logs/checkpoints/dir/a9ffc412-595d-4f8c-8e35-41f000cd35ed
      checkpoint.latest.pt
      checkpoint.best.pt
      store.h5
      tensorboard/
      save/ 

This is the file structure of a data store from the
`Cox <https://github.com/madrylab/cox>`_ logging library. It contains all the
tables (stored as Pandas dataframes, in HDF5 format) of data we wrote about the
experiment:

.. code-block:: python

   >>> from cox import store
   >>> s = store.Store('/logs/checkpoints/dir/', '6aeae7de-3549-49d5-adb6-52fe04689b4e')
   >>> s.tables
   {'ckpts': <cox.store.Table object at 0x7f09a6ae99b0>, 'logs': <cox.store.Table object at 0x7f09a6ae9e80>, 'metadata': <cox.store.Table object at 0x7f09a6ae9dd8>}

We can get the metadata by looking at the metadata table and extracting values
we want. For example, if we wanted to get the learning rate, 0.1:

.. code-block:: python

   >>> s['metadata'].df['lr']
   0    0.1
   Name: lr, dtype: float64

Or, if we wanted to find out which epoch had the highest validation accuracy:

.. code-block:: python

   >>> l_df = s['logs']
   >>> ldf[ldf['nat_prec1'] == max(ldf['nat_prec1'].tolist())]['epoch'].tolist()[0]
   32

In a similar manner, the 'ckpts' table contains all the previous checkpoints,
and the 'logs' table contains logging information pertaining to the training.
Cox allows us to really easily aggregate training logs across different training
runs and compare/analyze them---we recommend taking a look at the `Cox documentation
<https://cox.readthedocs.io>`_ for more information on how to use it.

Note that when training models programmatically (as in our walkthrough
:doc:`Part 1 <../example_usage/training_lib_part_1>` and :doc:`Part 2 <../example_usage/training_lib_part_2>`), it is possible to add on custom
logging functionalities and keep track of essentially anything during training.
