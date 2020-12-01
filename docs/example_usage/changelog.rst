CHANGELOG
=========

robustness 1.2.post2 
'''''''''''''''''''''
- Add SqueezeNet architectures
- (Preliminary) Torch 1.7 support
- Support for specifying device_ids in DataParallel

robustness 1.2
'''''''''''''''
- Biggest new features:
    - New ImageNet models
    - Mixed-precision training
    - OpenImages and Places365 datasets added
    - Ability to specify a custom accuracy function (custom loss functions
        were already supported, this is just for logging)
    - Improved resuming functionality
- Changes to CLI-based training:
    - ``--custom-lr-schedule`` replaced by ``--custom-lr-multiplier`` (same format)
    - ``--eps-fadein-epochs`` replaced by general ``--custom-eps-multiplier`` 
        (now same format as custom-lr schedule)
    - ``--step-lr-gamma`` now available to change the size of learning rate
        drops (used to be fixed to 10x drops)
    - ``--lr-interpolation`` argument added (can choose between linear and step
        interpolation between learning rates in the schedule)
    - ``--weight_decay`` is now called ``--weight-decay``, keeping with
        convention
    - ``--resume-optimizer`` is a 0/1 argument for whether to resume the
        optimizer and LR schedule, or just the model itself
    - ``--mixed-precision`` is a 0/1 argument for whether to use mixed-precision
        training or not (required PyTorch compiled with AMP support)
- Model and data loading:
    - DataParallel is now *off* by default when loading models, even when
        resume_path is specified (previously it was off for new models, and on
        for resumed models by default)
    - New ``add_custom_forward`` for ``make_and_restore_model`` (see docs for
        more details)
    - Can now pass a random seed for training data subsetting 
- Training:
    - See new CLI features---most have training-as-a-library counterparts
    - Fixed a bug that did not resume the optimizer and schedule 
    - Support for custom accuracy functions
    - Can now disable ``torch.nograd`` for test set eval (in case you have a
        custom accuracy function that needs gradients even on the val set)
- PGD:
    - Better random start for l2 attacks
    - Added a ``RandomStep`` attacker step (useful for large-noise training with
        varying noise over training)
    - Fixed bug in the ``with_image`` argument (minor)
- Model saving:
    - Accuracies are now saved in the checkpoint files themselves (instead of
        just in the log stores)
    - Removed redundant checkpoints table from the log store, as it is a
        duplicate of the latest checkpoint file and just wastes space
- Cleanup:
    - Remove redundant ``save_checkpoint`` function in helpers file 
    - Code flow improvements


robustness 1.1.post2
'''''''''''''''''''''
- Critical fix in :meth:`robustness.loaders.TransformedLoader`, allow for data shuffling

robustness 1.1
''''''''''''''
- Added ability to superclass ImageNet to make 
  custom datasets (:doc:`docs <custom_imagenet>`)
- Added ``shuffle_train`` and ``shuffle_test`` options to
  :meth:`~robustness.datasets.Dataset.make_loaders`
- Added support for cyclic learning rate (``--custom-schedule cyclic`` via command line or ``{"custom_schedule": "cyclic"}`` from Python
- Added support for transfer learning/partial parameter updates,
  :meth:`robustness.train.train_model` now takes ``update_params`` argument,
  list of parameters to update
- Allow ``random_start`` (random start for adversarial attacks) to be set via
  command line
- Change defaults for ImageNet training (``200`` epochs instead of ``350``)
- Small fixes/refinements to :mod:`robustness.tools.vis_tools` module
