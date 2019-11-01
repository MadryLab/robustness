CHANGELOG
=========

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
