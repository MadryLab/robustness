"""
This module is used to set up arguments and defaults. For information on how to
use it, see Step 2 of the :doc:`../example_usage/training_lib_part_1`
walkthrough.
"""

from . import attacker, datasets
from .tools import helpers

BY_DATASET = 'varies by dataset'
REQ = 'REQUIRED'

TRAINING_DEFAULTS = {
    datasets.CIFAR: {
        "epochs": 150,
        "batch_size": 128,
        "weight_decay":5e-4,
        "step_lr": 50
    },
    datasets.CINIC: {
        "epochs": 150,
        "batch_size": 128,
        "weight_decay":5e-4,
        "step_lr": 50
    },
    datasets.ImageNet: {
        "epochs": 200,
        "batch_size":256,
        "weight_decay":1e-4,
        "step_lr": 50
    },
    datasets.Places365: {
        "epochs": 200,
        "batch_size":256,
        "weight_decay":1e-4,
        "step_lr": 50
    },
    datasets.RestrictedImageNet: {
        "epochs": 150,
        "batch_size": 256,
        "weight_decay": 1e-4,
        "step_lr": 50
    },
    datasets.CustomImageNet: {
         "epochs": 200,
         "batch_size": 256,
         "weight_decay": 1e-4,
         "step_lr": 50
    },
    datasets.A2B: {
        "epochs": 150,
        "batch_size": 64,
        "weight_decay": 5e-4,
        "step_lr": 50
    },
    datasets.OpenImages: {
        "epochs": 200,
        "batch_size":256,
        "weight_decay":1e-4,
        "step_lr": 50
    },
}
"""
Default hyperparameters for training by dataset (tested for resnet50).
Parameters can be accessed as `TRAINING_DEFAULTS[dataset_class][param_name]`
"""

TRAINING_ARGS = [
    ['out-dir', str, 'where to save training logs and checkpoints', REQ],
    ['epochs', int, 'number of epochs to train for', BY_DATASET],
    ['lr', float, 'initial learning rate for training', 0.1],
    ['weight-decay', float, 'SGD weight decay parameter', BY_DATASET],
    ['momentum', float, 'SGD momentum parameter', 0.9],
    ['step-lr', int, 'number of steps between step-lr-gamma x LR drops', BY_DATASET],
    ['step-lr-gamma', float, 'multiplier by which LR drops in step scheduler', 0.1],
    ['custom-lr-multiplier', str, 'LR multiplier sched (format: [(epoch, LR),...])', None],
    ['lr-interpolation', ["linear", "step"], 'Drop LR as step function or linearly', "step"],
    ['adv-train', [0, 1], 'whether to train adversarially', REQ],
    ['adv-eval', [0, 1], 'whether to adversarially evaluate', None], 
    ['log-iters', int, 'how frequently (in epochs) to log', 5],
    ['save-ckpt-iters', int, 'how frequently (epochs) to save \
            (-1 for none, only saves best and last)', -1]
]
"""
Arguments essential for the `train_model` function.

*Format*: `[NAME, TYPE/CHOICES, HELP STRING, DEFAULT (REQ=required,
BY_DATASET=looked up in TRAINING_DEFAULTS at runtime)]`
"""

PGD_ARGS = [
    ['attack-steps', int, 'number of steps for PGD attack', 7],
    ['constraint', list(attacker.STEPS.keys()), 'adv constraint', REQ],
    ['eps', str , 'adversarial perturbation budget', REQ],
    ['attack-lr', str, 'step size for PGD', REQ],
    ['use-best', [0, 1], 'if 1 (0) use best (final) PGD step as example', 1],
    ['random-restarts', int, 'number of random PGD restarts for eval', 0],
    ['random-start', [0, 1], 'start with random noise instead of pgd step', 0],
    ['custom-eps-multiplier', str, 'eps mult. sched (same format as LR)', None]
]
"""
Arguments essential for the :meth:`robustness.train.train_model` function if
adversarially training or evaluating.

*Format*: `[NAME, TYPE/CHOICES, HELP STRING, DEFAULT (REQ=required,
BY_DATASET=looked up in TRAINING_DEFAULTS at runtime)]`
"""

MODEL_LOADER_ARGS = [
    ['dataset', list(datasets.DATASETS.keys()), '', REQ],
    ['data', str, 'path to the dataset', '/tmp/'],
    ['arch', str, 'architecture (see {cifar,imagenet}_models/', REQ],
    ['batch-size', int, 'batch size for data loading', BY_DATASET],
    ['workers', int, '# data loading workers', 30],
    ['resume', str, 'path to checkpoint to resume from', None],
    ['resume-optimizer', [0, 1], 'whether to also resume optimizers', 0],
    ['data-aug', [0, 1], 'whether to use data augmentation', 1],
    ['mixed-precision', [0, 1], 'whether to use MP training (faster)', 0],
]
"""
Arguments essential for constructing the model and dataloaders that will be fed
into :meth:`robustness.train.train_model` or :meth:`robustness.train.eval_model`

*Format*: `[NAME, TYPE/CHOICES, HELP STRING, DEFAULT (REQ=required,
BY_DATASET=looked up in TRAINING_DEFAULTS at runtime)]`
"""

CONFIG_ARGS = [
    ['config-path', str, 'config path for loading in parameters', None],
    ['eval-only', [0, 1], 'just run evaluation (no training)', 0],
    ['exp-name', str, 'where to save in (inside out_dir)', None]
]
"""
Arguments for main.py specifically

*Format*: `[NAME, TYPE/CHOICES, HELP STRING, DEFAULT (REQ=required,
BY_DATASET=looked up in TRAINING_DEFAULTS at runtime)]`
"""

def add_args_to_parser(arg_list, parser):
    """
    Adds arguments from one of the argument lists above to a passed-in
    arparse.ArgumentParser object. Formats helpstrings according to the
    defaults, but does NOT set the actual argparse defaults (*important*).

    Args:
        arg_list (list) : A list of the same format as the lists above, i.e.
            containing entries of the form [NAME, TYPE/CHOICES, HELP, DEFAULT]
        parser (argparse.ArgumentParser) : An ArgumentParser object to which the
            arguments will be added

    Returns:
        The original parser, now with the arguments added in.
    """
    for arg_name, arg_type, arg_help, arg_default in arg_list:
        has_choices = (type(arg_type) == list) 
        kwargs = {
            'type': type(arg_type[0]) if has_choices else arg_type,
            'help': f"{arg_help} (default: {arg_default})"
        }
        if has_choices: kwargs['choices'] = arg_type
        parser.add_argument(f'--{arg_name}', **kwargs)
    return parser

def check_and_fill_args(args, arg_list, ds_class):
    """
    Fills in defaults based on an arguments list (e.g., TRAINING_ARGS) and a
    dataset class (e.g., datasets.CIFAR).

    Args:
        args (object) : Any object subclass exposing :samp:`setattr` and
            :samp:`getattr` (e.g. cox.utils.Parameters)
        arg_list (list) : A list of the same format as the lists above, i.e.
            containing entries of the form [NAME, TYPE/CHOICES, HELP, DEFAULT]
        ds_class (type) : A dataset class name (i.e. a
            :class:`robustness.datasets.DataSet` subclass name)

    Returns:
        args (object): The :samp:`args` object with all the defaults filled in according to :samp:`arg_list` defaults.
    """
    for arg_name, _, _, arg_default in arg_list:
        name = arg_name.replace("-", "_")
        if helpers.has_attr(args, name): continue
        if arg_default == REQ: raise ValueError(f"{arg_name} required")
        elif arg_default == BY_DATASET:
            setattr(args, name, TRAINING_DEFAULTS[ds_class][name])
        elif arg_default is not None: 
            setattr(args, name, arg_default)
    return args



