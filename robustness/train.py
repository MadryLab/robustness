import torch as ch
import numpy as np
import torch.nn as nn
from torch.optim import SGD, Adam, lr_scheduler
from torchvision.utils import make_grid
from torch.nn.utils import parameters_to_vector as flatten
from cox.utils import Parameters

from .tools import helpers
from .tools.helpers import AverageMeter, ckpt_at_epoch, has_attr
from .tools import constants as consts
import dill
import os
import time
import warnings
from pytorch_loss import LabelSmoothSoftmaxCEV3

from torch.cuda.amp import autocast
from apex import amp

if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

def check_required_args(args, eval_only=False):
    """
    Check that the required training arguments are present.

    Args:
        args (argparse object): the arguments to check
        eval_only (bool) : whether to check only the arguments for evaluation
    """
    required_args_eval = ["adv_eval"]
    required_args_train = ["epochs", "out_dir", "adv_train",
        "log_iters", "lr", "momentum", "weight_decay"]
    adv_required_args = ["attack_steps", "eps", "constraint",
            "use_best", "attack_lr", "random_restarts"]

    # Generic function for checking all arguments in a list
    def check_args(args_list):
        for arg in args_list:
            assert has_attr(args, arg), f"Missing argument {arg}"

    # Different required args based on training or eval:
    if not eval_only: check_args(required_args_train)
    else: check_args(required_args_eval)
    # More required args if we are robustly training or evaling
    is_adv = bool(args.adv_train) or bool(args.adv_eval)
    if is_adv:
        check_args(adv_required_args)
    # More required args if the user provides a custom training loss
    has_custom_train = has_attr(args, 'custom_train_loss')
    has_custom_adv = has_attr(args, 'custom_adv_loss')
    if has_custom_train and is_adv and not has_custom_adv:
        raise ValueError("Cannot use custom train loss \
            without a custom adversarial loss (see docs)")


def make_optimizer_and_schedule(args, model, checkpoint, params, iters_per_epoch):
    """
    *Internal Function* (called directly from train_model)

    Creates an optimizer and a schedule for a given model, restoring from a
    checkpoint if it is non-null.

    Args:
        args (object) : an arguments object, see
            :meth:`~robustness.train.train_model` for details
        model (AttackerModel) : the model to create the optimizer for
        checkpoint (dict) : a loaded checkpoint saved by this library and loaded
            with `ch.load`
        params (list|None) : a list of parameters that should be updatable, all
            other params will not update. If ``None``, update all params

    Returns:
        An optimizer (ch.nn.optim.Optimizer) and a scheduler
            (ch.nn.optim.lr_schedulers module).
    """
    # Make optimizer
    param_list = model.parameters() if params is None else params

    if args.optimizer == 'Adam':
        optimizer = Adam(param_list, lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                         weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD_no_bn_wd':
        all_params = {k: v for (k, v) in model.named_parameters()}
        param_groups = [{
                            'params': [all_params[k] for k in all_params if ('bn' in k)],
                            'weight_decay': 0.
                        }, {
                            'params': [all_params[k] for k in all_params if not ('bn' in k)],
                            'weight_decay': args.weight_decay
                        }]
        optimizer = SGD(param_groups, args.lr, momentum=args.momentum)
    else:
        optimizer = SGD(param_list, args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # Make schedule
    schedule = None
    if args.custom_lr_multiplier == 'reduce_on_plateau':
        schedule = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                  patience=5, mode='min')
    elif args.custom_lr_multiplier.startswith('cyclic'):
        # E.g. `cyclic_5` for peaking at 5 epochs
        # peak = int(args.custom_lr_multiplier.split('_')[-1])
        peak = float(args.custom_lr_multiplier.split('_')[-1])
        cyc_lr_func = lambda t: np.interp([t+1], [0, peak, args.epochs], [0, 1, 0])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, cyc_lr_func)
    elif args.custom_lr_multiplier.startswith('itercyclic'):
        # peak = int(args.custom_lr_multiplier.split('_')[-1])
        peak = float(args.custom_lr_multiplier.split('_')[-1])
        # itercyc_lr_func = lambda t: np.interp([float(t+1) / iters_per_epoch], [0, peak, args.epochs], [0, 1, 0])[0]
        itercyc_lr_func = lambda t: np.interp([float(t+1) / iters_per_epoch], [0, peak, args.epochs], [0, 1, 0])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, itercyc_lr_func)
    elif args.custom_lr_multiplier:
        cs = args.custom_lr_multiplier
        periods = eval(cs) if type(cs) is str else cs
        if args.lr_interpolation == 'linear':
            lin_lr_func = lambda t: np.interp([t], *zip(*periods))[0]
        else:
            def lr_func(ep):
                for (milestone, lr) in reversed(periods):
                    if ep >= milestone: return lr
                return 1.0
            lin_lr_func = lr_func
        schedule = lr_scheduler.LambdaLR(optimizer, lin_lr_func)
    elif args.step_lr:
        schedule = lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=args.step_lr_gamma)

    # Fast-forward the optimizer and the scheduler if resuming
    if checkpoint:
        raise ValueError('Not supported in this fork')

    return optimizer, schedule

"""
def eval_model(args, model, loader, store):
    Evaluate a model for standard (and optionally adversarial) accuracy.

    Args:
        args (object) : A list of arguments---should be a python object
            implementing ``getattr()`` and ``setattr()``.
        model (AttackerModel) : model to evaluate
        loader (iterable) : a dataloader serving `(input, label)` batches from
            the validation set
        store (cox.Store) : store for saving results in (via tensorboardX)
    check_required_args(args, eval_only=True)
    start_time = time.time()

    if store is not None:
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)
    writer = store.tensorboard if store else None

    assert not hasattr(model, "module"), "model is already in DataParallel."
    model = ch.nn.DataParallel(model)
    prec1, nat_loss = _model_loop(args, 'val', loader, model, None, 0, False,
                                  writer)

    adv_prec1, adv_loss = float('nan'), float('nan')
    if args.adv_eval:
        args.eps = eval(str(args.eps)) if has_attr(args, 'eps') else None
        args.attack_lr = eval(str(args.attack_lr)) if has_attr(args, 'attack_lr') else None
        adv_prec1, adv_loss = _model_loop(args, 'val', loader, model, None, 0,
                                          True, writer)
    log_info = {
        'epoch':0,
        'nat_prec1':prec1,
        'adv_prec1':adv_prec1,
        'nat_loss':nat_loss,
        'adv_loss':adv_loss,
        'train_prec1':float('nan'),
        'train_loss':float('nan'),
        'time': time.time() - start_time
    }

    # Log info into the logs table
    if store: store[consts.LOGS_TABLE].append_row(log_info)
    return log_info
"""

def train_model(args, model, data_aug, loaders, *, checkpoint=None, dp_device_ids=None,
            store=None, update_params=None, disable_no_grad=False,
            log_checkpoints=False):
    """
    Main function for training a model.

    Args:
        args (object) : A python object for arguments, implementing
            ``getattr()`` and ``setattr()`` and having the following
            attributes. See :attr:`robustness.defaults.TRAINING_ARGS` for a
            list of arguments, and you can use
            :meth:`robustness.defaults.check_and_fill_args` to make sure that
            all required arguments are filled and to fill missing args with
            reasonable defaults:

            adv_train (int or bool, *required*)
                if 1/True, adversarially train, otherwise if 0/False do
                standard training
            epochs (int, *required*)
                number of epochs to train for
            lr (float, *required*)
                learning rate for SGD optimizer
            weight_decay (float, *required*)
                weight decay for SGD optimizer
            momentum (float, *required*)
                momentum parameter for SGD optimizer
            step_lr (int)
                if given, drop learning rate by 10x every `step_lr` steps
            custom_lr_multiplier (str)
                If given, use a custom LR schedule, formed by multiplying the
                    original ``lr`` (format: [(epoch, LR_MULTIPLIER),...])
            lr_interpolation (str)
                How to drop the learning rate, either ``step`` or ``linear``,
                    ignored unless ``custom_lr_multiplier`` is provided.
            adv_eval (int or bool)
                If True/1, then also do adversarial evaluation, otherwise skip
                (ignored if adv_train is True)
            log_iters (int, *required*)
                How frequently (in epochs) to save training logs
            save_ckpt_iters (int, *required*)
                How frequently (in epochs) to save checkpoints (if -1, then only
                save latest and best ckpts)
            attack_lr (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack step size
            constraint (str, *required if adv_train or adv_eval*)
                the type of adversary constraint
                (:attr:`robustness.attacker.STEPS`)
            eps (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack budget
            attack_steps (int, *required if adv_train or adv_eval*)
                number of steps to take in adv attack
            custom_eps_multiplier (str, *required if adv_train or adv_eval*)
                If given, then set epsilon according to a schedule by
                multiplying the given eps value by a factor at each epoch. Given
                in the same format as ``custom_lr_multiplier``, ``[(epoch,
                MULTIPLIER)..]``
            use_best (int or bool, *required if adv_train or adv_eval*) :
                If True/1, use the best (in terms of loss) PGD step as the
                attack, if False/0 use the last step
            random_restarts (int, *required if adv_train or adv_eval*)
                Number of random restarts to use for adversarial evaluation
            custom_train_loss (function, optional)
                If given, a custom loss instead of the default CrossEntropyLoss.
                Takes in `(logits, targets)` and returns a scalar.
            custom_adv_loss (function, *required if custom_train_loss*)
                If given, a custom loss function for the adversary. The custom
                loss function takes in `model, input, target` and should return
                a vector representing the loss for each element of the batch, as
                well as the classifier output.
            custom_accuracy (function)
                If given, should be a function that takes in model outputs
                and model targets and outputs a top1 and top5 accuracy, will
                displayed instead of conventional accuracies
            regularizer (function, optional)
                If given, this function of `model, input, target` returns a
                (scalar) that is added on to the training loss without being
                subject to adversarial attack
            iteration_hook (function, optional)
                If given, this function is called every training iteration by
                the training loop (useful for custom logging). The function is
                given arguments `model, iteration #, loop_type [train/eval],
                current_batch_ims, current_batch_labels`.
            epoch hook (function, optional)
                Similar to iteration_hook but called every epoch instead, and
                given arguments `model, log_info` where `log_info` is a
                dictionary with keys `epoch, nat_prec1, adv_prec1, nat_loss,
                adv_loss, train_prec1, train_loss`.

        model (AttackerModel) : the model to train.
        loaders (tuple[iterable]) : `tuple` of data loaders of the form
            `(train_loader, val_loader)`
        checkpoint (dict) : a loaded checkpoint previously saved by this library
            (if resuming from checkpoint)
        dp_device_ids (list|None) : if not ``None``, a list of device ids to
            use for DataParallel.
        store (cox.Store) : a cox store for logging training progress
        update_params (list) : list of parameters to use for training, if None
            then all parameters in the model are used (useful for transfer
            learning)
        disable_no_grad (bool) : if True, then even model evaluation will be
            run with autograd enabled (otherwise it will be wrapped in a ch.no_grad())
    """
    # scaler = GradScaler()
    # Logging setup
    writer = store.tensorboard if store else None
    prec1_key = f"{'adv' if args.adv_train else 'nat'}_prec1"
    if store is not None:
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)

    # Reformat and read arguments
    check_required_args(args) # Argument sanity check
    for p in ['eps', 'attack_lr', 'custom_eps_multiplier']:
        setattr(args, p, eval(str(getattr(args, p))) if has_attr(args, p) else None)
    if args.custom_eps_multiplier is not None:
        eps_periods = args.custom_eps_multiplier
        args.custom_eps_multiplier = lambda t: np.interp([t], *zip(*eps_periods))[0]

    # Initial setup
    train_loader, val_loader = loaders
    if not args.opt_model_and_schedule:
        opt, schedule = make_optimizer_and_schedule(args, model, checkpoint,
                                                    update_params, len(train_loader))
        assert not hasattr(model, "module"), "model is already in DataParallel."
        model = ch.nn.DataParallel(model, device_ids=dp_device_ids).cuda()
    else:
        opt, _, schedule = args.opt_model_and_schedule

    if args.custom_lr_multiplier.startswith('itercyclic'):
        assert args.iteration_hook is None
        def iter_hook(*_): schedule.step()
        args.iteration_hook = iter_hook

    best_prec1, start_epoch = (0, 0)
    if checkpoint:
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint[prec1_key] if prec1_key in checkpoint \
            else _model_loop(args, 'val', val_loader, model, None,
                            start_epoch-1, args.adv_train, writer)[0]

    # Timestamp for training start time
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_loader.dataset, 'next_epoch'):
            train_loader.dataset.next_epoch()
        # train for one epoch
        train_prec1, train_loss = _model_loop(args, 'train', train_loader,
                model, opt, epoch, args.adv_train, writer, data_aug=data_aug)
        last_epoch = (epoch == (args.epochs - 1))

        # evaluate on validation set
        sd_info = {
            'model':model.state_dict(),
            'optimizer':opt.state_dict(),
            'schedule':(schedule and schedule.state_dict()),
            'epoch': epoch+1
        }

        def save_checkpoint(filename):
            if log_checkpoints:
                ckpt_save_path = os.path.join(args.out_dir if not store else \
                                            store.path, filename)
                ch.save(sd_info, ckpt_save_path, pickle_module=dill)

        save_its = args.save_ckpt_iters
        should_save_ckpt = (epoch % save_its == 0) and (save_its > 0)
        should_log = (epoch % args.log_iters == 0)

        if epoch > 0 and (should_log or last_epoch or should_save_ckpt):
            # log + get best
            ctx = ch.enable_grad() if disable_no_grad else ch.no_grad()
            with ctx:
                prec1, nat_loss = _model_loop(args, 'val', val_loader, model,
                        None, epoch, False, writer)

            # loader, model, epoch, input_adv_exs
            should_adv_eval = args.adv_eval or args.adv_train
            adv_val = should_adv_eval and _model_loop(args, 'val', val_loader,
                    model, None, epoch, True, writer)
            adv_prec1, adv_loss = adv_val or (-1.0, -1.0)

            # remember best prec@1 and save checkpoint
            our_prec1 = adv_prec1 if args.adv_train else prec1
            is_best = our_prec1 > best_prec1
            best_prec1 = max(our_prec1, best_prec1)
            sd_info[prec1_key] = our_prec1

            # log every checkpoint
            log_info = {
                'epoch':epoch + 1,
                'nat_prec1':prec1,
                'adv_prec1':adv_prec1,
                'nat_loss':nat_loss,
                'adv_loss':adv_loss,
                'train_prec1':train_prec1,
                'train_loss':train_loss,
                'time': time.time() - start_time
            }

            # Log info into the logs table
            if store: store[consts.LOGS_TABLE].append_row(log_info)
            # If we are at a saving epoch (or the last epoch), save a checkpoint
            if should_save_ckpt or last_epoch: save_checkpoint(ckpt_at_epoch(epoch))

            # Update the latest and best checkpoints (overrides old one)
            # save_checkpoint(consts.CKPT_NAME_LATEST)
            if is_best: save_checkpoint(consts.CKPT_NAME_BEST)
        else:
            log_info = {
                'epoch':epoch + 1,
                'time':time.time() - start_time
            }

        if schedule:
            if 'reduce_on_plateau' in args.custom_lr_multiplier:
                schedule.step(nat_loss)
            elif not args.custom_lr_multiplier.startswith('itercyclic'):
                schedule.step()
        if has_attr(args, 'epoch_hook'): args.epoch_hook(model, log_info)

    return model

def _model_loop(args, loop_type, loader, model, opt, epoch, adv, writer,
                data_aug=None):
    """
    *Internal function* (refer to the train_model and eval_model functions for
    how to train and evaluate models).

    Runs a single epoch of either training or evaluating.

    Args:
        args (object) : an arguments object (see
            :meth:`~robustness.train.train_model` for list of arguments
        loop_type ('train' or 'val') : whether we are training or evaluating
        loader (iterable) : an iterable loader of the form
            `(image_batch, label_batch)`
        model (AttackerModel) : model to train/evaluate
        opt (ch.optim.Optimizer) : optimizer to use (ignored for evaluation)
        epoch (int) : which epoch we are currently on
        adv (bool) : whether to evaluate adversarially (otherwise standard)
        writer : tensorboardX writer (optional)

    Returns:
        The average top1 accuracy and the average loss across the epoch.
    """

    if not loop_type in ['train', 'val']:
        err_msg = "loop_type ({0}) must be 'train' or 'val'".format(loop_type)
        raise ValueError(err_msg)
    is_train = (loop_type == 'train')

    # losses = AverageMeter('losses')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    prec = 'NatPrec' if not adv else 'AdvPrec'
    loop_msg = 'Train' if loop_type == 'train' else 'Val'

    # switch to train/eval mode depending
    model = model.train() if is_train else model.eval()

    # If adv training (or evaling), set eps and random_restarts appropriately
    if adv:
        eps = args.custom_eps_multiplier(epoch) * args.eps \
                if (is_train and args.custom_eps_multiplier) else args.eps
        random_restarts = 0 if is_train else args.random_restarts

    # Custom training criterion
    has_custom_train_loss = has_attr(args, 'custom_train_loss')
    default_train_crit = LabelSmoothSoftmaxCEV3(lb_smooth=args.label_smoothing) if \
        args.label_smoothing is not None else ch.nn.CrossEntropyLoss()
    train_criterion = args.custom_train_loss if has_custom_train_loss \
            else default_train_crit

    has_custom_adv_loss = has_attr(args, 'custom_adv_loss')
    adv_criterion = args.custom_adv_loss if has_custom_adv_loss else None

    attack_kwargs = {}
    iterator = tqdm(enumerate(loader), total=len(loader))
    
    should_crop = hasattr(loader.dataset, 'current_crop') 
    if should_crop:
        crop_x, crop_y = loader.dataset.current_crop.numpy()
        
    total_correct, total = 0., 0.
    for i, (inp, target) in iterator:
        if loop_type == 'train':
            with autocast():
                inp = data_aug(inp)

        if should_crop:
            inp = inp[:, :, :crop_x, :crop_y]
        output = model(inp)
        loss = train_criterion(output, target)
        # with ch.no_grad():
            # losses.update(loss)

        if is_train:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        else:
            corrects = output.argmax(1).eq(target)
            total_correct += corrects.sum()
            total += corrects.shape[0]

        if has_attr(args, 'iteration_hook'):
            args.iteration_hook(None)

    if not is_train: 
        print(f'Val epoch {epoch}, accuracy {total_correct / total * 100:.2f}%', flush=True)
    # print(f'{loop_msg} avg loss', losses.avg, flush=True)
    # return 0., losses.avg
    return 0., 0.

