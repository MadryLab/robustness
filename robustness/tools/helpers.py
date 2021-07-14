import torch as ch

import shutil
import dill
import os
from subprocess import Popen, PIPE
import pandas as pd
from PIL import Image
from . import constants

def has_attr(obj, k):
    """Checks both that obj.k exists and is not equal to None"""
    try:
        return (getattr(obj, k) is not None)
    except KeyError as e:
        return False
    except AttributeError as e:
        return False

def calc_est_grad(func, x, y, rad, num_samples):
    B, *_ = x.shape
    Q = num_samples//2
    N = len(x.shape) - 1
    with ch.no_grad():
        # Q * B * C * H * W
        extender = [1]*N
        queries = x.repeat(Q, *extender)
        noise = ch.randn_like(queries)
        norm = noise.view(B*Q, -1).norm(dim=-1).view(B*Q, *extender)
        noise = noise / norm
        noise = ch.cat([-noise, noise])
        queries = ch.cat([queries, queries])
        y_shape = [1] * (len(y.shape) - 1)
        l = func(queries + rad * noise, y.repeat(2*Q, *y_shape)).view(-1, *extender) 
        grad = (l.view(2*Q, B, *extender) * noise.view(2*Q, B, *noise.shape[1:])).mean(dim=0)
    return grad

def ckpt_at_epoch(num):
    return '%s_%s' % (num, constants.CKPT_NAME)

def accuracy(output, target, topk=(1,), exact=False):
    """
        Computes the top-k accuracy for the specified values of k

        Args:
            output (ch.tensor) : model output (N, classes) or (N, attributes) 
                for sigmoid/multitask binary classification
            target (ch.tensor) : correct labels (N,) [multiclass] or (N,
                attributes) [multitask binary]
            topk (tuple) : for each item "k" in this tuple, this method
                will return the top-k accuracy
            exact (bool) : whether to return aggregate statistics (if
                False) or per-example correctness (if True)

        Returns:
            A list of top-k accuracies.
    """
    with ch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [ch.round(ch.sigmoid(output)).eq(ch.round(target)).float().mean()], [-1.0] 

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact

class InputNormalize(ch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        # x = ch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized

# class DataPrefetcher():
#     def __init__(self, loader, stop_after=None):
#         self.loader = loader
#         self.dataset = loader.dataset

#     def __len__(self):
#         return len(self.loader)

#     def __iter__(self):
#         # count = 0
#         # self.loaditer = iter(self.loader)
#         # self.preload()
#         # for i in range(loaditer)
#         # while self.next_input is not None:
#         #     ch.cuda.current_stream().wait_stream(self.stream)
#         #     input = self.next_input
#         #     target = self.next_target
#         #     self.preload()
#         #     count += 1
#         #     yield input, target
#         #     if type(self.stop_after) is int and (count > self.stop_after):
#         #         break
#         prev_x, prev_y = None, None
#         for _, (x, y) in enumerate(self.loader):
#             x = x.to(device='cuda', memory_format=ch.channels_last,
#                      non_blocking=True).to(dtype=ch.float32, non_blocking=True)
#             y = y.to(device='cuda', non_blocking=True)
#             if prev_x is None:
#                 prev_x, prev_y = x, y
#             else:
#                 yield prev_x, prev_y
#                 prev_x, prev_y = x, y

#         yield prev_x, prev_y

class DataPrefetcher():
    def __init__(self, loader, stop_after=None):
        self.loader = loader
        self.dataset = self.loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        prefetcher = data_prefetcher(self.loader)
        x, y = prefetcher.next()
        while x is not None:
            yield x, y
            x, y = prefetcher.next()

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = ch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = ch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = ch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(ch.cuda.current_stream())
        with ch.cuda.stream(self.stream):
            # self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.to(device='cuda', memory_format=ch.channels_last,
                                                 non_blocking=True).to(dtype=ch.float32, non_blocking=True)
            # to(dtype=ch.float32, non_blocking=True)

    def next(self):
        ch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(ch.cuda.current_stream())
        if target is not None:
            target.record_stream(ch.cuda.current_stream())
        self.preload()
        return input, target


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(self.count, 1)

    def update(self, val, _=0):
        self.val = val
        self.sum += val
        self.count += 1
        # self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg2' + self.fmt + '} ({sum' + self.fmt + '})'
        self.avg2 = self.avg
        # self.avg = (self.sum / max(self.count, 1))
        return fmtstr.format(**self.__dict__)

# ImageNet label mappings
def get_label_mapping(dataset_name, ranges):
    if dataset_name == 'imagenet':
        label_mapping = None
    elif dataset_name == 'restricted_imagenet':
        def label_mapping(classes, class_to_idx):
            return restricted_label_mapping(classes, class_to_idx, ranges=ranges)
    elif dataset_name == 'custom_imagenet':
        def label_mapping(classes, class_to_idx):
            return custom_label_mapping(classes, class_to_idx, ranges=ranges)
    else:
        raise ValueError('No such dataset_name %s' % dataset_name)

    return label_mapping

def restricted_label_mapping(classes, class_to_idx, ranges):
    range_sets = [
        set(range(s, e+1)) for s,e in ranges
    ]

    # add wildcard
    # range_sets.append(set(range(0, 1002)))
    mapping = {}
    for class_name, idx in class_to_idx.items():
        for new_idx, range_set in enumerate(range_sets):
            if idx in range_set:
                mapping[class_name] = new_idx
        # assert class_name in mapping
    filtered_classes = list(mapping.keys()).sort()
    return filtered_classes, mapping

def custom_label_mapping(classes, class_to_idx, ranges):

    mapping = {}
    for class_name, idx in class_to_idx.items():
        for new_idx, range_set in enumerate(ranges):
            if idx in range_set:
                mapping[class_name] = new_idx

    filtered_classes = list(mapping.keys()).sort()
    return filtered_classes, mapping
