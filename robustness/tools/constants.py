import torch as ch
import cox
from cox import store

# dog (117), cat (5), frog (3), turtle (5), bird (21), 
# monkey (14), fish (9), crab (4), insect (20) 
RESTRICTED_IMAGNET_RANGES = [(151, 268), (281, 285), 
        (30, 32), (33, 37), (80, 100), (365, 382),
          (389, 397), (118, 121), (300, 319)]

CKPT_NAME = 'checkpoint.pt'
BEST_APPEND = '.best'
CKPT_NAME_LATEST = CKPT_NAME + '.latest'
CKPT_NAME_BEST = CKPT_NAME + BEST_APPEND

ATTACK_KWARG_KEYS = [
        'criterion',
        'constraint',
        'eps',
        'step_size',
        'iterations',
        'random_start',
        'random_restarts']

LOGS_SCHEMA = {
    'epoch':int,
    'nat_prec1':float,
    'adv_prec1':float,
    'nat_loss':float,
    'adv_loss':float,
    'train_prec1':float,
    'train_loss':float,
    'time':float
}

LOGS_TABLE = 'logs'

