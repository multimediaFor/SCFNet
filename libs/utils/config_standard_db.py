
""" Configuration file."""

import yaml
import numpy as np
from enum import Enum
import os.path as osp
from easydict import EasyDict as edict
import os
# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from FAST-RCNN (Ross Girshick)
# ----------------------------------------------------------------------------

class phase(Enum):
    TRAIN = 'train'
    VAL = 'val'


__C = edict()

# Public access to configuration settings
cfg = __C

# Number of CPU cores used to parallelize evaluation.
__C.N_JOBS = 32

# Paths to dataset folders
__C.PATH = edict()


__C.PHASE = phase.VAL

# Multiobject evaluation (Set to False only when evaluating DAVIS 2016)
__C.MULTIOBJECT = True

# Root folder of project
__C.PATH.ROOT = osp.abspath('')

# Data folder
__C.PATH.DATA = osp.abspath('')

# sub_Path to input images
__C.SEQUENCES = "videos"

# Path to annotations
__C.ANNOTATIONS = "masks"


# Paths to files
__C.FILES = edict()

# Measures and Statistics
__C.EVAL = edict()

def db_read_info():
    """ Read dataset properties from file."""
    with open(cfg.FILES.DB_INFO, 'r') as f:
        return edict(yaml.load(f))


def db_read_attributes():
    """ Read list of sequences. """
    return db_read_info().attributes


def db_read_years():
    """ Read list of sequences. """
    return db_read_info().years


def db_read_sequences(year=None,db_phase=None):
    """ Read list of sequences. """

    sequences = db_read_info().sequences

    if year is not None:
        sequences = filter(
            lambda s: int(s.year) <= int(year), sequences)

    if db_phase is not None:
        if db_phase == phase.TRAINVAL:
            sequences = filter(lambda s: ((s.set == phase.VAL) or
                                          (s.set == phase.TRAIN)), sequences)
        else:
            sequences = filter(lambda s: s.set == db_phase, sequences)
    # return list(sequences)
    return sequences

def db_video_list(db_phase=None):
    _path=os.path.join(cfg.PATH.DATA,db_phase,cfg.SEQUENCES)
    videos=os.listdir(_path)
    
    return videos
# # Load all sequences
# __C.SEQUENCES = dict([(sequence.name, sequence)
#                       for sequence in db_read_sequences()])

# __C.palette = np.loadtxt(__C.PATH.PALETTE, dtype=np.uint8).reshape(-1, 3)
