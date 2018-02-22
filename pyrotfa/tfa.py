"""Perform plain topographic factor analysis on a given fMRI data file."""

import logging
import math
import os
import pickle
import time

# import some dependencies
import torch
from torch.autograd import Variable

import hypertools as hyp
import nilearn.plotting as niplot
import numpy as np
import scipy.io as sio
from sklearn.cluster import KMeans
import torch
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import torch.utils.data

import pyro
import pyro.distributions as dist

from . import utils

NUM_FACTORS = 10
SOURCE_WEIGHT_STD_DEV = np.sqrt(2.0)
SOURCE_LOG_WIDTH_STD_DEV = np.sqrt(3.0)
VOXEL_NOISE = 0.1

