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

def initialize_generative(activations, locations, num_factors, voxel_noise):
    num_times = activations.shape[0]
    num_voxels = activations.shape[1]

    brain_center = torch.mean(locations, 0).unsqueeze(0)
    brain_center_std_dev = torch.sqrt(10 * torch.var(locations, 0).unsqueeze(0))

    mean_weight = Variable(torch.zeros((num_times, num_factors)))
    weight_std_dev = Variable(SOURCE_WEIGHT_STD_DEV * torch.ones(
        (num_times, num_factors)
    ))

    mean_factor_center = Variable(
        brain_center.expand(num_factors, 3) *
        torch.ones((num_factors, 3))
    )
    factor_center_std_dev = Variable(
        brain_center_std_dev.expand(num_factors, 3) *
        torch.ones((num_factors, 3))
    )

    mean_factor_log_width = Variable(torch.ones(num_factors))
    factor_log_width_std_dev = Variable(
        SOURCE_LOG_WIDTH_STD_DEV * torch.ones(num_factors)
    )

    def generative_model(times=None):
        weight_mu = mean_weight
        weight_sigma = weight_std_dev
        if times is not None:
            weight_mu = weight_mu[times[0]:times[1], :]
            weight_sigma = weight_sigma[times[0]:times[1], :]
        weights = pyro.sample('weights', dist.normal, weight_mu, weight_sigma)

        factor_centers = pyro.sample('factor_centers', dist.normal,
                                     mean_factor_center, factor_center_std_dev)
        factor_log_widths = pyro.sample('factor_log_widths', dist.normal,
                                        mean_factor_log_width,
                                        factor_log_width_std_dev)
        factors = Variable(utils.radial_basis(locations, factor_centers.data,
                                              factor_log_widths.data))

        return pyro.sample(
            'activations',
            dist.normal,
            torch.matmul(weights, factors),
            Variable(voxel_noise * torch.ones((weights.shape[0], locations.shape[0])))
        )

    return generative_model
