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
NUM_SAMPLES = 100
SOURCE_WEIGHT_STD_DEV = np.sqrt(2.0)
SOURCE_LOG_WIDTH_STD_DEV = np.sqrt(3.0)
VOXEL_NOISE = 0.1

def initialize_tfa_model(activations, locations, num_factors, voxel_noise):
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

    def tfa(times=None):
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

    return tfa

def initialize_tfa_guide(activations, locations, num_factors):
    num_times = activations.shape[0]

    # Initialize our center, width, and weight parameters via K-means
    kmeans = KMeans(init='k-means++', n_clusters=num_factors, n_init=10,
                    random_state=100)
    kmeans.fit(locations.numpy())

    mean_centers = torch.Tensor(kmeans.cluster_centers_)
    mean_log_widths = np.log(2) + 2 * np.log(np.nanmax(np.std(locations.numpy(), axis=0)))
    mean_log_widths *= torch.ones(num_factors)

    initial_factors = utils.radial_basis(locations, mean_centers,
                                         mean_log_widths)
    initial_factors_T = initial_factors.t()
    activations_T = activations.t()
    initial_weights = torch.Tensor(
        np.linalg.solve(initial_factors.matmul(initial_factors_T),
                        initial_factors.matmul(activations_T))
    )

    def tfa_guide(times=None):
        weight_mu = pyro.param('mean_weight', Variable(initial_weights.t(), requires_grad=True))
        weight_sigma = pyro.param(
            'weight_std_dev',
            Variable(torch.sqrt(torch.rand((num_times, num_factors))), requires_grad=True)
        )
        if times is not None:
            weight_mu = weight_mu[times[0]:times[1], :]
            weight_sigma = weight_sigma[times[0]:times[1], :]
        weight = pyro.sample('weights', dist.normal, weight_mu, weight_sigma)

        centers_mu = pyro.param(
            'mean_centers',
            Variable(mean_centers, requires_grad=True)
        )
        center_sigma = pyro.param(
            'factor_center_std_dev',
            Variable(torch.sqrt(torch.rand((num_factors, 3))),
                     requires_grad=True)
        )
        factor_center = pyro.sample('factor_centers', dist.normal,
                                    centers_mu, center_sigma)

        log_width_mu = pyro.param(
            'mean_factor_log_width',
            Variable(mean_log_widths, requires_grad=True)
        )
        log_width_sigma = pyro.param(
            'factor_log_width_std_dev',
            Variable(torch.sqrt(torch.rand((num_factors))), requires_grad=True)
        )
        factor_log_width = pyro.sample('factor_log_widths', dist.normal,
                                       log_width_mu,
                                       log_width_sigma)
        return (weight, factor_center, factor_log_width)

    return tfa_guide

class TopographicalFactorAnalysis:
    """Overall container for a run of TFA"""
    def __init__(self, data_file, num_factors=NUM_FACTORS):
        self.num_factors = num_factors

        name, ext = os.path.splitext(data_file)
        if ext == '.nii':
            dataset = utils.nii2cmu(data_file)
            self._template = data_file
        else:
            dataset = sio.loadmat(data_file)
        _, self._name = os.path.split(name)
        # pull out the voxel activations and locations
        data = dataset['data']
        R = dataset['R']
        self.activations = torch.Tensor(data).t()
        self.locations = torch.Tensor(R)

        # This could be a huge file.  Close it
        del dataset

        self.tfa_model = initialize_tfa_model(self.activations,
                                              self.locations,
                                              num_factors=num_factors,
                                              voxel_noise=VOXEL_NOISE)
        self.tfa_guide = initialize_tfa_guide(self.activations, self.locations,
                                              num_factors=num_factors)
