"""Models for plain topographic factor analysis on a given fMRI data file."""

import numpy as np
from sklearn.cluster import KMeans

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter

import pyro
import pyro.distributions as dist

from . import utils

SOURCE_WEIGHT_STD_DEV = np.sqrt(2.0)
SOURCE_LOG_WIDTH_STD_DEV = np.sqrt(3.0)
VOXEL_NOISE = 0.1

def tfa_prior(times=None, expand_weight_params=True, num_times=None, params={}):
    if times is None:
        times = (0, num_times)

    if expand_weight_params:
        for k in params['weights']:
            shape = params['weights'][k].shape
            params['weights'][k] =\
                params['weights'][k].expand(times[1] - times[0], *shape)

    weights = utils.param_sample('weights', dist.normal, params)
    factor_centers = utils.param_sample('factor_centers', dist.normal, params)
    factor_log_widths = utils.param_sample('factor_log_widths', dist.normal,
                                           params)

    return weights, factor_centers, factor_log_widths

def tfa_likelihood(weights, factor_centers, factor_log_widths, locations=None,
                   params={}):
    factors = utils.radial_basis(Variable(locations, requires_grad=True),
                                 factor_centers, factor_log_widths)

    return utils.param_sample('activations', dist.normal, params,
                              mu=weights @ factors)

def parameterize_tfa_model(activations, locations, num_factors, voxel_noise):
    num_times = activations.shape[0]
    num_voxels = activations.shape[1]

    brain_center = torch.mean(locations, 0).unsqueeze(0)
    brain_center_std_dev = torch.sqrt(torch.var(locations, 0)).unsqueeze(0)

    prior = utils.PyroPartial(tfa_prior)
    prior.num_times = num_times
    prior.register_params({
        'weights': {
            'mu': torch.zeros(num_factors),
            'sigma': SOURCE_WEIGHT_STD_DEV * torch.ones(num_factors),
        },
        'factor_centers': {
            'mu': brain_center.expand(num_factors, 3),
            'sigma': brain_center_std_dev.expand(num_factors, 3),
        },
        'factor_log_widths': {
            'mu': torch.ones(num_factors),
            'sigma': SOURCE_LOG_WIDTH_STD_DEV * torch.ones(num_factors),
        },
    }, trainable=False)

    likelihood = utils.PyroPartial(tfa_likelihood)
    likelihood.register_buffer('locations', locations)
    likelihood.register_params({
        'activations': {
            'sigma': voxel_noise * torch.ones(num_times, num_voxels)
        }
    }, trainable=False)

    result = utils.PyroPartial.compose(likelihood, prior, unpack=True,
                                       name='tfa_model')
    result.SOFTPLUS = utils.SOFTPLUS
    return result

def parameterize_tfa_guide(activations, locations, num_factors):
    num_times = activations.shape[0]

    # Initialize our center, width, and weight parameters via K-means
    kmeans = KMeans(init='k-means++', n_clusters=num_factors, n_init=10,
                    random_state=100)
    kmeans.fit(locations.numpy())

    mean_centers = torch.Tensor(kmeans.cluster_centers_)
    mean_log_widths = torch.log(torch.Tensor([2]))
    mean_log_widths += 2 * torch.log(torch.Tensor([locations.std(dim=0).max()]))
    mean_log_widths = mean_log_widths * torch.ones(num_factors)

    initial_factors = utils.radial_basis(locations, mean_centers,
                                         mean_log_widths)
    initial_weights = torch.Tensor(
        np.linalg.solve(initial_factors @ initial_factors.t(),
                        initial_factors @ activations.t())
    ).t()

    prior = utils.PyroPartial(tfa_prior)
    prior.num_times = num_times
    prior.expand_weight_params = False
    prior.register_params({
        'weights': {
            'mu': initial_weights,
            'sigma': torch.sqrt(torch.rand(num_times, num_factors)),
        },
        'factor_centers': {
            'mu': mean_centers,
            'sigma': torch.sqrt(torch.rand(num_factors, 3)),
        },
        'factor_log_widths': {
            'mu': mean_log_widths,
            'sigma': torch.sqrt(torch.rand(num_factors)),
        },
    }, trainable=True)

    pyro.module('tfa_guide', prior)
    return prior
