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

softplus = nn.Softplus()

def tfa_prior(times=None, expand_weight_params=True,
              weight_mean=None, weight_std_dev=None,
              factor_center_mean=None, factor_center_std_dev=None,
              factor_log_width_mean=None, factor_log_width_std_dev=None,
              num_times=None):
    if times is None:
        times = (0, num_times)

    if expand_weight_params:
        weight_mean = utils.unsqueeze_and_expand(weight_mean, 0,
                                                 times[1] - times[0], True)
        weight_std_dev = utils.unsqueeze_and_expand(weight_std_dev, 0,
                                                    times[1] - times[0], True)

    weights = pyro.sample('weights', dist.normal, weight_mean,
                          softplus(weight_std_dev))

    factor_centers = pyro.sample('factor_centers', dist.normal,
                                 factor_center_mean,
                                 softplus(factor_center_std_dev))
    factor_log_widths = pyro.sample('factor_log_widths', dist.normal,
                                    factor_log_width_mean,
                                    softplus(factor_log_width_std_dev))

    return weights, factor_centers, factor_log_widths

def tfa_likelihood(weights, factor_centers, factor_log_widths, locations=None,
                   activation_noise=None):
    factors = utils.radial_basis(locations, factor_centers.data,
                                 factor_log_widths.data)

    return pyro.sample(
        'activations',
        dist.normal,
        weights @ Variable(factors),
        softplus(activation_noise)
    )

def parameterize_tfa_model(activations, locations, num_factors, voxel_noise):
    num_times = activations.shape[0]
    num_voxels = activations.shape[1]

    brain_center = torch.mean(locations, 0).unsqueeze(0)
    brain_center_std_dev = torch.sqrt(10 * torch.var(locations, 0).unsqueeze(0))

    prior = utils.parameterized(tfa_prior)
    prior.register_buffer('weight_mean', Variable(torch.zeros(num_factors)))
    prior.register_buffer(
        'weight_std_dev',
        Variable(SOURCE_WEIGHT_STD_DEV * torch.ones(num_factors))
    )
    prior.register_buffer('factor_center_mean',
                          Variable(brain_center.expand(num_factors, 3)))
    prior.register_buffer('factor_center_std_dev', Variable(
        brain_center_std_dev.expand(num_factors, 3)
    ))
    prior.register_buffer('factor_log_width_mean',
                          Variable(torch.ones(num_factors)))
    prior.register_buffer('factor_log_width_std_dev',
                          Variable(SOURCE_LOG_WIDTH_STD_DEV *\
                                   torch.ones(num_factors)))

    likelihood = utils.parameterized(tfa_likelihood)
    likelihood.register_buffer('locations', locations)
    likelihood.register_buffer('activation_noise',
                               Variable(voxel_noise *\
                                        torch.eye(num_times, num_voxels)))

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.prior = prior
            self.likelihood = likelihood

        def forward(self, *args, **kwargs):
            pyro.module('tfa_model', self)
            kwargs['num_times'] = num_times
            return self.likelihood(*self.prior(*args, **kwargs))

    return Model()

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

    prior = utils.parameterized(tfa_prior)
    prior.register_parameter('weight_mean', Parameter(initial_weights))
    prior.register_parameter(
        'weight_std_dev',
        Parameter(torch.sqrt(torch.rand((num_times, num_factors))))
    )
    prior.register_parameter('factor_center_mean', Parameter(mean_centers))
    prior.register_parameter('factor_center_std_dev', Parameter(
        torch.sqrt(torch.rand(num_factors, 3))
    ))
    prior.register_parameter('factor_log_width_mean',
                             Parameter(mean_log_widths))
    prior.register_parameter('factor_log_width_std_dev',
                             Parameter(torch.sqrt(torch.rand(num_factors))))

    class Guide(nn.Module):
        def __init__(self):
            super(Guide, self).__init__()
            self.prior = prior

        def forward(self, *args, **kwargs):
            pyro.module('tfa_guide', self)
            kwargs['num_times'] = num_times
            kwargs['expand_weight_params'] = False
            return self.prior(*args, **kwargs)

    return Guide()
