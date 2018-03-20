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
import torch.cuda
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import torch.utils.data

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim

from . import utils

EPOCHS=20
EPOCH_MSG = '[Epoch %d] (%dms) Posterior ELBO %.8e'
LEARNING_RATE = 1e-4
LOSS = 'ELBO'
NUM_FACTORS = 10
NUM_PARTICLES = 10
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
            kwargs['num_times'] = num_times
            kwargs['expand_weight_params'] = False
            return self.prior(*args, **kwargs)

    return Guide()

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

        self.tfa_model = parameterize_tfa_model(self.activations,
                                                self.locations,
                                                num_factors=num_factors,
                                                voxel_noise=VOXEL_NOISE)
        self.tfa_guide = parameterize_tfa_guide(self.activations,
                                                self.locations,
                                                num_factors=num_factors)

        self.reconstruction = None

    def infer(self, epochs=EPOCHS, learning_rate=LEARNING_RATE, loss=LOSS,
              log_level=logging.WARNING, num_particles=NUM_PARTICLES):
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)

        pyro.clear_param_store()
        data = {'activations': Variable(self.activations)}
        if torch.cuda.is_available():
            softplus.cuda()
            data['activations'].cuda()
        conditioned_tfa = pyro.condition(self.tfa_model, data=data)

        svi = pyro.infer.SVI(model=conditioned_tfa, guide=self.tfa_guide,
                             optim=pyro.optim.Adam({'lr': learning_rate}),
                             loss=loss, num_particles=num_particles)

        losses = np.zeros(epochs)
        for e in range(epochs):
            start = time.time()

            losses[e] = svi.step()
            self.reconstruct(*self.tfa_guide())

            end = time.time()
            logging.info(EPOCH_MSG, e + 1, (end - start) * 1000, losses[e])

        if torch.cuda.is_available():
            data['activations'].cpu()
            softplus.cpu()

        return losses

    def reconstruct(self, weights, centers, log_widths):
        factors = utils.radial_basis(Variable(self.locations), centers,
                                     log_widths)
        self.reconstruction = weights @ factors

        logging.info(
            'Reconstruction Error (Frobenius Norm): %.8e',
            np.linalg.norm(self.reconstruction.data - self.activations)
        )

        return self.reconstruction

    def guide_means(self, log_level=logging.WARNING, matfile=None,
                    reconstruct=False):
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)

        params = pyro.get_param_store()
        means = {}
        for (name, var) in params.named_parameters():
            if 'mean' in name:
                means[name] = var.data

        if matfile is not None:
            sio.savemat(matfile, means, do_compression=True)

        if reconstruct:
            self.reconstruct(Variable(means['mean_weight']),
                             Variable(means['mean_centers']),
                             Variable(means['mean_factor_log_width']))

        return means

    def plot_voxels(self):
        hyp.plot(self.locations.numpy(), 'k.')

    def plot_factor_centers(self, filename=None, show=True,
                            log_level=logging.WARNING):
        means = self.guide_means(log_level=log_level)

        plot = niplot.plot_connectome(
            np.eye(self.num_factors),
            means['mean_centers'],
            node_color='k'
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_original_brain(self, filename=None, show=True, plot_abs=False):
        original_image = utils.cmu2nii(self.activations.numpy(),
                                       self.locations.numpy(),
                                       self._template)
        plot = niplot.plot_glass_brain(original_image, plot_abs=plot_abs)

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_reconstruction(self, filename=None, show=True, plot_abs=False,
                            log_level=logging.WARNING):
        self.guide_means(log_level=log_level, reconstruct=True)

        image = utils.cmu2nii(self.reconstruction,
                              self.locations.numpy(),
                              self._template)
        plot = niplot.plot_glass_brain(image, plot_abs=plot_abs)

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot
