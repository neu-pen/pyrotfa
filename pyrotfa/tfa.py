"""Perform plain topographic factor analysis on a given fMRI data file."""

import logging
import math
import os
import pickle
import time

# import some dependencies
import torch
from torch.autograd import Variable
import torch.cuda

import hypertools as hyp
import nilearn.plotting as niplot
import numpy as np
import scipy.io as sio

import pyro
import pyro.infer
import pyro.optim

from . import tfa_models
from . import utils

EPOCHS = 20
EPOCH_MSG = '[Epoch %d] (%dms) Posterior ELBO %.8e'
LEARNING_RATE = 1e-4
LOSS = 'ELBO'
NUM_FACTORS = 10
NUM_PARTICLES = 10

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

        self.model = tfa_models.parameterize_tfa_model(
            self.activations, self.locations, num_factors=num_factors,
            voxel_noise=tfa_models.VOXEL_NOISE
        )
        pyro.module('tfa_model', self.model)
        self.guide = tfa_models.parameterize_tfa_guide(
            self.activations, self.locations, num_factors=num_factors
        )
        pyro.module('tfa_guide', self.guide)

        self.reconstruction = None

    def infer(self, epochs=EPOCHS, learning_rate=LEARNING_RATE, loss=LOSS,
              log_level=logging.WARNING, num_particles=NUM_PARTICLES):
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)

        pyro.clear_param_store()
        data = {'activations': Variable(self.activations)}
        if torch.cuda.is_available():
            tfa_models.softplus.cuda()
            data['activations'].cuda()
        conditioned_tfa = pyro.condition(self.model, data=data)

        svi = pyro.infer.SVI(model=conditioned_tfa, guide=self.guide,
                             optim=pyro.optim.Adam({'lr': learning_rate}),
                             loss=loss, num_particles=num_particles)

        losses = np.zeros(epochs)
        for e in range(epochs):
            start = time.time()

            losses[e] = svi.step()
            self.reconstruct(*self.guide())

            end = time.time()
            logging.info(EPOCH_MSG, e + 1, (end - start) * 1000, losses[e])

        if torch.cuda.is_available():
            data['activations'].cpu()
            tfa_models.softplus.cpu()

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
            self.guide.prior.factor_center_mean.data.numpy(),
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
        self.reconstruct(self.guide.prior.weight_mean,
                         self.guide.prior.factor_center_mean,
                         self.guide.prior.factor_log_width_mean,
                         save=True)

        image = utils.cmu2nii(self.reconstruction,
                              self.locations.numpy(),
                              self._template)
        plot = niplot.plot_glass_brain(image, plot_abs=plot_abs)

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot
