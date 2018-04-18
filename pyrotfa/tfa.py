"""Perform plain topographic factor analysis on a given fMRI data file."""

import gc
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
              log_level=logging.WARNING, num_particles=NUM_PARTICLES,
              use_cuda=torch.cuda.is_available(), checkpoint_epochs=100):
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)

        pyro.clear_param_store()
        data = {'activations': Variable(self.activations)}
        if use_cuda:
            data['activations'] = data['activations'].cuda()
            self.model.cuda()
            self.guide.cuda()
        conditioned_tfa = pyro.condition(self.model, data=data)

        svi = pyro.infer.SVI(model=conditioned_tfa, guide=self.guide,
                             optim=pyro.optim.Adam({'lr': learning_rate}),
                             loss=loss, num_particles=num_particles)

        losses = np.zeros(epochs)
        for e in range(epochs):
            start = time.time()

            losses[e] = svi.step()

            end = time.time()
            logging.info(EPOCH_MSG, e + 1, (end - start) * 1000, losses[e])

            if e % checkpoint_epochs == 0:
                gc.collect()

        if use_cuda:
            self.guide.cpu()
            self.model.cpu()
            data['activations'] = data['activations'].cpu()

        return losses

    def plot_voxels(self):
        hyp.plot(self.locations.numpy(), 'k.')

    def plot_factor_centers(self, filename=None, show=True):
        centers = self.guide.prior.factor_center_mean.data
        center_uncertainties = self.guide.prior.factor_center_std_dev.data
        widths = self.guide.prior.factor_log_width_mean.data

        plot = niplot.plot_connectome(
            np.eye(self.num_factors),
            centers.numpy(),
            node_color=utils.uncertainty_palette(center_uncertainties),
            node_size=np.exp(widths.numpy() - np.log(2))
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
                            refresh=False):
        if self.reconstruction is None or refresh:
            self.reconstruction = utils.reconstruct(
                self.guide.prior.weight_mean,
                self.guide.prior.factor_center_mean,
                self.guide.prior.factor_log_width_mean,
                self.locations,
                self.activations
            )

        image = utils.cmu2nii(self.reconstruction,
                              self.locations.numpy(),
                              self._template)
        plot = niplot.plot_glass_brain(image, plot_abs=plot_abs)

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot
