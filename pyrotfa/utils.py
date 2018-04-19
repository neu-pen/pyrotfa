#!/usr/bin/env python
"""Utilities for topographic factor analysis"""

import flatdict
import inspect
import logging
import types
import warnings

try:
    if __name__ == '__main__':
        import matplotlib
        matplotlib.use('TkAgg')
finally:
    import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import nibabel as nib
from nilearn.input_data import NiftiMasker
import torch
from torch.autograd import Variable
import torch.nn as nn

import pyro

def reconstruct(weights, centers, log_widths, locations, activations):
    factors = radial_basis(Variable(locations), centers, log_widths)
    reconstruction = weights @ factors

    logging.info(
        'Reconstruction Error (Frobenius Norm): %.8e',
        np.linalg.norm(reconstruction.data - activations)
    )

    return reconstruction

def vardict(existing=None):
    vdict = flatdict.FlatDict(delimiter='__')
    if existing:
        for (k, v) in existing.items():
            vdict[k] = v
    return vdict

def vardict_keys(vdict):
    first_level = [k.rsplit('__', 1)[0] for k in vdict.keys()]
    return list(set(first_level))

class PyroPartial(nn.Module):
    def __init__(self, f, params_namespace='params'):
        super(PyroPartial, self).__init__()
        self._function = f
        self._params_namespace = params_namespace
        pyro.module(self._function.__name__, self)

    @classmethod
    def compose(cls, outer, inner, unpack=False, name=None):
        if unpack:
            result = lambda *args, **kwargs: outer(*inner(*args, **kwargs))
        else:
            result = lambda *args, **kwargs: outer(inner(*args, **kwargs))

        if name is not None:
            result.__name__ = name

        return PyroPartial(result)

    def register_params(self, pdict, trainable=True):
        for (k, v) in vardict(pdict).iteritems():
            if self._params_namespace is not None:
                k = self._params_namespace + '__' + k
            if trainable:
                self.register_parameter(k, nn.Parameter(v))
            else:
                self.register_buffer(k, v)

    def params_vardict(self, keep_vars=False):
        result = self.state_dict(keep_vars=keep_vars)
        if self._params_namespace is not None:
            prefix = self._params_namespace + '__'
            result = {k[len(prefix):]: v for (k, v) in result.items()
                      if k.startswith(prefix)}
        for k, v in result.items():
            if isinstance(v, nn.Parameter):
                pyro.param(k, v)
            elif not isinstance(v, Variable):
                result[k] = Variable(v)
        return vardict(result)

    def kwargs_dict(self):
        members = dict(self.__dict__, **self.state_dict(keep_vars=True))
        return {k: v for (k, v) in members.items()
                if k in inspect.signature(self._function).parameters.keys()}

    def forward(self, *args, **kwargs):
        params = {**self.kwargs_dict(), **kwargs}
        if self._params_namespace is not None:
            params[self._params_namespace] = self.params_vardict(keep_vars=True)
        return self._function(*args, **params)

def unsqueeze_and_expand(tensor, dim, size, clone=False):
    if clone:
        tensor = tensor.clone()

    shape = list(tensor.shape)
    shape.insert(dim, size)
    return tensor.unsqueeze(dim).expand(*shape)

def radial_basis(locations, centers, log_widths):
    """The radial basis function used as the shape for the factors"""
    delta2s = (locations.unsqueeze(0) - centers.unsqueeze(1))**2
    return torch.exp(-delta2s.sum(2) / torch.exp(log_widths.unsqueeze(1)))

def plot_losses(losses):
    epochs = range(losses.shape[0])

    elbo_fig = plt.figure(figsize=(10, 10))

    plt.plot(epochs, losses, 'b-', label='Free energy')
    plt.legend()

    elbo_fig.tight_layout()
    plt.title('Free-energy change over training')
    elbo_fig.axes[0].set_xlabel('Epoch')
    elbo_fig.axes[0].set_ylabel('Nats')

    plt.show()

def full_fact(dimensions):
    """
    Replicates MATLAB's fullfact function (behaves the same way)
    """
    vals = np.asmatrix(range(1, dimensions[0] + 1)).T
    if len(dimensions) == 1:
        return vals
    else:
        after_vals = np.asmatrix(full_fact(dimensions[1:]))
        independents = np.asmatrix(np.zeros((np.prod(dimensions), len(dimensions))))
        row = 0
        for i in range(after_vals.shape[0]):
            independents[row:(row + len(vals)), 0] = vals
            independents[row:(row + len(vals)), 1:] = np.tile(
                after_vals[i, :], (len(vals), 1)
            )
            row += len(vals)
        return independents

def nii2cmu(nifti_file, mask_file=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = nib.load(nifti_file)
        mask = NiftiMasker(mask_strategy='background')
        if mask_file is None:
            mask.fit(nifti_file)
        else:
            mask.fit(mask_file)

    header = image.header
    sform = image.get_sform()
    voxel_size = header.get_zooms()

    voxel_activations = np.float64(mask.transform(nifti_file)).transpose()

    vmask = np.nonzero(np.array(np.reshape(
        mask.mask_img_.dataobj,
        (1, np.prod(mask.mask_img_.shape)),
        order='C'
    )))[1]
    voxel_coordinates = full_fact(image.shape[0:3])[vmask, ::-1] - 1
    voxel_locations = np.array(np.dot(voxel_coordinates, sform[0:3, 0:3])) + sform[:3, 3]

    return {'data': voxel_activations, 'R': voxel_locations}

def cmu2nii(activations, locations, template):
    image = nib.load(template)
    sform = image.affine
    coords = np.array(
        np.dot(locations - sform[:3, 3],
               np.linalg.inv(sform[0:3, 0:3])),
        dtype='int'
    )
    data = np.zeros(image.shape[0:3] + (activations.shape[0],))

    for i in range(activations.shape[0]):
        for j in range(locations.shape[0]):
            x, y, z = coords[j, 0], coords[j, 1], coords[j, 2]
            data[x, y, z, i] = activations[i, j]

    return nib.Nifti1Image(data[:, :, :, 0], affine=sform)

def uncertainty_alphas(uncertainties):
    if len(uncertainties.shape) > 1:
        uncertainties = np.array([
            [u] for u in np.linalg.norm((uncertainties**-2).numpy(), axis=1)
        ])
    else:
        uncertainties = uncertainties.numpy()
    return uncertainties / (1.0 + uncertainties)

def compose_palette(length, base='dark', alphas=None):
    palette = np.array(sns.color_palette(base, length))
    if alphas is not None:
        return np.concatenate([palette, alphas], axis=1)
    return palette

def uncertainty_palette(uncertainties):
    alphas = uncertainty_alphas(uncertainties)
    return compose_palette(uncertainties.shape[0], base='cubehelix',
                           alphas=alphas)
