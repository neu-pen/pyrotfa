#!/usr/bin/env python

import argparse
import logging

import pyrotfa.utils as utils

parser = argparse.ArgumentParser(description='Topographical factor analysis for fMRI data')
parser.add_argument('data_file', type=str, help='fMRI filename')
parser.add_argument('--steps', type=int, default=100, help='Number of optimization steps')
parser.add_argument('--log-optimization', action='store_true', help='Whether to log optimization')

if __name__ == '__main__':
    args = parser.parse_args()
