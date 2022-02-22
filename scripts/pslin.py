#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

import numpy as np

import sys
import flowpm
import flowpm.tfpower as tfpower
import flowpm.scipy.interpolate as interpolate
from skopt.sampler import Lhs

# Cosmological parameters:
#     h:        tf.Tensor(0.6774, shape=(), dtype=float32)
#     Omega_b:  tf.Tensor(0.0486, shape=(), dtype=float32)
#     Omega_c:  tf.Tensor(0.2589, shape=(), dtype=float32)
#     Omega_k:  tf.Tensor(0.0, shape=(), dtype=float32)
#     w0:       tf.Tensor(-1.0, shape=(), dtype=float32)
#     wa:       tf.Tensor(0.0, shape=(), dtype=float32)
#     n:        tf.Tensor(0.9667, shape=(), dtype=float32)
#     sigma8:   tf.Tensor(0.8159, shape=(), dtype=float32)

#


@tf.function
def ps(params):
  #Omega_c, sigma8, Omega_b, h = params
  #cosmology = flowpm.cosmology.Planck15(Omega_c=Omega_c, sigma8=sigma8, Omega_b=Omega_b, h=h)
  cosmology = flowpm.cosmology.Planck15(Omega_c=params[0], sigma8=params[1], Omega_b=params[2], h=params[3])
  k = tf.constant(np.logspace(-3, 0, 50), dtype=tf.float32)
  pk = tfpower.linear_matter_power(cosmology, k)
  return k, pk



#%%

if __name__ == "__main__":

  np.random.seed(100)
  omc_range = [0.2, 0.3]
  s8_range = [0.6, 1.0]
  omb_range = [0.03, 0.06]
  h_range = [0.55, 0.85]
  
  for mode in ['train', 'test']:
    print(mode)
    if mode == 'train': npoints, folder = 5000, '../data/traindata/'
    elif mode == 'test': npoints, folder = 250, '../data/testdata/'
    lhs = Lhs(criterion="ratio", iterations=1000)
    points = np.array(lhs.generate([omc_range, s8_range, omb_range, h_range], npoints)).astype(np.float32)
    print(points.shape)
    for i in range(points.shape[0]):
      if i%100 == 0: print(i)
      k, pk = ps(tf.constant(points[i]))
      np.save(folder + 'pk%04d'%i, np.array([k.numpy(), pk.numpy()]))
    np.save(folder + 'cosmology', points)
    
