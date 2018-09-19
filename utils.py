#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils 
"""

import numpy as np


def sph_dot(theta1, theta2, delta_phi):
    """
    Compute the dot product of two unit vectors in spherical coordinates
    """
    return np.sin(theta1) * np.cos(delta_phi) * np.sin(theta2) + np.cos(theta1) * np.cos(theta2)


def sph_dot_lookup_table(ntheta, nphi):
    """
    Build lookup table for dot product of two unit vectors in spherical coordinates
    """
    tab = np.zeros(shape=(ntheta, ntheta, nphi))

    theta = np.linspace(0, np.pi, ntheta)
    phi = np.linspace(0, np.pi, nphi)

    for i, ii, j in np.ndindex(ntheta, ntheta, nphi):
        tab[i, ii, j] = sph_dot(theta[i], theta[ii], phi[j])

    return tab, theta, phi


def sph_dot_lotab(ntheta1, ntheta2, ndelta_phi, tab):
    pass


if __name__ == "__name__":
    tab = sph_dot_lookup_table(100, 100)
