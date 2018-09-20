#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical utility functions
"""

import numpy as np


def sph2xyz(theta, phi):
    """
    Convert spherical coordinates to unit vector
    :param theta: theta angle
    :param phi: phi angle
    :return: (x, y, z) coordinates
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def sph_dot(theta1, theta2, delta_phi):
    """
    Compute the dot product of two unit vectors in spherical coordinates
    """
    return np.sin(theta1) * np.cos(delta_phi) * np.sin(theta2) + np.cos(theta1) * np.cos(theta2)


def rand_sph():
    """
    Generate random unit vector in spherical coordinates
    :return: (theta, phi) the two angles
    """
    u, v = np.random.uniform(size=(2))
    theta = np.arccos(2 * u - 1)
    phi = 2 * np.pi * v

    return theta, phi


def sph_dot_lookup_table(ntheta, nphi):
    """
    Build lookup table for dot product of two unit vectors in spherical coordinates
    :param ntheta: number of theta points
    :param nphi:   number of phi point
    :return: (tab, theta, phi) arrays
    """
    tab = np.zeros(shape=(ntheta, ntheta, nphi))

    theta = np.linspace(0, np.pi, ntheta)
    phi = np.linspace(0, np.pi, nphi)

    for i, ii, j in np.ndindex(ntheta, ntheta, nphi):
        tab[i, ii, j] = sph_dot(theta[i], theta[ii], phi[j])

    return tab, theta, phi


def sph_dot_lotab(ntheta1, ntheta2, ndelta_phi, tab):
    pass


