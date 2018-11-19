#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical utility functions
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def xyz2sph(v):
    """
    Convert a unit vector from cartesian coordinates to spherical coordinates
    :param v: unit vector
    :return: [theta, phi] polar coordinates
    """

    return np.array([np.arccos(v[2]), np.arctan2(v[1], v[0])])


@jit(nopython=True, cache=True)
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
    return np.array([x, y, z])


@jit(nopython=True, cache=True)
def sph_dot(theta1, theta2, delta_phi):
    """
    Compute the dot product of two unit vectors in spherical coordinates
    """
    return np.sin(theta1) * np.cos(delta_phi) * np.sin(theta2) + np.cos(theta1) * np.cos(theta2)


@jit(nopython=True, cache=True)
def sph_urand():
    """
    Generate random unit vector in spherical coordinates
    :return: (theta, phi) the two angles
    """

    phi = np.random.uniform(0, 2 * np.pi)
    u = np.random.uniform(0, 1)
    theta = np.arccos(2 * u - 1)

    return theta, phi


@jit(nopython=True, cache=True)
def xyz_sph_urand():
    """
    Generate random unit vector in cartesian coordinates
    :return: [x, y, z] the unit vector
    """

    phi = np.random.uniform(0, 2 * np.pi)
    u = np.random.uniform(0, 1)
    theta = np.arccos(2 * u - 1)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


# A maybe faster implementation
# @jit(nopython=True, cache=True)
# def xyz_sph_urand():
#     """
#     Generate random unit vector in cartesian coordinates
#     :return: [x, y, z] the unit vector
#     """
#
#     phi = np.random.uniform(0, 2 * np.pi)
#     u = np.random.uniform(-1, 1)
#     v = np.sqrt(1 - u ** 2)
#
#     x = v * np.cos(phi)
#     y = v * np.sin(phi)
#     z = u
#
#     return np.array([x, y, z])