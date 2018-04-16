"""Utilities for optimal coaddition.
"""
import numpy as np

import scipy.interpolate


def centers_to_edges(centers, kind='cubic'):
    """Calculate bin edges from bin centers.

    Interpolate from integer to half-integer indices.

    Parameters
    ----------
    centers : array
        1D array of increasing center values.
    kind : str or int
        Passed to :func:`scipy.interpolate.interp1d`.
    """
    centers = np.asarray(centers)
    if len(centers.shape) != 1:
        raise ValueError('Expected 1D array of centers.')
    if not np.all(np.diff(centers) > 0):
        raise ValueError('Expected increasing center values.')

    center_idx = np.arange(len(centers))
    interpolator = scipy.interpolate.interp1d(
        center_idx, centers, fill_value='extrapolate', copy=False,
        assume_sorted=True, kind=kind)

    edge_idx = np.arange(len(centers) + 1.) - 0.5
    return interpolator(edge_idx)
