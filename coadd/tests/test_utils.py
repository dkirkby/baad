import numpy as np

from coadd.utils import *


def test_edges_pow():
    """Check that edges for power=1,2,3 are recovered
    """
    center_idx = np.arange(10, 20)
    edge_idx = np.arange(10, 21) - 0.5
    for pow in (1, 2, 3):
        centers = center_idx ** pow
        edges = centers_to_edges(centers, kind=pow)
        true_edges = edge_idx ** pow
        assert np.allclose(edges, edge_idx ** pow)


def test_edges_log():
    """Check that SDSS log-wavelength edges are recovered
    """
    wlen0, npix = 3500.26, 4800
    idx = np.arange(npix)
    centers = wlen0 * 10 ** (1e-4 * idx)
    true_edges = wlen0 * 10 ** (1e-4 * (np.arange(npix + 1) - 0.5))
    edges = centers_to_edges(centers)
    assert np.allclose(edges, true_edges, rtol=1e-10, atol=1e-10)
