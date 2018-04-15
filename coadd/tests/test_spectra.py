import numpy as np

from coadd.spectra import *


def test_ctor():
    """Normal construction.
    """
    c = CoAdder(100., 200., 1., 10.)
    assert c.grid[0] == 100.
    assert c.grid[-1] == 200.
    assert c.grid_scale == 1.
    assert np.all(c.psf_grid == np.arange(-10., 11.))


def test_add_psf():
    """Addition with different types of PSF inputs.
    """
    c = CoAdder(100., 200., 0.5, 10.)
    psf = np.zeros(c.n_psf)
    psf[c.n_psf // 2] = 1
    psfs = np.tile(psf, [3, 1])
    data = [1, 3, 2], [150, 160, 170, 180], [0.1, 0.2, 0.1]
    for convolve in True, False:
        c.add(*data, 5, convolve)
        c.add(*data, [4, 5, 6], convolve)
        c.add(*data, psf, convolve)
        c.add(*data, psfs, convolve)


def test_add_analytic_vs_tabulated():
    """Compare analytic vs tabulated Gaussian PSFs.
    """
    c = CoAdder(100., 200., 0.5, 10.)
    data = [1, 3, 2], [150, 160, 170, 180], [0.1, 0.2, 0.1]
    for convolve in True, False:
        # Common PSF for all pixels.
        rms = 1.5
        gp0, _, _ = c.add(*data, rms, convolve, retval=True)
        psf = np.exp(-0.5 * (c.psf_grid / rms) ** 2)
        gp1, _, _ = c.add(*data, psf, convolve, retval=True)
        assert np.allclose(gp0.toarray(), gp1.toarray(), atol=0.05, rtol=0.05)
        # Individual PSFs for each pixel.
        rms = np.array([1.4, 1.5, 1.6])
        gp0, _, _ = c.add(*data, rms, convolve, retval=True)
        psf = np.exp(-0.5 * (c.psf_grid / rms.reshape(-1, 1)) ** 2)
        gp1, _, _ = c.add(*data, psf, convolve, retval=True)
        assert np.allclose(gp0.toarray(), gp1.toarray(), atol=0.05, rtol=0.05)
