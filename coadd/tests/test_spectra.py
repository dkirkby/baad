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
    args = [1, 3, 2], [150, 160, 170, 180], [0.1, 0.2, 0.1]
    for convolve in True, False:
        c.add(*args, 5, convolve)
        c.add(*args, [4, 5, 6], convolve)
        c.add(*args, psf, convolve)
        c.add(*args, psfs, convolve)


def test_add_analytic_vs_tabulated():
    """Compare analytic vs tabulated Gaussian PSFs.
    """
    c = CoAdder(100., 200., 0.5, 10.)
    args = [1, 3, 2], [150, 160, 170, 180], [0.1, 0.2, 0.1]
    pixelwidth = 10.
    for convolve in True, False:
        # Common PSF for all pixels.
        rms = 1.5
        gp0, _, _ = c.add(*args, rms, convolve)
        psf = np.exp(-0.5 * (c.psf_grid / rms) ** 2)
        if convolve:
            psf /= psf.sum()
        else:
            psf *= pixelwidth / (np.sqrt(2 * np.pi) * rms)
        gp1, _, _ = c.add(*args, psf, convolve)
        assert np.allclose(gp0, gp1, atol=0.05, rtol=0.05)
        # Individual PSFs for each pixel.
        rms = np.array([1.4, 1.5, 1.6])
        gp0, _, _ = c.add(*args, rms, convolve)
        psf = np.exp(-0.5 * (c.psf_grid / rms.reshape(-1, 1)) ** 2)
        if convolve:
            psf /= psf.sum(axis=1).reshape(-1, 1)
        else:
            psf *= pixelwidth / (np.sqrt(2 * np.pi) * rms.reshape(-1, 1))
        gp1, _, _ = c.add(*args, psf, convolve)
        assert np.allclose(gp0, gp1, atol=0.05, rtol=0.05)
