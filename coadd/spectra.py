"""Optimal coaddition of 1D binned data.
"""
import numpy as np

import scipy.special


class CoAdder(object):

    def __init__(self, wlen_lo, wlen_hi, max_dispersion, n_grid):
        """Initialize coadder for 1D binned data.

        Parameters
        ----------
        wlen_lo : float
            Low edge to use for the internal grid. Should be below the lowest
            value expected in data. Must be < wlen_hi. Units are arbitrary.
        wlen_hi : float
            High edge to use for the internal grid. Should be above the lowest
            value expected in data. Must be > wlen_lo. Units are arbitrary.
        max_dispersion : float
            The maximum expected dispersion in wlen units. Any PSF with
            tails beyond this value will be truncated.
        n_grid : int
            Number of internal grid points to use. Should be ~5x the smallest
            expected RMS dispersion. Must be > 0. Internal storage requirements
            scale with n_grid ** 2.
        """
        if n_grid <= 0:
            raise ValueError('Expected n_grid > 0.')
        self.n_grid = n_grid
        if wlen_lo >= wlen_hi:
            raise ValueError('Expected wlen_lo < wlen_hi.')
        self.grid = np.linspace(wlen_lo, wlen_hi, n_grid)
        self.grid_scale = (wlen_hi - wlen_lo) / (n_grid - 1.)
        if max_dispersion <= 0:
            raise ValueError('Expected max_dispersion > 0.')
        n = int(np.ceil(max_dispersion / self.grid_scale))
        self.n_psf = 2 * n + 1
        self.psf_grid = np.arange(-n, n+1) * self.grid_scale
        self.phi_sum = np.zeros(num_grid)
        self.A_sum = np.zeros((num_grid, num_grid))
        self.root2pi = np.sqrt(2 * np.pi)

    def add(self, data, edges, ivar, psf, convolve_with_pixel=True):
        """Add a single observation to the coadd.

        Parameters
        ----------
        data : array
            Array of N pixel values for this observation.
        edges : array
            Array of N+1 increasing pixels edges for this observation.
        ivar : array
            Array of N inverse variance values for this observation. Must
            all be >= 0. Covariances between pixels are assumed to be zero.
        psf : float or array
            The dispersion at each pixel center can be specified four different
            ways:
              - A single Gaussian RMS value in wlen units that applies
              to all pixels.
              - An array of N Gaussian RMS values in wlen units for each pixel.
              - A array of length self.n_psf of dispersions tabulated at
              self.psf_grid that applies to all pixels.
              - An array with shape (N, self.n_psf) of dispersions tabulated
              at self.psf_grid for each pixel.
            Raises ValueError if the input is ambiguous because N == self.n_psf.
        convolve_with_pixel : bool
            When True, the PSF represents a delta function response and should
            be convolved over each pixel to calculate the range of true that
            contribute to the pixel.  When False, the PSF is already convolved
            over each pixel.

        Returns
        -------
        tuple
            Tuple (psf, phi, A) giving this observation's contributions to
            the coadd, tabulated on the internal grid.
        """
        data = np.asarray(data)
        edges = np.asarray(edges)
        ivar = np.asarray(ivar)
        npixels = len(data)
        if len(edges) != npixels + 1:
            raise ValueError('Length of edges and data arrays do not match.')
        if len(ivar) != npixels:
            raise ValueError('Length of ivar and data arrays do not match.')
        if np.any(ivar < 0):
            raise ValueError('All ivar values must >= 0.')
        if not np.all(np.diff(edges) > 0):
            raise ValueError('Pixel edges are not in increasing order.')

        psf = np.atleast_1d(psf)
        if len(psf.shape) == 1:
            if len(psf) not in (1, npixels, self.n_psf):
                raise ValueError('Invalid psf array length.')
            if len(psf.shape) == npixels and npixels == self.n_psf:
                raise ValueError('Ambiguous psf since npixels = n_psf.')
            if len(psf) == self.n_psf:
                if not np.allclose(psf.sum(), 1):
                    raise ValueError('Input psf is not normalized.')
                psf = psf.reshape(1, self.n_psf)
            else:
                if np.any(psf <= 0):
                    raise ValueError('Expected all PSF RMS > 0.')
                # Tabulate normalized PSF with specified RMS value(s).
                rms = psf.reshape(1, -1)
                psf = self.grid_scale * np.exp(
                    -0.5 * (self.psf_grid / rms) ** 2) / (self.root2pi * rms)
        elif psf.shape == (npixels, self.n_psf):
            if not np.allclose(psf.sum(axis=1), 1):
                raise ValueError('Input psfs are not normalized.')
        else:
            raise ValueError('Unexpected psf shape.')
        '''
        phi = np.zeros_like(self.phi_sum)
        A = np.zeros_like(self.A_sum)
        gp = np.empty((npixels, len(self.x_grid)))
        M = np.digitize(self.x_grid[:, np.newaxis] + self.x_grid, edges)
        for p in range(npixels):
            gp[p] = ((M == p + 1) * psf).sum(axis=1)
            phi += gp[p] * image[p] * ivar[p]
            A += np.outer(gp[p], gp[p]) * ivar[p]
        self.phi_sum += phi
        self.A_sum += A
        assert np.allclose(self.A_sum, self.A_sum.T)
        '''
