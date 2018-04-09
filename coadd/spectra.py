"""Optimal coaddition of 1D binned data.
"""
import numpy as np

import scipy.special


class CoAdder(object):

    def __init__(self, wlen_lo, wlen_hi, wlen_step, max_dispersion):
        """Initialize coadder for 1D binned data.

        Parameters
        ----------
        wlen_lo : float
            Low edge to use for the internal grid. Should be below the lowest
            value expected in data. Must be < wlen_hi. Units are arbitrary.
        wlen_hi : float
            High edge to use for the internal grid. Should be above the lowest
            value expected in data. Must be > wlen_lo. Units are arbitrary.
        wlen_step : float
            Internal step size will rounded down from this value to uniformly
            cover [wlen_lo, wlen_hi].  Should be ~5x the smallest expected
            RMS dispersion.
        max_dispersion : float
            The maximum expected dispersion in wlen units. Any PSF with
            tails beyond this value will be truncated.
        """
        if wlen_lo >= wlen_hi:
            raise ValueError('Expected wlen_lo < wlen_hi.')
        if wlen_step <= 0:
            raise ValueError('Expected wlen_step > 0.')
        self.n_grid = int(np.ceil((wlen_hi - wlen_lo) / wlen_step)) + 1
        self.grid, self.grid_scale = np.linspace(
            wlen_lo, wlen_hi, self.n_grid, retstep=True)
        if max_dispersion <= 0:
            raise ValueError('Expected max_dispersion > 0.')
        n = int(np.ceil(max_dispersion / self.grid_scale))
        self.n_psf = 2 * n + 1
        self.psf_grid = np.arange(-n, n+1) * self.grid_scale
        self.phi_sum = np.zeros(self.n_grid)
        self.A_sum = np.zeros((self.n_grid, self.n_grid))
        self.root2pi = np.sqrt(2 * np.pi)

    def add(self, data, edges, ivar, psf, convolve_with_pixel=True):
        """Add a single observation to the coadd.

        Parameters
        ----------
        data : array
            Array of N pixel values for this observation.
        edges : array
            Array of N+1 increasing pixels edges for this observation.
            The first and last edges must be inset enough for the maximum
            dispersion.
        ivar : array
            Array of N inverse variances for this observation's data. Must
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
        npixels, data, edges, ivar = self.check_data(data, edges, ivar)

        # Tabulate the support of each pixel i as gp[i].
        psf = np.atleast_1d(psf)
        gp = np.zeros((npixels, self.n_grid))
        if len(psf.shape) == 1:
            # psf specifies either a single RMS, per-pixel RMS, or a
            # tabulated psf shared by all pixels.
            if len(psf) not in (1, npixels, self.n_psf):
                raise ValueError('Invalid psf array length.')
            if len(psf.shape) == npixels and npixels == self.n_psf:
                raise ValueError('Ambiguous psf since npixels = n_psf.')
            if len(psf) == self.n_psf:
                if convolve_with_pixel:
                    if not np.allclose(psf.sum(), 1):
                        raise ValueError('Input psf is not normalized.')
                    for i in range(npixels):
                        self.psf_convolve(edges[i], edges[i + 1], psf, gp[i])
                else:
                    for i in range(npixels):
                        self.psf_center(edges[i], edges[i + 1], psf, gp[i])
            else:
                # psf specifies either a single RMS or per-pixel RMS values.
                if np.any(psf <= 0):
                    raise ValueError('Expected all PSF RMS > 0.')
                # Tabulate PSF with specified RMS value(s).
                rms = psf.reshape(-1, 1)
                wlen = self.grid
                if convolve_with_pixel:
                    erf_lo = scipy.special.erf(
                        (edges[:-1, np.newaxis] - wlen) / (np.sqrt(2) * rms))
                    erf_hi = scipy.special.erf(
                        (edges[1:, np.newaxis] - wlen) / (np.sqrt(2) * rms))
                    gp[:] = 0.5 * (erf_hi - erf_lo)
                else:
                    wlen0 = 0.5 * (edges[1:] + edges[:-1]).reshape(-1, 1)
                    dwlen = (edges[1:] - edges[:-1]).reshape(-1, 1)
                    gp[:] = dwlen * np.exp(
                        -0.5 * ((wlen - wlen0) / rms) ** 2) / (self.root2pi * rms)
        elif psf.shape == (npixels, self.n_psf):
            if convolve_with_pixel:
                if not np.allclose(psf.sum(axis=1), 1):
                    raise ValueError('Input psfs are not normalized.')
                for i in range(npixels):
                    self.psf_convolve(edges[i], edges[i + 1], psf[i], gp[i])
            else:
                for i in range(npixels):
                    self.psf_center(edges[i], edges[i + 1], psf[i], gp[i])
        else:
            raise ValueError('Unexpected psf shape.')

        # Calculate this observation's contributions to phi, A.
        phi = np.zeros_like(self.phi_sum)
        A = np.zeros_like(self.A_sum)
        for i in range(npixels):
            phi += gp[i] * data[i] * ivar[i]
            A += np.outer(gp[i], gp[i]) * ivar[i]
        self.phi_sum += phi
        self.A_sum += A

        return gp, phi, A

    def psf_convolve(self, lo, hi, psf, out):
        """Convolve psf with a pixel spanning [lo,hi] and save results in out.
        """
        # Find the closest grid indices to lo, hi.
        edges = np.array([lo, hi]) - 0.5 * self.grid_scale
        idx_lo, idx_hi = np.searchsorted(self.grid, edges)
        assert np.abs(lo - self.grid[idx_lo]) <= 0.5 * self.grid_scale
        assert np.abs(hi - self.grid[idx_hi]) <= 0.5 * self.grid_scale
        pixel = np.ones(idx_hi - idx_lo + 1, dtype=out.dtype)
        n = (self.n_psf - 1) // 2
        assert idx_lo >= n and idx_hi < self.n_grid - n
        out[idx_lo - n: idx_hi + n + 1] = np.convolve(pixel, psf, mode='full')

    def psf_center(self, lo, hi, psf, out):
        """Center psf in a pixel spanning [lo,hi] and save results in out.
        """
        wmid = 0.5 * (lo + hi)
        # Find the closest grid index to wmid.
        idx = np.searchsorted(self.grid, [wmid - 0.5 * self.grid_scale])[0]
        assert np.abs(wmid - self.grid[idx]) <= 0.5 * self.grid_scale
        n = (self.n_psf - 1) // 2
        out[idx - n: idx + n + 1] = psf

    def check_data(self, data, edges, ivar):
        """Perform checks for valid input data.
        """
        npixels = len(data)
        data = np.asarray(data)
        edges = np.asarray(edges)
        ivar = np.asarray(ivar)
        if len(edges) != npixels + 1:
            raise ValueError('Length of edges and data arrays do not match.')
        if len(ivar) != npixels:
            raise ValueError('Length of ivar and data arrays do not match.')
        if np.any(ivar < 0):
            raise ValueError('All ivar values must >= 0.')
        if not np.all(np.diff(edges) > 0):
            raise ValueError('Pixel edges are not in increasing order.')
        if edges[0] + self.psf_grid[0] < self.grid[0]:
            raise ValueError('First edge not inset enough for dispersion.')
        if edges[-1] + self.psf_grid[-1] > self.grid[-1]:
            raise ValueError('Last edge not inset enough for dispersion.')
        return npixels, data, edges, ivar
