"""Optimal coaddition of 1D binned data.
"""
import numpy as np

import scipy.special
import scipy.sparse


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
        self.n_half = int(np.ceil(max_dispersion / self.grid_scale))
        self.n_psf = 2 * self.n_half + 1
        self.psf_grid = np.arange(
            -self.n_half, self.n_half + 1) * self.grid_scale
        self.phi_sum = np.zeros(self.n_grid)
        self.A_sum = scipy.sparse.lil_matrix((self.n_grid, self.n_grid))
        self.root2pi = np.sqrt(2 * np.pi)

    def add(self, data, edges, ivar, psf, convolve_with_pixel=True,
            sigma_clip=3.0, retval=False):
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
              - A array of 2n+1 dispersions tabulated on a uniform grid centered
              at zero with spacing self.grid_scale that applies to all pixels.
              - A 2D array of shape (N, 2n+1) with per-pixel dispersions
              tabulated on the same grid.
            PSF normalization is handled automatically. A tabulated PSF must
            must be non-negative and RMS values must all be positive.
        convolve_with_pixel : bool
            When True, the PSF represents a delta function response and should
            be convolved over each pixel to calculate the range of true
            wavelengths that contribute to the pixel.  When False, the PSF is
            already convolved over each pixel.
        sigma_clip : float
            Number of sigmas to use to truncate Gaussian PSFs specified by RMS.
            Ignored when the input PSF is tabulated.
        retval : bool
            Returns a tuple of arrays (support, phi, A) that summarize this
            observation's contribution to the coadd.

        Returns
        -------
        tuple
            When retval is True, return (support, phi, A) giving this
            observation's contributions to the coadd, tabulated on the internal
            grid.  support is a CSR sparse array with shape (N,n_grid) with
            the normalized support of each pixel.  phi is a 1D array of
            length n_grid with this observation's contribution to phi_sum.
            A is a LIL sparse array of shape (n_grid, n_grid) with this
            observation's contribution to A_sum.
        """
        npixels, data, edges, ivar = self.check_data(data, edges, ivar)

        if convolve_with_pixel:
            # Find the closest grid indices to each edge.
            edge_idx = np.searchsorted(self.grid, edges - 0.5 * self.grid_scale)
            assert np.all(np.abs(
                self.grid[edge_idx] - edges) <= 0.5 * self.grid_scale)
        else:
            # Find the closest grid indices to each pixel midpoint.
            mid = 0.5 * (edges[1:] + edges[:-1])
            mid_idx = np.searchsorted(self.grid, mid - 0.5 * self.grid_scale)
            assert np.all(np.abs(
                self.grid[mid_idx] - mid) <= 0.5 * self.grid_scale)

        # Calculate the (un-normalized) support of each pixel.
        psf = np.atleast_1d(psf)
        supports = []
        if len(psf.shape) == 1 and len(psf) in (1, npixels):
            # psf specifies either a single RMS or per-pixel RMS values.
            if np.any(psf <= 0):
                raise ValueError('Expected all PSF RMS > 0.')
            # Calculate extent of clipped PSF for each pixel in grid units.
            extent = np.ceil(sigma_clip * psf / self.grid_scale).astype(int)
            r2rms = np.sqrt(2) * psf * np.ones(npixels)
            # Tabulate clipped PSF with specified RMS value(s).
            if convolve_with_pixel:
                ilo = edge_idx[:-1] - extent
                ihi = edge_idx[1:] + extent + 1
                if np.any(ilo < 0) or np.any(ihi > self.n_grid):
                    raise ValueError('Pixels disperse outside grid.')
                for i in range(npixels):
                    wlen = self.grid[ilo[i]:ihi[i]]
                    supports.append(0.5 * (
                        scipy.special.erf((edges[i+1] - wlen) / r2rms[i]) -
                        scipy.special.erf((edges[i] - wlen) / r2rms[i])))
                    assert len(supports[-1] == ihi[i] - ilo[i])
            else:
                ilo = mid_idx - extent
                ihi = mid_idx + extent + 1
                if np.any(ilo < 0) or np.any(ihi > self.n_grid):
                    raise ValueError('Pixels disperse outside grid.')
                for i in range(npixels):
                    wlen = self.grid[ilo[i]:ihi[i]]
                    supports.append(
                        np.exp(-((wlen - mid[i]) / r2rms[i]) ** 2))
                    assert len(supports[-1] == ihi[i] - ilo[i])
        else:
            if len(psf.shape) > 2:
                raise ValueError('Invalid psf shape.')
            if np.any(psf < 0):
                raise ValueError('Tabulated psf values must all be >= 0.')
            shared = len(psf.shape) == 1
            n_psf = psf.shape[-1]
            if n_psf % 2 != 1:
                raise ValueError('Tabulated psf must be have odd length.')
            # Calculate extent of clipped PSF for each pixel in grid units.
            extent = (n_psf - 1) // 2
            # psf is tabulated on internal grid.
            if convolve_with_pixel:
                ilo = edge_idx[:-1] - extent
                ihi = edge_idx[1:] + extent + 1
                for i in range(npixels):
                    img = psf if shared else psf[i]
                    pix = np.ones(edge_idx[i + 1] - edge_idx[i] + 1)
                    supports.append(np.convolve(pix, img, mode='full'))
                    assert len(supports[-1] == ihi[i] - ilo[i])
            else:
                ilo = mid_idx - extent
                ihi = mid_idx + extent + 1
                for i in range(npixels):
                    supports.append(psf if shared else psf[i])
                    assert len(supports[-1] == ihi[i] - ilo[i])

        # Normalize each pixel's support in place.
        norm = (edges[1:] - edges[:-1]) / self.grid_scale
        for i in range(npixels):
            supports[i] *= norm[i] / supports[i].sum()

        if retval:
            # Initialize arrays to return.
            phi = np.zeros_like(self.phi_sum)
            A = scipy.sparse.lil_matrix((self.n_grid, self.n_grid))
            iptr = np.empty(npixels + 1, int)
            iptr[0] = 0
            iptr[1:] = np.cumsum(ihi - ilo)
            nsparse = iptr[-1]
            sparse = np.concatenate(supports)
            assert sparse.shape == (nsparse,)
            idx = np.empty(nsparse, int)
            for i in range(npixels):
                idx[iptr[i]:iptr[i + 1]] = np.arange(ilo[i], ihi[i], dtype=int)
            support = scipy.sparse.csr_matrix(
                (sparse, idx, iptr), (npixels, self.n_grid))
            '''
            # Check sparse against dense...
            gp = np.zeros((npixels, self.n_grid))
            for i in range(npixels):
                gp[i,ilo[i]:ihi[i]] = supports[i]
            assert np.all(support.toarray() == gp)
            '''

        # Accumulate each pixel's contribution to phi, A.
        for i, S in enumerate(supports):
            if ivar[i] == 0:
                continue
            dphi = data[i] * ivar[i] * S
            if retval:
                phi[ilo[i]:ihi[i]] += dphi
            self.phi_sum[ilo[i]:ihi[i]] += dphi
            dA = ivar[i] * np.outer(S, S)
            if retval:
                A[ilo[i]:ihi[i],ilo[i]:ihi[i]] += dA
            self.A_sum[ilo[i]:ihi[i],ilo[i]:ihi[i]] += dA

        if retval:
            return support, phi, A

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
