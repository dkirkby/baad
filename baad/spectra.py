"""Combine observations of 1D data.
"""
import numpy as np

import scipy.special
import scipy.sparse
import scipy.optimize

import baad.sparse


class CoAdd1D(object):

    def __init__(self, wlen_lo, wlen_hi, wlen_step, max_spread,
                 dtype=np.float):
        """Initialize coadder for 1D binned data.

        Parameters
        ----------
        wlen_lo : float
            Low edge to use for the internal grid. Should be below the lowest
            value expected in data. Must be < wlen_hi. Units are arbitrary.
        wlen_hi : float
            High edge to use for the internal grid. Should be above the lowest
            value expected in data. Must be > wlen_lo. Units are arbitrary.
            Will be rounded up to the nearest ``wlen_step`` boundary, if
            necessary.
        wlen_step : float
            Step size for the internal uniform grid covering the wavelength
            range. Determines the discretization of PSF calculations.
        max_spread : float
            Maximum spread of true wavelengths contributing to a single pixel,
            assumed to be symmetric so that the maximum PSF support is twice
            this value. Used to determine the sparse structure of internal
            arrays.
        dtype : numpy datatype
            Datatype to use for accumulating summary statistics.
        """
        if wlen_lo >= wlen_hi:
            raise ValueError('Expected wlen_lo < wlen_hi.')
        if wlen_step <= 0:
            raise ValueError('Expected wlen_step > 0.')
        self.grid_scale = wlen_step
        # Tabulate the internal grid centers.
        self.n_grid = int(np.ceil((wlen_hi - wlen_lo) / wlen_step)) + 1
        self.grid = wlen_lo + np.arange(self.n_grid) * self.grid_scale
        self.phi_sum = np.zeros(self.n_grid, dtype)
        self.n_spread = int(np.ceil(max_spread / wlen_step))
        self.A_sum = baad.sparse.SparseAccumulator(
            self.n_grid, self.n_spread, dtype)
        self.nadded = 0

    def reset(self):
        """Reset this coadder to its initial state.
        """
        self.phi_sum[:] = 0.
        self.A_sum.reset()
        self.nadded = 0

    @property
    def nbytes(self):
        """Number of bytes used by internal numpy arrays.
        """
        return self.phi_sum.nbytes + self.A_sum.nbytes + self.grid.nbytes

    def tabulate(self, psf_model, wlen):
        """Tabulate a PSF model on our internal grid.

        Parameters
        ----------
        psf_model : callable
            Function of (wlen, dwlen) that broadcasts over input arrays and
            returns the relative contribution of true wlen+dwlen to observed
            wlen. The absolute normalization of the returned values can be
            arbitrary.
        wlen : float or array
            Observed wavelength(s) where the PSF should be tabulated, normally
            corresponding to one or more pixel centers.

        Returns
        -------
        array
            2D array with shape (len(wlen), 2 * n_spread + 1) containing the
            (un-normalized) tabulated PSF.
        """
        wlen = np.asarray(wlen)
        dwlen = self.grid_scale * np.arange(-self.n_spread, self.n_spread + 1)
        return dwlen, psf_model(wlen.reshape(-1, 1), dwlen)

    def add(self, data, edges, ivar, psf, convolve_with_pixel=True,
            sigma_clip=3.0, auto_clip=False, retval=False):
        """Add a single observation to our accumulated summary statistics.

        An observation is defined by arbitrary pixel edges, per-pixel inverse
        variances (which can zero for missing pixels), and a per-pixel PSF.

        Parameters
        ----------
        data : array
            Array of N pixel values for this observation.
        edges : array
            Array of N+1 increasing pixels edges for this observation.
            The first and last edges must be inset enough for the maximum
            dispersion. If you only know pixel centers, you can use
            func:`baad.utils.centers_to_edges` to estimate pixel edges.
        ivar : array or float
            Array of N inverse variances for this observation's data. Must
            all be >= 0. Covariances between pixels are assumed to be zero.
            A single float value is assumed to apply to all pixels.
        psf : float or array
            The dispersion at each pixel center can be specified five different
            ways:
              - A single Gaussian RMS value in wlen units that applies
              to all pixels.
              - An array of N Gaussian RMS values in wlen units for each pixel.
              - A array of 2n+1 dispersions tabulated on a uniform grid
              centered at zero with spacing self.grid_scale that applies to all
              pixels.
              - A 2D array of shape (N, 2n+1) with per-pixel dispersions
              tabulated on the same grid.
              - A function of (wlen, dwlen) used to tabulate the PSF using
              :meth:`tabulate`.
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
        auto_clip : bool
            Automatically clip a tabulated PSF (after pixel convolution) if
            it extends beyond ``max_spread``.  Otherwise, raise a ValueError
            when this occurs.  Ignored unless input PSF is tabulated.
        retval : bool
            Returns a tuple of arrays (support, phi, A) that summarize this
            observation's contribution to the baad.

        Returns
        -------
        tuple
            When retval is True, return (support, phi, A) giving this
            observation's contributions to the coadd, tabulated on the internal
            grid.  Support is a CSR sparse array with shape (N, n_grid) with
            the normalized support of each pixel.  phi is a 1D array of
            length n_grid with this observation's contribution to phi_sum.
            A is a CSR sparse array of shape (n_grid, n_grid) with this
            observation's contribution to A_sum.
        """
        npixels, data, edges, ivar = self.check_data(data, edges, ivar)

        # Find the closest grid indices to each pixel midpoint.
        mid = 0.5 * (edges[1:] + edges[:-1])
        mid_idx = np.searchsorted(self.grid, mid - 0.5 * self.grid_scale)
        assert np.all(np.abs(
            self.grid[mid_idx] - mid) <= (0.5 + 1e-8) * self.grid_scale)
        if convolve_with_pixel:
            # Find the closest grid indices to each edge.
            edge_idx = np.searchsorted(
                self.grid, edges - 0.5 * self.grid_scale)
            assert np.all(np.abs(
                self.grid[edge_idx] - edges) <= (0.5 + 1e-8) * self.grid_scale)

        # Calculate the (un-normalized) support of each pixel.
        if callable(psf):
            _, psf = self.tabulate(psf, mid)
            assert psf.shape == (npixels, 2 * self.n_spread + 1)
            auto_clip = True
        else:
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
                if ilo[0] < 0 or ihi[-1] > self.n_grid:
                    raise ValueError('Pixels disperse outside grid.')
                for i in range(npixels):
                    img = psf if shared else psf[i]
                    pix = np.ones(edge_idx[i + 1] - edge_idx[i] + 1)
                    supports.append(np.convolve(pix, img, mode='full'))
                    assert len(supports[-1] == ihi[i] - ilo[i])
            else:
                ilo = mid_idx - extent
                ihi = mid_idx + extent + 1
                if ilo[0] < 0 or ihi[-1] > self.n_grid:
                    raise ValueError('Pixels disperse outside grid.')
                for i in range(npixels):
                    supports.append(psf if shared else psf[i])
                    assert len(supports[-1] == ihi[i] - ilo[i])

        # Check that no pixels spread beyond our sparse cutoff.
        max_support = 2 * self.n_spread + 1
        if np.any(ihi - ilo > max_support):
            if auto_clip:
                nclip = ihi - ilo - max_support
                for i in np.where(nclip > 0)[0]:
                    dlo = mid_idx[i] - extent - ilo[i]
                    dhi = ihi[i] - mid_idx[i] - extent - 1
                    assert dlo + dhi == nclip[i]
                    supports[i] = supports[i][dlo:-dhi]
                    ilo[i] += dlo
                    ihi[i] -= dhi
                assert np.all(ihi - ilo <= max_support)
            else:
                raise ValueError(
                    'Dispersed pixel spread {:.1f} '
                    'exceeds maximum spread {:.1f} '
                    'and auto_clip is False.'
                    .format(self.grid_scale * np.max(ihi - ilo),
                            self.grid_scale * self.n_spread))

        # Normalize each pixel's support in place.
        norm = (edges[1:] - edges[:-1]) / self.grid_scale
        for i in range(npixels):
            supports[i] *= norm[i] / supports[i].sum()

        if retval:
            # Initialize arrays to return.
            phi = np.zeros_like(self.phi_sum)
            A = baad.sparse.SparseAccumulator(self.n_grid, self.n_spread)
            iptr = np.empty(npixels + 1, int)
            iptr[0] = 0
            iptr[1:] = np.cumsum(ihi - ilo)
            nsparse = iptr[-1]
            sparse_data = np.concatenate(supports)
            assert sparse_data.shape == (nsparse,)
            idx = np.empty(nsparse, int)
            for i in range(npixels):
                idx[iptr[i]:iptr[i + 1]] = np.arange(ilo[i], ihi[i], dtype=int)
            support = scipy.sparse.csr_matrix(
                (sparse_data, idx, iptr), (npixels, self.n_grid))
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
                A.add(dA, ilo[i])
            self.A_sum.add(dA, ilo[i])

        self.nadded += 1

        if retval:
            return support, phi, A

    def check_data(self, data, edges, ivar):
        """Perform checks for valid input data.
        """
        npixels = len(data)
        data = np.asarray(data)
        edges = np.asarray(edges)
        ivar = np.atleast_1d(ivar)
        if len(edges) != npixels + 1:
            raise ValueError('Length of edges and data arrays do not match.')
        if len(ivar) == 1:
            ivar = np.full(npixels, ivar)
        elif len(ivar) != npixels:
            raise ValueError('Length of ivar and data arrays do not match.')
        if np.any(ivar < 0):
            raise ValueError('All ivar values must >= 0.')
        if not np.all(np.diff(edges) > 0):
            raise ValueError('Pixel edges are not in increasing order.')
        return npixels, data, edges, ivar

    def get_phi(self):
        """Return the phi vector summary statistic.

        The arrays returned by :meth:`get_phi` and :meth:`get_A` capture all of
        the information contained in the multi-exposure likelihood function.

        Returns
        -------
        array
            1D array of length ``n_grid`` tabulating the phi statistic at the
            wavelengths ``grid``.
        """
        return self.phi_sum

    def get_A(self, sigma_f=0, sparse=True):
        """Return the symmetric A matrix summary statistic conditioned with a prior.

        The arrays returned by :meth:`get_phi` and :meth:`get_A` capture all of
        the information contained in the multi-exposure likelihood function.

        Note that the returned matrix is almost certainly invertible when
        sigma_f > 0, except in the unlikely case that sigma_f equals one of the
        eigenvalues of A with sigma_f = 0.

        Returns a copy if sigma_f is nonzero or sparse is False.

        When ``sparse`` is True, returns a :class:`CSR sparse matrix
        <scipy.sparse.csr_matrix>`. However, since the matrix is symmetric, it
        can be efficiently converted to :class:`CSC format
        <scipy.sparse.csc_matrix>` by taking its transpose.

        Parameters
        ----------
        sigma_f : float or array
            Value(s) of the hyperparameter giving the RMS of the Gaussian flux
            prior on the high-resolution internal grid.
        sparse : bool
            Return a CSR sparse matrix when True, otherwise return a dense
            2D numpy array.

        Returns
        -------
        array or sparse matrix
            Symmetric N x N matrix as a dense array or a :class:`CSR sparse
            matrix <scipy.sparse.csr_matrix>`.
        """
        if sigma_f < 0:
            raise ValueError('Expected sigma_f >= 0.')
        if sigma_f == 0 and sparse:
            return self.A_sum.csr
        csr = self.A_sum.csr.copy()
        if sigma_f > 0:
            csr += sigma_f ** -2 * scipy.sparse.identity(self.n_grid)
        if sparse:
            return csr
        else:
            return csr.toarray()

    def get_f(self, sigma_f):
        """Return the deconvolved baad.

        Results will generally be sensitive to the choice of prior
        hyperparameter sigma_f since the deconvolution attempts to
        reconstruct high-frequency information that has been erased from
        the data by PSF convolution. Use :meth:`get_A` (with the same
        ``sigma_f`` value) to obtain the corresponding inverse covariance
        of the estimated true flux.

        You should normally use one of the ``extract_...`` methods
        to minimize sensitivity to the choice of hyperparameter.

        Parameters
        ----------
        sigma_f : float or array
            Value(s) of the hyperparameter giving the RMS of the Gaussian flux
            prior on the high-resolution internal grid.

        Returns
        -------
        array
            1D array of N estimated true flux values.
        """
        phi = self.get_phi()
        A = self.get_A(sigma_f, sparse=True)
        # Convert CSR to CSC format before taking the inverse, for efficiency.
        Ainv = scipy.sparse.linalg.inv(A.T)
        return Ainv.dot(phi)

    def get_log_evidence(self, sigma_f):
        """Calculate the log of the evidence P(D|sigma_f).

        Note that this is relatively slow since it requires expanding the
        sparse A into a dense matrix in order to calculate its (log)
        determinant.

        TODO:
        * Does this method need to be faster?
        * How much does the logdet term contribute? Would a fast
          approximation to |A0+eps| help here?
        * How much does the phi.Ainv.phi term contribute? Would a fast
          approximation to phi.(A0+eps)**-1.phi help here?
        * Benchmark dense vs sparse methods, e.g., using suggestions in
          https://stackoverflow.com/questions/19107617/
          how-to-compute-scipy-sparse-matrix-determinant-without-turning-it-to-dense

        Parameters
        ----------
        sigma_f : float or array
            Value(s) of the hyperparameter giving the RMS of the Gaussian flux
            prior on the high-resolution internal grid.

        Returns
        -------
        float or array
            Value(s) of the log evidence corresponding to each input sigma_f.
        """
        phi = self.get_phi()
        A0 = self.get_A(sigma_f=0, sparse=False)
        scalar = np.asarray(sigma_f).shape == ()
        sigma_f = np.atleast_1d(sigma_f).astype(float)
        if np.any(sigma_f <= 0):
            raise ValueError('Expected all sigma_f > 0.')
        logE = np.empty(sigma_f.shape, float)
        eye = np.identity(self.n_grid)
        for i, ivar in enumerate(sigma_f ** -2):
            A = A0 + ivar * eye
            s, logdetA = np.linalg.slogdet(A)
            assert s > 0, f'Invalid sign|A|={s} for sigma_f={sigma_f}'
            Ainv = np.linalg.inv(A)
            logE[i] = 0.5 * (
                self.n_grid * np.log(ivar) - logdetA + phi.dot(Ainv.dot(phi)))
        return logE[0] if scalar else logE

    def get_most_probable(self, sigma_f_min, sigma_f_max, rtol=1e-4):
        """Calculate most probable value of the hyperparameter sigma_f.

        Uses root finding on the logarithmic derivative of the log evidence.
        The input min/max limits must bracket the root.

        Parameters
        ----------
        sigma_f_min : float
            Minimum value to consider.
        sigma_f_max : float
            Maximum value to consider.
        rtol : float
            Desired fractional accuracy.

        Returns
        -------
        float
            Value of sigma_f where :meth:`get_log_evidence` returns its maximum
            value on the specified interval.
        """
        phi = self.get_phi()
        A0 = self.get_A(sigma_f=0, sparse=False)
        eye = np.identity(self.n_grid)

        # Define a function of log(sigma_f) to pass to brentq.
        def f(log_sigma_f):
            ivar = np.exp(-2 * log_sigma_f)
            A = A0 + ivar * eye
            Ainv = np.linalg.inv(A)
            return -self.n_grid + ivar * (
                np.trace(Ainv) + phi.dot(Ainv.dot(Ainv.dot(phi))))

        return np.exp(scipy.optimize.brentq(
            f, np.log(sigma_f_min), np.log(sigma_f_max), xtol=rtol))

    def extract_downsampled(self, H, sigma_f, return_cov):
        """Extract downsampled values with a Bayesian prior.

        The coefficient matrix H specifies P arbitrary linear combinations
        of the high-resolution true flux to extract, using a prior specified
        by the hyperparameter ``sigma_f`` and marginalizing over the remaining
        N-P degrees of freedom.

        A suitably chosen set of coefficients can filter out high-frequency
        components of the true flux that have not been measured due to
        PSF convolution, yielding downsampled results that are insensitive
        to the choice of prior hyperparameter ``sigma_f``.

        The methods :meth:`extract_pixels` and :meth:`extract_gaussian` are
        wrappers around this method to handle some common cases. As an
        alternative, :meth:`extract_flatnoise` uses the data to calculate
        an "optimal" PSF that decorrelates and flattens the noise.

        Parameters
        ----------
        H : scipy.sparse.csr_matrix
            CSR sparse matrix of shape (P, N) that transforms true fluxes to
            downsampled output quantities.
        sigma_f : float
            Value of the hyperparameter to use for the extraction.
        return_cov : bool
            Return both the mean and covariance of the downsampled variables
            when True.  Otherwise, return only the mean, which will usually
            be significantly faster.

        Returns
        -------
        array or tuple
            When ``return_cov`` is True, return a tuple (mu, cov) that
            specifies the multivariate Gaussian posterior probability density,
            with mu a 1D array of length P and cov a 2D postive-definite P x P
            array of covariances.  When ``return_cov`` is False, return only
            the array mu.
        """
        P, N = H.shape
        if N != self.n_grid:
            raise ValueError('Wrong number of columns in coefficient matrix H.')
        if P > N:
            raise ValueError('Specified coefficient matrix H requires an upsample.')
        # Extract the summary statistics.
        phi = self.get_phi()
        A = self.get_A(sigma_f, sparse=True)
        # Embed H into an invertible matrix K.
        Kinv, Kt, k = baad.sparse.get_embedded(H)
        # Combine A and Kinv.
        AKinv = A.dot(Kinv).tocsc()
        # Find mu as the solution to AKinv.mu = phi
        mu = scipy.sparse.linalg.spsolve(AKinv, phi)
        # Marginalize out nuisance parameters.
        mu = mu[k]
        if not return_cov:
            return mu
        # Find covariance C as the solution to AKinv.C = Kt
        C = scipy.sparse.linalg.spsolve(AKinv, Kt)
        # Marginalize out nuisance paramters.
        C = C[np.ix_(k, k)]
        # Average any round-off errors to force C to be exactly symmetric.
        C = 0.5 * (C + C.T)
        return mu, C

    def extract_pixels(self, size, sigma_f, return_cov=False):
        """Extract downsampled pixels with a Bayesian prior.

        Uses :meth:`extract_downsampled` using boxcar weights.

        The returned covariance will generally not be diagonal.

        If ``size`` does not divide the grid evenly, the extraction will be
        trimmed on the right-hand edge.

        Parameters
        ----------
        size : int
            Size of output pixels in high-resolution ``grid_scale`` units.
            Must be > 0 and <= n_grid.
        sigma_f : float
            Value of the hyperparameter to use for the extraction.
        return_cov : bool
            Calculate and return the covariance.

        Returns
        -------
        tuple
            Tuple (edges, mu, cov) where edges is a 1D array of n+1 increasing
            output pixel edges, mu is a 1D array of n output mean values, and
            cov is a 2D n x n symmetric array of output covariances. If
            return_cov is False, then only return (edges, mu).
        """
        size = int(size)
        if size <= 0 or size > self.n_grid:
            raise ValueError('Expected size > 0 and <= n_grid.')
        n_extracted = int(np.floor(self.n_grid / size))
        edges = (
            self.grid[0] + size * self.grid_scale * np.arange(n_extracted + 1))
        # Create a CSR sparse matrix H of pixel downsampling coefficients.
        nnz = n_extracted * size
        data = np.ones(nnz)
        indices = np.arange(nnz, dtype=int)
        indptr = size * np.arange(n_extracted + 1, dtype=int)
        H = scipy.sparse.csr_matrix((data, indices, indptr), shape=(n_extracted, n_grid))
        return (edges,) + self.extract_downsampled(coefs, sigma_f, return_cov)

    def extract_gaussian(self, spacing, rms, sigma_f):
        """Extract downsampled fluxes with a Gaussian PSF.

        Estimate the posterior PDF of the true flux convolved with a
        Gaussian PSF, sampled on an equally spaced grid. With ``rms`` chosen
        large enough to erase high-frequency information that is not present
        in the data (due to PSF convolution), the result should be insensitive
        to the choice of hyperparameter ``sigma_f``.

        The returned covariance will generally not be diagonal.

        Uses :meth:`extract_downsampled` using boxcar weights.

        If ``spacing`` does not divide the grid evenly, the extraction will be
        trimmed on the right-hand edge.

        Parameters
        ----------
        spacing : float
            Spacing between extracted centers in wavelength units.
        rms : float
            RMS of Gaussian PSF to convolve with the estimated true flux.
        sigma_f : float
            Value of the hyperparameter to use for the extraction.

        Returns
        -------
        tuple
            Tuple (centers, mu, cov) where centers is a 1D array of n
            increasing output centers, mu is a 1D array of n output mean
            values, and cov is a 2D n x n symmetric array of output
            covariances.
        """
        # Determine locations of the extracted centers.
        n_extracted = int(np.floor((self.grid[-1] - self.grid[0]) / spacing))
        centers = self.grid[0] + (np.arange(n_extracted) + 0.5) * spacing
        # Build PSFs at each extracted center.
        dx = self.grid - centers[:, np.newaxis]
        coefs = spacing * np.exp(
            -0.5 * (dx / rms) ** 2) / (np.sqrt(2 * np.pi) * rms)
        mu, cov = self.extract_downsampled(coefs, sigma_f)
        return centers, mu, cov

    def extract_whitened(self, sigma_f):
        """Estimate true flux convolved to have flat uncorrelated noise.

        Calculate an effective PSF that, when convolved with the estimated
        true flux, results in a covariance equal to the identiy matrix.

        The effective PSF depends on the data accumulated so far and will
        change as more data is added.

        Parameters
        ----------
        sigma_f : float
            Value of the hyperparameter to use for the extraction.

        Returns
        -------
        tuple
            Tuple (psfs, mu) where psfs is an NxN 2D array of PSFs
            tabulated at each grid point and mu is a 1D array of N output
            mean values, whose covariance is the NxN identity matrix.
        """
        phi = self.get_phi()
        A = self.get_A(sigma_f, sparse=True)
        # Calculate H as the matrix square-root of Ainv.
        # TODO: implement using sparse linear algebra.
        # TODO: remove assertion tests.
        s, V = np.linalg.eigh(A)
        assert np.all(s > 0)
        S = np.diag(s)
        assert np.allclose(V.dot(S.dot(V.T)), A)
        assert np.allclose(V.T, np.linalg.inv(V))
        H = V.dot(np.diag(s ** +0.5).dot(V.T))
        Hinv = V.dot(np.diag(s ** -0.5).dot(V.T))
        assert np.allclose(H.dot(Hinv), np.identity(self.n_grid))
        assert np.allclose(Hinv.T, Hinv)
        AHinv = A.dot(Hinv)
        assert np.allclose(Hinv.T.dot(AHinv), np.identity(self.n_grid))
        # Perform the change of parameters.
        mu = np.linalg.inv(AHinv).dot(phi)
        return H, mu
