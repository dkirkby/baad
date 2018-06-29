import pytest
import numpy as np

from coadd.spectra import CoAdder


def test_ctor():
    """Normal construction.
    """
    c = CoAdder(100., 200., 1., 50.)
    assert c.grid[0] == 100.
    assert c.grid[-1] == 200.
    assert c.grid_scale == 1.
    assert np.all(c.phi_sum == 0)
    assert np.all(c.A_sum.csr.toarray() == 0)


def test_add_psf():
    """Addition with different types of PSF inputs.
    """
    c = CoAdder(100., 200., 0.5, 50.)
    psf = np.zeros(21)
    psf[10] = 1
    psfs = np.tile(psf, [3, 1])
    data = [1, 3, 2], [150, 160, 170, 180], [0.1, 0.2, 0.1]
    for convolve in True, False:
        c.add(*data, 5, convolve)
        c.add(*data, [4, 5, 6], convolve)
        c.add(*data, psf, convolve)
        c.add(*data, psfs, convolve)


def test_reset():
    """Reset after adding.
    """
    c = CoAdder(100., 200., 0.5, 50.)
    psf = np.zeros(21)
    psf[10] = 1
    data = [1, 3, 2], [150, 160, 170, 180], [0.1, 0.2, 0.1]
    c.add(*data, 5)
    assert not np.all(c.phi_sum == 0)
    assert not np.all(c.A_sum.csr.toarray() == 0)
    c.reset()
    assert np.all(c.phi_sum == 0)
    assert np.all(c.A_sum.csr.toarray() == 0)


def test_add_analytic_vs_tabulated():
    """Compare analytic vs tabulated Gaussian PSFs.
    """
    c = CoAdder(100., 200., 0.5, 50.)
    data = [1, 3, 2], [150, 160, 170, 180], [0.1, 0.2, 0.1]
    psf_grid = c.grid_scale * np.arange(-10, +11)
    for convolve in True, False:
        # Common PSF for all pixels.
        rms = 1.5
        gp0, _, _ = c.add(*data, rms, convolve, retval=True)
        psf = np.exp(-0.5 * (psf_grid / rms) ** 2)
        gp1, _, _ = c.add(*data, psf, convolve, retval=True)
        assert np.allclose(gp0.toarray(), gp1.toarray(), atol=0.05, rtol=0.05)
        # Individual PSFs for each pixel.
        rms = np.array([1.4, 1.5, 1.6])
        gp0, _, _ = c.add(*data, rms, convolve, retval=True)
        psf = np.exp(-0.5 * (psf_grid / rms.reshape(-1, 1)) ** 2)
        gp1, _, _ = c.add(*data, psf, convolve, retval=True)
        assert np.allclose(gp0.toarray(), gp1.toarray(), atol=0.05, rtol=0.05)


def test_get_phi():
    """Test calculation of phi vector summary statistic.
    """
    c = CoAdder(100., 200., 0.5, 50.)
    assert np.all(c.get_phi() == 0)
    data = [1, 3, 2], [150, 160, 170, 180], [0.1, 0.2, 0.1]
    c.add(*data, 5)
    assert np.allclose(np.mean(c.get_phi()), 0.0895522)
    assert np.allclose(np.std(c.get_phi()), 0.1422825)


def test_get_A():
    """Test calculation of A matrix summary statistic.
    """
    c = CoAdder(100., 200., 0.5, 50.)
    assert np.all(c.get_A() == 0)
    assert np.array_equal(
        c.get_A(sparse=True).toarray(), c.get_A(sparse=False))
    assert np.array_equal(c.get_A(sigma_f=2), 0.25 * np.identity(c.n_grid))
    data = [1, 3, 2], [150, 160, 170, 180], [0.1, 0.2, 0.1]
    c.add(*data, 5)
    A = c.get_A()
    assert np.all(A.T == A)
    assert np.allclose(np.trace(A), 3.89099485)
    assert np.linalg.slogdet(A)[1] == -np.inf
    A = c.get_A(sigma_f=1.1)
    assert np.all(A.T == A)
    assert np.allclose(np.trace(A), 170.006697)
    assert np.allclose(np.linalg.slogdet(A)[1], -35.734250)
    A = c.get_A(sigma_f=0.5, sparse=True)
    assert A.getformat() == 'csr'
    assert A.T.getformat() == 'csc'


def test_get_f():
    """Test calculation of deconvolved true flux f.
    """
    c = CoAdder(100., 200., 0.5, 50.)
    assert np.all(c.get_f(sigma_f=1) == 0)
    data = [1, 3, 2], [150, 160, 170, 180], [0.1, 0.2, 0.1]
    c.add(*data, 5)
    assert np.allclose(np.sum(c.get_f(sigma_f=1)), 4.950472)


def test_get_log_evidence():
    """Test calculation of log evidence.
    """
    c = CoAdder(100., 200., 0.5, 50.)
    assert c.get_log_evidence(sigma_f=1) == 0
    data = [1, 3, 2], [150, 160, 170, 180], [0.1, 0.2, 0.1]
    c.add(*data, 5)
    assert np.allclose(c.get_log_evidence(sigma_f=1), -0.32704586)
    assert np.allclose(c.get_log_evidence(sigma_f=[1]), [-0.32704586])
    assert np.allclose(
        c.get_log_evidence(sigma_f=[0.1, 1, 10]),
        [0.00843854, -0.32704586, -5.79924322])
    with pytest.raises(ValueError):
        c.get_log_evidence(sigma_f=0)
    with pytest.raises(ValueError):
        c.get_log_evidence(sigma_f=[1, -1])


def test_extract_downsampled():
    """Test extraction of downsampled coadd.
    """
    c = CoAdder(100., 200., 0.5, 50.)
    n = 10
    sigma_f = 1.5
    coefs = np.identity(c.n_grid)[:n]
    mu, cov = c.extract_downsampled(coefs, sigma_f)
    assert np.array_equal(mu, np.zeros(n))
    assert np.array_equal(cov, sigma_f ** 2 * np.identity(n))


def test_extract_pixels():
    """Test extraction to downsampled pixels.
    """
    c = CoAdder(100., 200., 0.5, 50.)
    size = 8
    sigma_f = 1.5
    edges, mu, cov = c.extract_pixels(size, sigma_f)
    n = len(edges) - 1
    assert mu.shape == (n,)
    assert cov.shape == (n, n)
    assert np.array_equal(cov.T, cov)
    assert edges[0] == c.grid[0]
    assert edges[-1] == c.grid[n * size]
    assert np.array_equal(mu, np.zeros(n))
    assert np.allclose(cov, sigma_f ** 2 * size * np.identity(n))
    data = [1, 3, 2], [150, 160, 170, 180], [0.1, 0.2, 0.1]
    c.add(*data, 5)
    edges, mu, cov = c.extract_pixels(size, sigma_f)
    assert np.array_equal(cov.T, cov)
    assert np.allclose(np.sum(mu), 5.7583230)
    assert np.allclose(np.sum(np.diagonal(cov)), 414.52807)


def test_extract_gaussian():
    c = CoAdder(100., 200., 0.5, 50.)
    size = 8
    spacing = size * c.grid_scale
    rms = spacing
    sigma_f = 1.5
    centers, mu, cov = c.extract_gaussian(spacing, rms, sigma_f)
    n = len(centers)
    assert mu.shape == (n,)
    assert cov.shape == (n, n)
    assert np.array_equal(cov.T, cov)
    assert centers[0] == c.grid[0] + 0.5 * spacing
    assert centers[-1] == centers[0] + (n - 1) * spacing
    assert np.array_equal(mu, np.zeros(n))
    data = [1, 3, 2], [150, 160, 170, 180], [0.1, 0.2, 0.1]
    c.add(*data, 5)
    centers, mu, cov = c.extract_gaussian(spacing, rms, sigma_f)
    assert np.array_equal(cov.T, cov)
    assert np.allclose(np.sum(mu), 5.7581933)
    assert np.allclose(np.sum(np.diagonal(cov)), 96.95508)


def test_extract_whitened():
    c = CoAdder(100., 200., 0.5, 50.)
    sigma_f = 1.5
    psfs, mu = c.extract_whitened(sigma_f)
    n = len(mu)
    assert psfs.shape == (n, n)
    assert np.array_equal(mu, np.zeros(n))
    data = [1, 3, 2], [150, 160, 170, 180], [0.1, 0.2, 0.1]
    c.add(*data, 5)
    psfs, mu = c.extract_whitened(sigma_f)
    assert np.allclose(np.sum(mu), 10.262560)
