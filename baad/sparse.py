"""Sparse algebra utilities.
"""
import numpy as np

import scipy.sparse


class SparseAccumulator(object):
    """Sparse matrix representation with fixed sparsity structure.
    """
    def __init__(self, N, M, dtype=np.float):
        """Initialize a sparse NxN matrix with M nonzero diagonals.

        >>> SA = SparseAccumulator(6, 2, int)
        >>> SA.csr.data[:] = 1
        >>> print(SA.csr.toarray())
        [[1 1 1 0 0 0]
         [1 1 1 1 0 0]
         [1 1 1 1 1 0]
         [0 1 1 1 1 1]
         [0 0 1 1 1 1]
         [0 0 0 1 1 1]]

        Parameters
        ----------
        N : int
            Dimensions of square matrix.
        M : int
            Number of non-zero diagonals above and below the main diagonal.
        """
        nsparse = N ** 2 - (N - M - 1) * (N - M)
        if nsparse <= 0:
            raise ValueError('Invalid sparse shape parameters.')
        data = np.zeros(nsparse, dtype)
        diag = np.arange(N, dtype=np.int)
        lo = np.maximum(0, diag - M)
        hi = np.minimum(N, diag + M + 1)
        indptr = np.empty(N + 1, np.int)
        indptr[0] = 0
        indptr[1:] = np.cumsum(hi - lo)
        assert indptr[-1] == nsparse
        assert indptr[1] == M + 1
        indices = np.empty(nsparse, np.int)
        for i in range(N):
            indices[indptr[i]:indptr[i + 1]] = np.arange(
                lo[i], hi[i], dtype=int)
        self.csr = scipy.sparse.csr_matrix((data, indices, indptr), (N, N))

    def reset(self):
        """Reset any accumulated non-sparse values to zero.
        """
        self.csr.data[:] = 0

    def add(self, B, offset):
        """Add the submatrix B at the specified offset.

        >>> SA = SparseAccumulator(6, 2, int)
        >>> SA.add(np.ones((2, 3), int), 1)
        >>> print(SA.csr.toarray())
        [[0 0 0 0 0 0]
         [0 1 1 1 0 0]
         [0 1 1 1 0 0]
         [0 0 0 0 0 0]
         [0 0 0 0 0 0]
         [0 0 0 0 0 0]]
        """
        nrow, ncol = B.shape
        idx = self.csr.indptr[offset:offset + nrow]
        col1 = idx - self.csr.indices[idx] + offset
        col2 = col1 + ncol
        for i, b in enumerate(B):
            self.csr.data[col1[i]:col2[i]] += b

    @property
    def nbytes(self):
        """Number of bytes used by internal numpy arrays.
        """
        return (self.csr.data.nbytes +
                self.csr.indices.nbytes +
                self.csr.indptr.nbytes)


def get_embedded(H, k=None):
    """Embed a rectangular downsampling matrix into an invertible matrix.

    The invertible matrix K is an N x N identity matrix with row k[p]
    replaced with row p of H.  By default, use k[p] = argmax(H[p,:]) but
    different row assignments can be specified.

    >>> H = scipy.sparse.csr_matrix(np.array([
    ...     [1., 1., 0., 0., 0.],
    ...     [0., 0., 1., 1., 0.]]))
    >>> Kinv, Kt, k = get_embedded(H)
    >>> Kinv.toarray()
    array([[ 1., -1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  1., -1.,  0.],
           [ 0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  1.]])
    >>> k
    array([0, 2])
    >>> K = np.linalg.inv(Kinv.toarray())
    >>> assert np.array_equal(K[k], H.toarray())
    >>> assert np.array_equal(Kt.toarray(), K.T)

    The calculation uses the block matrix inverse formula:

      Kinv = [[H1inv, -H1inv.H2], [0, 1]]

    for K = [[H1, H2], [0, 1]] without actually permuting the rows and
    columns of K and performing all matrix operations with sparse
    methods.

    This algorithm is optimized for the case where H1inv is sparse
    since it uses sparse matrix inversion of H1.

    Parameters
    ----------
    H : scipy.sparse.csr_matrix
        CSR sparse matrix of shape (P, N) that transforms true fluxes to
        downsampled output quantities.
    k : array or None
        1D array of length P with row assignments to use, or None to have these
        calculated automatically as k[p] = argmax(H[p,:]).

    Returns
    -------
    tuple
        Tuple (Kinv, Kt, k) where Kinv is the inverse of K in LIL sparse format,
        Kt is its transpose in CSC format and k is a 1D integer array of length
        P giving the values k[p].

    Raises
    ------
    RuntimeError
        H1 is not invertible.
    """
    P, N = H.shape
    if k is None:
        k = np.empty(P, dtype=int)
        for p in range(P):
            sl = slice(H.indptr[p], H.indptr[p + 1])
            # Determine the index k[p] where this row p of H will appear in K.
            k[p] = H.indices[sl][np.argmax(H.data[sl])]
    else:
        k = np.asarray(k)
        if k.shape != (P,):
            raise ValueError('Invalid row assignment array.')
    # Create the sparse H1 in CSC format.
    H1 = H[:, k].tocsc()
    # Calculate its inverse, also in CSC format.
    H1inv = scipy.sparse.linalg.inv(H1)
    mask = np.ones(N, dtype=bool)
    mask[k] = False
    # Create the sparse H2 in CSR format.
    H2 = H[:, mask]
    H1invH2 = H1inv.dot(H2)
    # Initialize Kinv as the NxN identity matrix in LIL format.
    Kinv = scipy.sparse.identity(N, format='lil')
    # Embed H1inv into Kinv[k, k].
    Kinv[np.ix_(k, k)] = H1inv
    # Embed -H1inv.H2 into the remaining columns of Kinv[k].
    Kinv[np.ix_(k, mask)] = -H1invH2
    # Initialize K as the NxN identity matrix in CSR format.
    K = scipy.sparse.identity(N, format='csr')
    # Embed H into Kt. This generates a warning that changing CSR
    # sparsity structure is expensive, which we ignore since it
    # is faster than using the recommended LIL format.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=scipy.sparse.SparseEfficiencyWarning)
        K[k] = H

    return Kinv, K.T, k
