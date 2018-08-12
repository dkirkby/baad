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
