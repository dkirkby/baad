# Coaddition of Binned Data

Combine individual observations of 1D or 2D data that satisfy the following conditions:
 - Each observation views the same static source.
 - Each observation is convolved with a known (possibly varying) point-spread function.
 - Observed data is accumulated into bins (pixels) with known edges.
 - Statistical fluctuations in the binned data are uncorrelated between bins.

Coaddition uses the sufficient summary statistics proposed in Kaiser 2004 "Addition of
Images with Varying Seeing" (unpublished technical note) to efficiently accumulate
the joint likelihood of all observations. Observations are then combined in Bayesian
framework with a Gaussian prior having a single hyperparameter.  Methods are provided
to calculate the data evidence as a function of this hyperparameter, and to support
model optimization or averaging.

An extracted coadd takes the form of a multivariate Gaussian posterior probability
density (specifed with a mean vector and covariance matrix) in parameters that are
arbitrary linear combinations of the true flux tabulated on a high-resolution grid.
Convenience methods are provided for:
 - Estimated true flux downsampled to square pixels.
 - Estimated true flux convolved with a Gaussian PSF of fixed size.
 - Estimated true flux convolved with an effective PSF that whitens the noise.