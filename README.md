# Bayesian Addition of Astronomical Data

Combine individual observations of 1D or 2D binned data that satisfy the following conditions:
 - Each observation views the same static source.
 - Each observation is convolved with a known (possibly varying) point-spread function.
 - Observed data is accumulated into bins (pixels) with known edges.
 - Statistical uncertainties in bin values are well described by a known Gaussian
   inverse covariance matrix (not necessarily diagonal).
 - Missing or bad data is flagged with zero inverse variance.

The joint likelihood of all observations is accumulated using the sufficient summary
statistics proposed in Kaiser 2004 "Addition of Images with Varying Seeing" (unpublished
technical note). Observations are then combined in a Bayesian framework with a Gaussian
prior having a single hyperparameter. (An equivalent view of this procedure is that the
hyperparameter regularizes the extraction of high-frequency information that has been
erased by PSF convolution.) Methods are provided to calculate the data evidence
as a function of this hyperparameter, and to support model optimization or averaging.

An extracted "coadd" takes the form of a multivariate Gaussian posterior probability
density (specifed with a mean vector and covariance matrix) in parameters that are
arbitrary linear combinations of the true flux tabulated on a high-resolution grid.
Convenience methods are provided for:
 - Estimated true flux downsampled to square pixels.
 - Estimated true flux convolved with a Gaussian PSF of fixed size.
 - Estimated true flux convolved with an effective PSF that whitens the noise.

Extracted coadds are numerically stable and well defined within a Bayesian framework
for any sequence of observations.  With a suitable choice of extracted pixel or
Gaussian PSF width, the results are also insensitive to the choice of hyperparameter.
