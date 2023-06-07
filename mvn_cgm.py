"""
Method for sampling from a multivariate Gaussian using the conjugate gradient method.
Included here to get a back-of-the-envelope sense of how much slower this method will be over classical Cholesky decomposition based sampling.
"""

from scipy import linalg
import numpy as np
from numpy import random
import math
import logging

# TODO could wrap these in a class so that we can precompute and retain the Cholesky factor of the LD portion.
# We would have to change the signatures of the functions somewhat to take in the LD / Psi separately though instead of combined as an unscaled precision matrix

def sample_mvn_alg_3_4(unscaled_Q, beta_hat, sigma2, N, max_iterations=None, error=None, preconditioned=False):
    # we will see a lot of squeezes because the base PRScs implementation typically works with 2Darrays with dimension 1 instead of 1Darrays
    max_iterations = len(unscaled_Q) if max_iterations is None else max_iterations
    sigma2 = np.squeeze(sigma2) # sigma2 sometimes comes in a 2darray
    beta_hat = np.squeeze(beta_hat)
    scaled_Q = (N / sigma2)*unscaled_Q
    Q_sample = math.sqrt(N / sigma2)*cholesky_sample(0, unscaled_Q) # or however else you want to sample it
    Q_sample = np.squeeze(Q_sample)
    eta = Q_sample + ((N / sigma2)*beta_hat)
    if not preconditioned:
        theta, n_iterations = cg_solve(scaled_Q, eta, max_iterations=max_iterations, error=error)
        # logging.info("Nonprecon MVN call on blocksize %d has condition number %f, iterations %d" % (len(unscaled_Q), condition_number(unscaled_Q), n_iterations))
        # where n_iterations is the number of iterations actually performed
        return theta, n_iterations
    preconditioner = linalg.inv(np.diag(np.diagonal(unscaled_Q).copy()))
    theta, n_iterations = preconditioned_cg_solve(scaled_Q, eta, preconditioner, max_iterations=max_iterations, error=error)
    # logging.info("Precond MVN call on blocksize %d has condition number %f, iterations %d" % (len(unscaled_Q), condition_number(preconditioner @ unscaled_Q), n_iterations))
    return theta, n_iterations

def cholesky_sample(mu, A):
    # Cholesky sample from the MVN with mean mu and covariance matrix A
    C = linalg.cholesky(A, lower=True) # I prefer the lower triangular convention
    z = random.randn(len(A), 1)
    w = linalg.solve_triangular(C, z, lower=True, trans='T')
    return mu + w

def cg_solve(A, b, max_iterations, error=None, x=None):
    """Approximates a solution x to A@x = b. Algorithm adapted from algorithm B2 in J. Shewchuk's "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain" """
    # the number of iterations one can perform tops off at the matrix dimension, which results in the exact answer. Alg is not well defined if we go over
    n = A.shape[0]
    if max_iterations > n or error is None:
        max_iterations = n
        error = 0
    if x is None:
        x = np.zeros(n)
    # print("A at x", (A @ x).shape)
    r = b - (A @ x)
    d = r
    delta_new = np.inner(r, r)

    max_error = delta_new * error * error
    n_iterations = 0
    
    # print('CGM initialization')
    # print('max_iterations is', max_iterations)
    # print('n is', n)
    # print('x is', x.shape)
    # print('r is', r.shape)
    # print('d is', d.shape)
    # print('delta is', delta_new)
    for i in range(max_iterations):
        if delta_new <= max_error:
            break
        # print(f' === iteration {i} ===')
        q = A @ d
        # print('q is', q)
        alpha = delta_new / np.inner(d, q)
        # print('alpha is', alpha)
        x = x + (alpha * d)
        # print('x is', x)
        if i % 50 == 0:
            r = b - (A @ x)
        else:
            r = r - (alpha * q)
        # print('r is', r)
        delta_old = delta_new
        delta_new = np.inner(r, r)
        # print('new delta is', delta_new)
        beta = delta_new / delta_old
        # print('beta is', beta)
        d = r + (beta * d)
        # print('d is', d)
        n_iterations += 1
    # TODO see if we can explicitly calculate the condition number or use Gershgorin circle theorem to bound the eigenvalues, which could more rigorously argue that the blowup phenomenon really does have to do with Psi_inv
    # when we refactor to take in LD/Psi separately, that would also let us explicitly look at Psi
    # print('trace of A is', np.trace(A), 'and the  number of iterations is', n_iterations)
    return x, n_iterations

def preconditioned_cg_solve(A, b, M_inv, max_iterations, error=None, x=None):
    """Approximates a solution x to M^-1Ax = M^-1bx with starting value `x` and `max_iterations` of the Conjugate Gradient method. Adapted from algorithm B2 in J. Shewchuk's "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"""
    n = A.shape[0]
    if max_iterations > n or error is None:
        max_iterations = n
        error = 0
    if x is None:
        x = np.zeros(n)
    r = b - (A @ x)
    d = M_inv @ r
    delta_new = np.inner(r, d)

    max_error = delta_new * error * error
    n_iterations = 0
    
    # print('initialization')
    # print('r is', r)
    # print('r is', r)
    # print('delta is', delta_new)
    for i in range(max_iterations):
        if delta_new <= max_error:
            break
        # print(f' === iteration {i} ===')
        q = A @ d
        # print('q is', q)
        alpha = delta_new / np.inner(d, q)
        # print('alpha is', alpha)
        x = x + (alpha * d)
        # print('x is', x)
        if i % 50 == 0:
            r = b - (A @ x)
        else:
            r = r - (alpha * q)
        # print('r is', r)
        s = M_inv @ r
        # print('s is', s)
        delta_old = delta_new
        delta_new = np.inner(r, s)
        # print('new delta is', delta_new)
        beta = delta_new / delta_old
        # print('beta is', beta)
        d = s + (beta * d)
        # print('d is', d)
        n_iterations += 1
    return x, n_iterations

def condition_number(A):
    """Calculates the conjugate gradient method condition number of a real symmetric matrix A."""
    eigvals = linalg.eigvalsh(A)
    tmp_eigvals = np.sort(np.absolute(eigvals))
    return tmp_eigvals[-1] / tmp_eigvals[0]
