#!/usr/bin/env python

"""
Markov Chain Monte Carlo (MCMC) sampler for polygenic prediction with continuous shrinkage (CS) priors.

"""


import scipy as sp
from scipy import linalg 
from numpy import random
import gigrnd
import mvn_cgm
import logging
import time
import csv
import os


def sample_mvn(unscaled_Q, beta_hat, sigma, n, d, use_cgm, error_tolerance):
    if use_cgm == 'False':
        dinvt_chol = linalg.cholesky(unscaled_Q)
        beta_tmp = linalg.solve_triangular(dinvt_chol, beta_hat, trans='T') + sp.sqrt(sigma/n)*random.randn(d,1)
        new_beta = linalg.solve_triangular(dinvt_chol, beta_tmp, trans='N')
        return new_beta, None
    preconditioned = True if use_cgm == 'Precond' else False
    beta_hat = beta_hat
    sample, n_iterations = mvn_cgm.sample_mvn_alg_3_4(unscaled_Q, beta_hat, sigma, n, max_iterations=None, error=error_tolerance, preconditioned=preconditioned)
    return sample.reshape(d, 1), n_iterations

def mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size, n_iter, n_burnin, thin, chrom, out_dir, beta_std, seed, use_cgm, error_tolerance, mvn_output_file=None):
    # where `mvn_output_file` is a parameter I have added for saving MVN-performance-pertinent information
    logging.info('... MCMC ...')
    if use_cgm == 'False':
        logging.info('Using vanilla Cholesky sampler')
    else:
        if use_cgm == 'True':
            logging.info('Using conjugate gradient method (CGM) based sampler with no preconditioner')
        elif use_cgm == 'Precond':
            logging.info('Using conjugate gradient method (CGM) based sampler with a diagonal/Jacobi preconditioner')

        if error_tolerance is not None:
            logging.info('Error tolerance is %.12f' % error_tolerance)
        else:
            logging.info('No error tolerance; exact solution desired.')
    preconditioned = True if use_cgm == 'Precond' else False

    # for use with `mvn_output_file`
    # each entry is of the form (MCMC iteration #, block number, block size, time to sample, # of CGM iterations or None if vanilla)
    samples = []

    # seed
    if seed != None:
        random.seed(seed)

    # derived stats
    beta_mrg = sp.array(sst_dict['BETA'], ndmin=2).T
    maf = sp.array(sst_dict['MAF'], ndmin=2).T
    n_pst = (n_iter-n_burnin)/thin
    p = len(sst_dict['SNP'])
    n_blk = len(ld_blk)

    # initialization
    beta = sp.zeros((p,1))
    psi = sp.ones((p,1))
    sigma = 1.0
    if phi == None:
        phi = 1.0; phi_updt = True
    else:
        phi_updt = False

    beta_est = sp.zeros((p,1))
    psi_est = sp.zeros((p,1))
    sigma_est = 0.0
    phi_est = 0.0

    # MCMC
    for itr in range(1,n_iter+1):
        if itr % 100 == 0:
            logging.info('--- iter-' + str(itr) + ' ---')

        mm = 0; quad = 0.0
        for kk in range(n_blk):
            if blk_size[kk] == 0:
                continue
            else:
                idx_blk = range(mm,mm+blk_size[kk]) # the indices corresponding to this particular block

                dinvt = ld_blk[kk]+sp.diag(1.0/psi[idx_blk].T[0]) # the precision matrix (without scaling) of the MVN of interest, i.e. (D + Psi^-1)

                start = time.time()
                sample, n_iterations = sample_mvn(dinvt, beta_mrg[idx_blk], sigma, n, len(idx_blk), use_cgm, error_tolerance)
                beta[idx_blk] = sample
                end = time.time()
                # (MCMC iteration #, block number, block size, time to sample, # of CGM iterations or None if vanilla)
                samples.append( (itr, kk, len(idx_blk), (end-start), n_iterations) )
                # logging.info('MVN sampling on a block of size %(blocksize)d took %(time_elapsed)f seconds' % {"blocksize": len(idx_blk),
                #                                                                                               "time_elapsed": (end-start)})

                quad += sp.dot(sp.dot(beta[idx_blk].T, dinvt), beta[idx_blk])
                mm += blk_size[kk]

        err = max(n/2.0*(1.0-2.0*sum(beta*beta_mrg)+quad), n/2.0*sum(beta**2/psi))
        sigma = 1.0/random.gamma((n+p)/2.0, 1.0/err)

        delta = random.gamma(a+b, 1.0/(psi+phi))

        for jj in range(p):
            psi[jj] = gigrnd.gigrnd(a-0.5, 2.0*delta[jj], n*beta[jj]**2/sigma)
        psi[psi>1] = 1.0

        if phi_updt == True:
            w = random.gamma(1.0, 1.0/(phi+1.0))
            phi = random.gamma(p*b+0.5, 1.0/(sum(delta)+w))

        # posterior
        if (itr>n_burnin) and (itr % thin == 0):
            beta_est = beta_est + beta/n_pst
            psi_est = psi_est + psi/n_pst
            sigma_est = sigma_est + sigma/n_pst
            phi_est = phi_est + phi/n_pst

    # convert standardized beta to per-allele beta
    if beta_std == 'False':
        beta_est /= sp.sqrt(2.0*maf*(1.0-maf))

    # write posterior effect sizes
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cgm_type = 'vanilla' if use_cgm == 'False' else 'cgm' if use_cgm == 'True' else 'precondcgm'

    errtol_string = 'exact'
    if error_tolerance is not None and error_tolerance != 0.0:
        errtol_string = 'errtol' + str(error_tolerance)

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if phi_updt == True:
        filename = 'pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, chrom)
    else:
        filename = 'pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)
        
    filename = '_'.join([timestamp, cgm_type, errtol_string, filename])
    eff_file = os.path.join(out_dir, filename)

    with open(eff_file, 'w') as ff:
        for snp, bp, a1, a2, beta in zip(sst_dict['SNP'], sst_dict['BP'], sst_dict['A1'], sst_dict['A2'], beta_est):
            ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp, bp, a1, a2, beta))
    logging.info('Wrote SNP effect sizes to %s' % eff_file)

    # print estimated phi
    if phi_updt == True:
        logging.info('... Estimated global shrinkage parameter: %1.2e ...' % phi_est )

    # save the samples to file
    if mvn_output_file is not None:
        path = os.path.split(mvn_output_file)[0]
        if not os.path.exists(path):
            os.makedirs(path)
        with open(mvn_output_file, 'w+', newline='') as f:
            logging.info('Logging MVN samples in %s' % mvn_output_file)
            writer = csv.writer(f)
            writer.writerow(['mcmc_iteration', 'block_number', 'block_size', 'sampling_time', 'n_iterations'])
            writer.writerows(samples)

    logging.info('... Done ...')


