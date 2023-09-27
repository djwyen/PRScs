#!/usr/bin/env python

"""
Markov Chain Monte Carlo (MCMC) sampler for polygenic prediction with continuous shrinkage (CS) priors.

"""


import scipy as sp
from scipy import linalg 
from numpy import random
import math
import numpy as np
import matplotlib.pyplot as plt
import gigrnd
import mvn_cgm
import logging
import time
import csv
import os

N_BENCHMARK_SAMPLES = 1000


def sample_mvn(unscaled_Q, beta_hat, sigma, n, d, use_cgm, max_iterations, error_tolerance):
    if use_cgm == 'False':
        dinvt_chol = linalg.cholesky(unscaled_Q)
        beta_tmp = linalg.solve_triangular(dinvt_chol, beta_hat, trans='T') + sp.sqrt(sigma/n)*random.randn(d,1)
        new_beta = linalg.solve_triangular(dinvt_chol, beta_tmp, trans='N')
        return new_beta, None
    preconditioned = True if use_cgm == 'Precond' else False
    sample, n_iterations = mvn_cgm.sample_mvn_alg_3_4(unscaled_Q, beta_hat, sigma, n, max_iterations=max_iterations, error=error_tolerance, preconditioned=preconditioned)
    return sample.reshape(d, 1), n_iterations

def mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size, n_iter, n_burnin, thin, chrom, out_dir, beta_std, seed, use_cgm, error_tolerance, max_cgm_iters, mvn_output_file=None):
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
            logging.info('No error tolerance set.')
        
        if max_cgm_iters is not None:
            logging.info('Maximum number of CGM iterations allowed is %d' % int(max_cgm_iters))
        else:
            logging.info('No max number of CGM iterations set; allowed to run to completion')

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

    # n_times_non_PSD = 0
    # blk_to_unconditioned_cond_numbers = {} # using the spectral norm
    # blk_to_preconditioned_cond_numbers = {}

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

                # === ALL DEBUG BELOW ===
                # logging.info('LD uncond spec cond number is ' + str(np.linalg.cond(ld_blk[kk], p=2)))
                # precond_ldblk = np.linalg.inv(np.diag(np.diag(ld_blk[kk]))) @ ld_blk[kk]
                # logging.info('LD cond spec cond number is ' + str(np.linalg.cond(precond_ldblk, p=2)))
                # sample = random.multivariate_normal(mean=np.zeros(dinvt.shape[0]), cov=dinvt)[:10]
                # logging.info('printing min and max of sample normal from the prec matrix')
                # logging.info(str(np.min(sample)) + ', ' + str(np.max(sample)))

                # sigma2 = np.squeeze(sigma) # sigma2 sometimes comes in a 2darray
                beta_hat = np.squeeze(beta_mrg[idx_blk])
                # scaled_Q = (n / sigma2)*dinvt
                # Q_sample = math.sqrt(n / sigma2)*mvn_cgm.cholesky_sample(0, dinvt)
                # Q_sample = np.squeeze(Q_sample)
                # eta = Q_sample + ((n / sigma2)*beta_hat)

                if kk == 2: # which we know to be the largest block, size 511
                    if itr == 1:
                        # # TODO refactor to work on all blocks, not just hardcoded blk2
                        # for block_folder in [f'blk{block_number}' for block_number in range(n_blk)]:
                        #     path = os.path.join('sampledata/')

                        for folder in ['LD', 'psi', 'psi_inv', 'sigma2', 'eta', 'beta_hat', 'beta', 'delta']:
                            path = os.path.join('sampledata/blk2/', folder)
                            # path = os.path.join('benchmark_samples/blk2/', folder)
                            if not os.path.exists(path):
                                os.makedirs(path)
                        # this is identical every time, so we save it only on the first iteration
                        np.savetxt(f'sampledata/blk2/beta_hat/{itr}.csv', beta_hat.reshape(1, -1), delimiter=',')
                    if itr >= n_burnin:
                        # TODO can save more than one iteration later
                        if itr == 888:
                            np.savetxt(f'sampledata/blk2/LD/{itr}.csv', 0.5 * ld_blk[kk], delimiter=',')
                            np.savetxt(f'sampledata/blk2/psi/{itr}.csv', psi[idx_blk].T[0], delimiter=',')
                            np.savetxt(f'sampledata/blk2/psi_inv/{itr}.csv', 1.0/psi[idx_blk].T[0], delimiter=',')
                            np.savetxt(f'sampledata/blk2/sigma2/{itr}.csv', np.reshape(sigma, (1, 1)), delimiter=',')
                            np.savetxt(f'sampledata/blk2/beta/{itr}.csv', beta[idx_blk], delimiter=',')
                            np.savetxt(f'sampledata/blk2/delta/{itr}.csv', delta[idx_blk], delimiter=',')
                            # np.savetxt(f'sampledata/blk2/eta/{itr}.csv', eta.reshape(1, -1), delimiter=',')

                # try:
                #     np.linalg.cholesky(dinvt)
                # except:
                #     n_times_non_PSD += 1

                # uncond_cond_number = np.linalg.cond(dinvt, p=2)
                # precond_dinvt = np.linalg.inv(np.diag(np.diag(dinvt))) @ dinvt
                # precond_cond_number = np.linalg.cond(precond_dinvt, p=2)
                
                # if kk not in blk_to_unconditioned_cond_numbers:
                #     blk_to_unconditioned_cond_numbers[kk] = []
                #     blk_to_preconditioned_cond_numbers[kk] = []
                # blk_to_unconditioned_cond_numbers[kk].append(uncond_cond_number)
                # blk_to_preconditioned_cond_numbers[kk].append(precond_cond_number)
                # === ALL DEBUG ABOVE ===

                start = time.time()
                beta_sample, n_iterations = sample_mvn(dinvt, beta_mrg[idx_blk], sigma, n, len(idx_blk), use_cgm, max_cgm_iters, error_tolerance)
                
                if kk == 2 and itr == 888:
                    benchmark_beta_samples = []
                    for ii in range(N_BENCHMARK_SAMPLES):
                        beta_sample_i, _ = sample_mvn(dinvt, beta_mrg[idx_blk], sigma, n, len(idx_blk), use_cgm, max_cgm_iters, error_tolerance)
                        benchmark_beta_samples.append(beta_sample_i)
                    benchmark_beta_samples = np.array(benchmark_beta_samples).squeeze()
                    np.savetxt('benchmark_samples/blk2/beta.csv', benchmark_beta_samples, delimiter=',')
                
                end = time.time()
                beta[idx_blk] = beta_sample

                assert use_cgm == 'False' or n_iterations > 0

                # (MCMC iteration #, block number, block size, time to sample, # of CGM iterations or None if vanilla)
                if mvn_output_file is not None:
                    samples.append( (itr, kk, len(idx_blk), (end-start), n_iterations) )
                # logging.info('MVN sampling on a block of size %(blocksize)d took %(time_elapsed)f seconds' % {"blocksize": len(idx_blk),
                #                                                                                               "time_elapsed": (end-start)})

                quad += sp.dot(sp.dot(beta[idx_blk].T, dinvt), beta[idx_blk])
                mm += blk_size[kk]

        err = max(n/2.0*(1.0-2.0*sum(beta*beta_mrg)+quad), n/2.0*sum(beta**2/psi))
        sigma = 1.0/random.gamma((n+p)/2.0, 1.0/err)

        if itr == 888:
            benchmark_sigma_samples = []
            for ii in range(N_BENCHMARK_SAMPLES):
                sigma_i = 1.0/random.gamma((n+p)/2.0, 1.0/err)
                benchmark_sigma_samples.append(sigma_i)
            np.savetxt('benchmark_samples/blk2/sigma2.csv', np.array(benchmark_sigma_samples).squeeze(), delimiter=',')

        delta = random.gamma(a+b, 1.0/(psi+phi))

        blk2_indices = range(311,822) # hardcoded lol

        if itr == 888:
            benchmark_delta_samples = []
            for ii in range(N_BENCHMARK_SAMPLES):
                delta_i = random.gamma(a+b, 1.0/(psi+phi))
                benchmark_delta_samples.append(delta_i[blk2_indices])
            benchmark_delta_samples = np.array(benchmark_delta_samples).squeeze()
            np.savetxt('benchmark_samples/blk2/delta.csv', benchmark_delta_samples, delimiter=',')

        for jj in range(p):
            psi[jj] = gigrnd.gigrnd(a-0.5, 2.0*delta[jj], n*beta[jj]**2/sigma)
        psi[psi>1] = 1.0

        if itr == 888:
            benchmark_psi_samples = []
            for ii in range(N_BENCHMARK_SAMPLES):
                psi_tmp = np.array([0]*p)
                for jj in blk2_indices:
                    psi_tmp[jj] = gigrnd.gigrnd(a-0.5, 2.0*delta[jj], n*beta[jj]**2/sigma)
                psi_tmp[psi_tmp>1] = 1.0
                benchmark_psi_samples.append(psi_tmp[blk2_indices])
            np.savetxt('benchmark_samples/blk2/psi.csv', np.array(benchmark_psi_samples).squeeze(), delimiter=',')

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

    errtol_string = ''
    if error_tolerance is not None and error_tolerance != 0.0:
        errtol_string = '_errtol=' + str(error_tolerance)
    
    cgm_iters_string = ''
    if max_cgm_iters is not None:
        cgm_iters_string = '_maxCGMiters=' + str(max_cgm_iters)

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if phi_updt == True:
        filename = 'pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, chrom)
    else:
        filename = 'pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)
        
    filename = '_'.join([timestamp, cgm_type, errtol_string, cgm_iters_string, filename])
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

    # logging.info('Number of times the prec matrix was not PSD: ' + str(n_times_non_PSD))
    logging.info('... Done ...')

    # Plotting evolution of condition number by iteration:
    # for kk in range(n_blk):
    #     if kk in blk_to_preconditioned_cond_numbers:
    #         plt.plot(blk_to_preconditioned_cond_numbers[kk], label=f'block {str(kk)}')
    # plt.xlabel("Iteration number")
    # plt.ylabel("Spectral condition number")
    # plt.legend()
    # plt.title("Preconditioned condition numbers by iteration")
    # plt.show()

    # for kk in range(n_blk):
    #     if kk in blk_to_unconditioned_cond_numbers:
    #         plt.plot(blk_to_unconditioned_cond_numbers[kk], label=f'block {str(kk)}')
    # plt.yscale('log')
    # plt.xlabel("Iteration number")
    # plt.ylabel("Spectral condition number")
    # plt.legend()
    # plt.title("Unconditioned condition numbers by iteration")
    # plt.show()

