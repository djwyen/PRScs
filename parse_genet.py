#!/usr/bin/env python

"""
Parse the reference panel, summary statistics, and validation set.

"""


import os
import scipy as sp
from scipy.stats import norm
from scipy import linalg
import h5py
import logging
import numpy as np
import csv


def parse_ref(ref_file, chrom):
    logging.info('... parse reference file: %s ...' % ref_file)

    ref_dict = {'CHR':[], 'SNP':[], 'BP':[], 'A1':[], 'A2':[], 'MAF':[]}
    with open(ref_file) as ff:
        header = next(ff)
        for line in ff:
            ll = (line.strip()).split()
            if int(ll[0]) == chrom:
                ref_dict['CHR'].append(chrom)
                ref_dict['SNP'].append(ll[1])
                ref_dict['BP'].append(int(ll[2]))
                ref_dict['A1'].append(ll[3])
                ref_dict['A2'].append(ll[4])
                ref_dict['MAF'].append(float(ll[5]))

    logging.info('... %d SNPs on chromosome %d read from %s ...' % (len(ref_dict['SNP']), chrom, ref_file))
    return ref_dict


def parse_bim(bim_file, chrom):
    logging.info('... parse bim file: %s ...' % (bim_file + '.bim'))

    vld_dict = {'SNP':[], 'A1':[], 'A2':[]}
    with open(bim_file + '.bim') as ff:
        for line in ff:
            ll = (line.strip()).split()
            if int(ll[0]) == chrom:
                vld_dict['SNP'].append(ll[1])
                vld_dict['A1'].append(ll[4])
                vld_dict['A2'].append(ll[5])

    logging.info('... %d SNPs on chromosome %d read from %s ...' % (len(vld_dict['SNP']), chrom, bim_file + '.bim'))
    return vld_dict


def parse_sumstats(ref_dict, vld_dict, sst_file, n_subj):
    logging.info('... parse sumstats file: %s ...' % sst_file)

    ATGC = ['A', 'T', 'G', 'C']
    sst_dict = {'SNP':[], 'A1':[], 'A2':[]}
    with open(sst_file) as ff:
        header = next(ff)
        for line in ff:
            ll = (line.strip()).split()
            if ll[1] in ATGC and ll[2] in ATGC:
                sst_dict['SNP'].append(ll[0])
                sst_dict['A1'].append(ll[1])
                sst_dict['A2'].append(ll[2])

    logging.info('... %d SNPs read from %s ...' % (len(sst_dict['SNP']), sst_file))


    mapping = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    vld_snp = set(zip(vld_dict['SNP'], vld_dict['A1'], vld_dict['A2']))

    ref_snp = set(zip(ref_dict['SNP'], ref_dict['A1'], ref_dict['A2'])) | set(zip(ref_dict['SNP'], ref_dict['A2'], ref_dict['A1'])) | \
              set(zip(ref_dict['SNP'], [mapping[aa] for aa in ref_dict['A1']], [mapping[aa] for aa in ref_dict['A2']])) | \
              set(zip(ref_dict['SNP'], [mapping[aa] for aa in ref_dict['A2']], [mapping[aa] for aa in ref_dict['A1']]))
    
    sst_snp = set(zip(sst_dict['SNP'], sst_dict['A1'], sst_dict['A2'])) | set(zip(sst_dict['SNP'], sst_dict['A2'], sst_dict['A1'])) | \
              set(zip(sst_dict['SNP'], [mapping[aa] for aa in sst_dict['A1']], [mapping[aa] for aa in sst_dict['A2']])) | \
              set(zip(sst_dict['SNP'], [mapping[aa] for aa in sst_dict['A2']], [mapping[aa] for aa in sst_dict['A1']]))

    comm_snp = vld_snp & ref_snp & sst_snp

    logging.info('... %d common SNPs in the reference, sumstats, and validation set ...' % len(comm_snp))


    n_sqrt = sp.sqrt(n_subj)
    sst_eff = {}
    with open(sst_file) as ff:
        header = (next(ff).strip()).split()
        header = [col.upper() for col in header]
        for line in ff:
            ll = (line.strip()).split()
            snp = ll[0]; a1 = ll[1]; a2 = ll[2]
            if a1 not in ATGC or a2 not in ATGC:
                continue
            if (snp, a1, a2) in comm_snp or (snp, mapping[a1], mapping[a2]) in comm_snp:
                if 'BETA' in header:
                    beta = float(ll[3])
                elif 'OR' in header:
                    beta = sp.log(float(ll[3]))

                p = max(float(ll[4]), 1e-323)
                beta_std = sp.sign(beta)*abs(norm.ppf(p/2.0))/n_sqrt
                sst_eff.update({snp: beta_std})
            elif (snp, a2, a1) in comm_snp or (snp, mapping[a2], mapping[a1]) in comm_snp:
                if 'BETA' in header:
                    beta = float(ll[3])
                elif 'OR' in header:
                    beta = sp.log(float(ll[3]))

                p = max(float(ll[4]), 1e-323)
                beta_std = -1*sp.sign(beta)*abs(norm.ppf(p/2.0))/n_sqrt
                sst_eff.update({snp: beta_std})


    sst_dict = {'CHR':[], 'SNP':[], 'BP':[], 'A1':[], 'A2':[], 'MAF':[], 'BETA':[], 'FLP':[]}
    for (ii, snp) in enumerate(ref_dict['SNP']):
        if snp in sst_eff:
            sst_dict['SNP'].append(snp)
            sst_dict['CHR'].append(ref_dict['CHR'][ii])
            sst_dict['BP'].append(ref_dict['BP'][ii])
            sst_dict['BETA'].append(sst_eff[snp])

            a1 = ref_dict['A1'][ii]; a2 = ref_dict['A2'][ii]
            if (snp, a1, a2) in comm_snp:
                sst_dict['A1'].append(a1)
                sst_dict['A2'].append(a2)
                sst_dict['MAF'].append(ref_dict['MAF'][ii])
                sst_dict['FLP'].append(1)
            elif (snp, a2, a1) in comm_snp:
                sst_dict['A1'].append(a2)
                sst_dict['A2'].append(a1)
                sst_dict['MAF'].append(1-ref_dict['MAF'][ii])
                sst_dict['FLP'].append(-1)
            elif (snp, mapping[a1], mapping[a2]) in comm_snp:
                sst_dict['A1'].append(mapping[a1])
                sst_dict['A2'].append(mapping[a2])
                sst_dict['MAF'].append(ref_dict['MAF'][ii])
                sst_dict['FLP'].append(1)
            elif (snp, mapping[a2], mapping[a1]) in comm_snp:
                sst_dict['A1'].append(mapping[a2])
                sst_dict['A2'].append(mapping[a1])
                sst_dict['MAF'].append(1-ref_dict['MAF'][ii])
                sst_dict['FLP'].append(-1)

    return sst_dict


def parse_ldblk(ldblk_dir, sst_dict, chrom):
    logging.info('... parse reference LD on chromosome %d ...' % chrom)

    if '1kg' in os.path.basename(ldblk_dir):
        chr_name = ldblk_dir + '/ldblk_1kg_chr' + str(chrom) + '.hdf5'
    elif 'ukbb' in os.path.basename(ldblk_dir):
        chr_name = ldblk_dir + '/ldblk_ukbb_chr' + str(chrom) + '.hdf5'

    hdf_chr = h5py.File(chr_name, 'r')
    n_blk = len(hdf_chr)
    ld_blk = [sp.array(hdf_chr['blk_'+str(blk)]['ldblk']) for blk in range(1,n_blk+1)]

    snp_blk = []
    for blk in range(1,n_blk+1):
        snp_blk.append([bb.decode("UTF-8") for bb in list(hdf_chr['blk_'+str(blk)]['snplist'])])

    blk_size = []
    mm = 0
    for blk in range(n_blk):
        idx = [ii for (ii, snp) in enumerate(snp_blk[blk]) if snp in sst_dict['SNP']]
        blk_size.append(len(idx))
        if idx != []:
            idx_blk = range(mm,mm+len(idx))
            flip = [sst_dict['FLP'][jj] for jj in idx_blk]
            ld_blk[blk] = ld_blk[blk][sp.ix_(idx,idx)]*sp.outer(flip,flip)

            _, s, v = linalg.svd(ld_blk[blk])
            h = sp.dot(v.T, sp.dot(sp.diag(s), v))
            ld_blk[blk] = (ld_blk[blk]+h)/2            

            mm += len(idx)
        else:
            ld_blk[blk] = sp.array([])

    return ld_blk, blk_size


def compute_sumstats_file(gwas_file, output_file):
    """Computes and writes the sumstats output file used by PRScs in testing."""
    with open(gwas_file) as f1, open(output_file, 'w+') as f2:
        gwas_reader = csv.reader(f1, delimiter='\t')
        sumstats_writer = csv.writer(f2, delimiter='\t')

        header = next(gwas_reader)
        attribute_to_idx = {header[i]: i for i in range(len(header))}
        sumstats_writer.writerow(['SNP', 'A1', 'A2', 'BETA', 'P'])
        for line in gwas_reader:
            snp_id = line[attribute_to_idx['ID']]
            ref, alt = line[attribute_to_idx['REF']], line[attribute_to_idx['ALT']]
            beta = line[attribute_to_idx['BETA']]
            p_value = line[attribute_to_idx['P']]
            if len(ref) == 1 and len(alt) == 1: # only keep those that are single-nucleotide substitutions
                # conventionally A1 is REF and A2 is ALT
                sumstats_writer.writerow([snp_id, ref, alt, beta, p_value])

def filter_geno_file(geno_file, gwas_file, bim_file, output_file):
    """Filters and saves a new geno file containing only those first 1000 SNPs (from the bim file) that survived the GWAS filters."""
    # read the BIM to get the SNP IDs for the columns to save, as well as the indices from the first 1000 lines
    snp_id_to_index = {}
    with open(bim_file) as f:
        for idx, line in enumerate(f):
            l = (line.strip()).split()
            snp_id = l[1]
            snp_id_to_index[snp_id] = idx

    saved_idxs = []
    with open(gwas_file) as f:
        gwas_reader = csv.reader(f, delimiter='\t')
        header = next(gwas_reader)
        attribute_to_idx = {header[i]: i for i in range(len(header))}
        for line in gwas_reader:
            snp_id = line[attribute_to_idx['ID']]
            if snp_id in snp_id_to_index:
                saved_idxs.append(snp_id_to_index[snp_id])

    genotypes = np.loadtxt(fname=geno_file)
    filtered_genotypes = genotypes[:, saved_idxs]
    assert genotypes.shape[0] == filtered_genotypes.shape[0] # sanity check
    assert len(saved_idxs) == filtered_genotypes.shape[1]
    np.savetxt(output_file, filtered_genotypes, fmt='%-1d', delimiter='       ')

def compute_snpinfo_file(gwas_file, geno_file, bim_file, output_file):
    snp_id_to_index = {}
    with open(bim_file) as f:
        for idx, line in enumerate(f):
            l = (line.strip()).split()
            snp_id = l[1]
            snp_id_to_index[snp_id] = idx

    genotypes = np.loadtxt(fname=geno_file)
    
    saved_idxs = []
    with open(gwas_file) as f1, open(output_file, 'w+') as f2:
        gwas_reader = csv.reader(f1, delimiter='\t')
        snpinfo_writer = csv.writer(f2, delimiter='\t')

        header = next(gwas_reader)
        attribute_to_idx = {header[i]: i for i in range(len(header))}
        snpinfo_writer.writerow(['CHR', 'SNP', 'BP', 'A1', 'A2', 'MAF'])

        for line in gwas_reader:
            snp_id = line[attribute_to_idx['ID']]
            if snp_id in snp_id_to_index:
                idx = snp_id_to_index[snp_id]
                saved_idxs.append(idx)

                ref, alt = line[attribute_to_idx['REF']], line[attribute_to_idx['ALT']]
                if len(ref) == 1 and len(alt) == 1: # only keep those that are single-nucleotide substitutions
                    chromosome = line[attribute_to_idx['#CHROM']]
                    position = line[attribute_to_idx['POS']]
                    # calculate the minor allele frequency (MAF) for this SNP
                    maf = np.sum(genotypes[:, idx]) / (2 * genotypes.shape[0]) # the factor of 2 owes to the fact that there are 2 alleles per gene
                    # conventionally A1 is REF and A2 is ALT
                    snpinfo_writer.writerow([chromosome, snp_id, position, ref, alt, maf])

def normalize_matrix(X, axis=0):
    """Normalize a matrix `X` along a given axis."""
    # need to account for columns that are constant and therefore have stdev 0 to prevent divide by zero. Replace with 1's so that the resulting normalized matrix has constant 0's across that feature.
    stdX = np.std(X, axis=axis)
    stdX = np.add(stdX, np.ones(stdX.shape), out=stdX, where=(stdX == 0))
    return (X - np.mean(X, axis=axis)) / stdX

def genotype_to_LD_matrix(X):
    """Computes the LD matrix corresponding to normalized genotype matrix `X`, where each row is an individual."""
    return (X.T @ X) / X.shape[0]

def LD_from_genofile(geno_file):
    genotypes = np.loadtxt(fname=geno_file)
    standardized_genotypes = normalize_matrix(genotypes, axis=0)
    ld_matrix = genotype_to_LD_matrix(standardized_genotypes)

    # we only have the one ld block, but wrap everything to look like we have multiple blocks
    ld_blk = {0: ld_matrix}
    blksize = {0: ld_matrix.shape[0]}
    return ld_blk, blksize


# if __name__ == '__main__':
    # compute_sumstats_file('./data/prscs_test_data/results.PHENO1.glm.linear', './prscs_test_test_data/sumstats.txt')
    # filter_geno_file(geno_file='./data/geno.txt',
    #                  gwas_file='./data/prscs_test_data/results.PHENO1.glm.linear',
    #                  bim_file='./data/prscs_test_data/geno_first_1000_snps.bim',
    #                  output_file='./data/prscs_test_data/first_1000_snps_geno.txt')
    # compute_snpinfo_file(
    #     gwas_file='./data/prscs_test_data/results.PHENO1.glm.linear',
    #     geno_file='./data/geno.txt',
    #     bim_file='./data/prscs_test_data/geno_first_1000_snps.bim',
    #     output_file='./data/prscs_test_data/snpinfo_prscs_test_first_1000_snps'
    # )
