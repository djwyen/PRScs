"""
Quick script to calculate the average per-SNP estimated effect scores, so as to compare them between different MVN samplers.
"""

import csv
import os
import contextlib
import numpy as np
import pandas as pd

RESULTS_DIR = './output/avgtrials/'

def calc_stats(filenames):
    # way of opening variable number of files simultaneously
    # taken from https://stackoverflow.com/questions/4617034/how-can-i-open-multiple-files-using-with-open-in-python
    snp_to_stats = {} # maps a given SNP name to a tuple (mean, stdev)
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(open(os.path.join(RESULTS_DIR, fname))) for fname in filenames]
        readers = [csv.reader(file, delimiter='\t') for file in files]
        for lines in zip(*readers):
            # hardcoded positions
            snp_name = lines[0][1]
            beta_sizes = [float(line[5]) for line in lines]
            snp_to_stats[snp_name] = (np.average(beta_sizes), np.std(beta_sizes))
    return pd.DataFrame(data=snp_to_stats, index=['mean', 'stdev']).transpose()


def main():
    filenames = os.listdir(RESULTS_DIR)

    vanilla_filenames = filter(lambda filename: 'vanilla' in filename, filenames)
    cgm_filenames = filter(lambda filename: '_cgm_' in filename, filenames)
    precondcgm_filenames = filter(lambda filename: '_precondcgm_' in filename, filenames)

    vanilla_stats = calc_stats(vanilla_filenames)
    cgm_stats = calc_stats(cgm_filenames)
    precondcgm_stats = calc_stats(precondcgm_filenames)

    vanilla_avgs = vanilla_stats['mean'].to_numpy()
    cgm_avgs = cgm_stats['mean'].to_numpy()
    precondcgm_avgs = precondcgm_stats['mean'].to_numpy()

    print(vanilla_avgs[10:20])
    print(cgm_avgs[10:20])
    print(precondcgm_avgs[10:20])
    print('\n')

    print(vanilla_stats['stdev'].to_numpy()[10:20])
    print(cgm_stats['stdev'].to_numpy()[10:20])
    print(precondcgm_stats['stdev'].to_numpy()[10:20])

    difference = vanilla_avgs - cgm_avgs

    print(difference[:10])
    print((vanilla_avgs - precondcgm_avgs)[:10])
    print((cgm_avgs - precondcgm_avgs)[:10])



if __name__ == '__main__':
    main()