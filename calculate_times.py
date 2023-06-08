"""
Quick script to calculate the times/avg iterations recorded for each sample from PRScs, put them into a dataframe to be workable, and plot them.
"""
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import math

# TODO run exact CGM of both types to baseline
CSV_DIR = 'mvn_csvs/'
VANILLA_CSV = 'vanilla.csv'

def calculate_stats(filepath):
    # maps a block number to its block size
    blkno_to_blocksize = {}
    # maps a block number to a list of times it took to sample an MVN for that block
    blkno_to_times = {}
    # maps a block number to a list of the number of iterations it took to sample an MVN for that block
    blkno_to_n_iterations = {}

    with open(filepath) as f:
        reader = csv.reader(f)
        header = next(reader)
        for entry in reader:
            mcmc_iteration, block_number, block_size, sampling_time, n_iterations = entry
            mcmc_iteration = int(mcmc_iteration)
            block_number = int(block_number)
            block_size = int(block_size)
            sampling_time = float(sampling_time)
            n_iterations = int(n_iterations) if n_iterations != '' else None

            if block_number not in blkno_to_blocksize:
                # we assume each will be updated in parallel
                blkno_to_blocksize[block_number] = block_size
                blkno_to_times[block_number] = []
                blkno_to_n_iterations[block_number] = []
            
            blkno_to_times[block_number].append(sampling_time)
            if n_iterations is not None:
                blkno_to_n_iterations[block_number].append(n_iterations)
    
    blkno_to_avgtime = {blkno: sum(times) / len(times) for blkno, times in blkno_to_times.items()}
    if n_iterations is not None:
        blkno_to_avgiters = {blkno: sum(iters) / len(iters) for blkno, iters in blkno_to_n_iterations.items()}
    else:
        blkno_to_avgiters = None

    # TODO could also calculate the stdev on the times, iters and return that too for more info.
    # the CGM alg is deterministic, but the particular eta we use is probabilistic due to containing an MVN sample, and also the Psi inv changes each mcmc iteration.
    return blkno_to_blocksize, blkno_to_avgtime, blkno_to_avgiters

def main():

    blocksizes = []
    vanilla_blocksize_to_avgtime = {}

    # maps a particular error rate (under a particular algorithm) to a list of values, which are in the listed order of the blocksizes
    # so for example, the entry '0.1' : [5, 10, 3] with blocksizes = [30, 50, 20] means that, at error rate 0.1, on blocksize 30 it has value 5, on blocksize 50 it has value 10, and on blocksize 20 it has value 3
    # this format is convenient for later creating a pandas dataframe
    precond_err_to_avgtimes = {}
    nonprecond_err_to_avgtimes = {}

    precond_err_to_avgiters = {}
    nonprecond_err_to_avgiters = {}

    # we will load up and scan the vanilla csv separately since it contains some crucial info we use later
    blkno_to_blocksize, blkno_to_avgtime, _ = calculate_stats(os.path.join(CSV_DIR, VANILLA_CSV))
    for blkno in range(len(blkno_to_blocksize)):
        blocksize = blkno_to_blocksize[blkno]
        blocksizes.append(blocksize)
        vanilla_blocksize_to_avgtime[blocksize] = blkno_to_avgtime[blkno]


    for filename in os.listdir(CSV_DIR):
        if 'vanilla' in filename:
            # we loaded up vanilla separately
            continue
        assert 'cgm' in filename
        # infer the properties of this file from the name
        samplertype = ''
        error_tolerance = None
        samplertype = filename.split('_')[0] # `preconditioned` or `nonpreconditioned`
        ending = filename.split('_')[-1]
        if ending != 'cgm.csv': # i.e. this run was with a specified error tolerance
            error_tolerance = float(ending.split('.csv')[0])
        else:
            error_tolerance = 0.0

        blkno_to_blocksize, blkno_to_avgtime, blkno_to_avgiters = calculate_stats(os.path.join(CSV_DIR, filename))
        
        if samplertype == 'preconditioned':
            precond_err_to_avgtimes[error_tolerance] = [blkno_to_avgtime[blkno] for blkno in range(len(blkno_to_blocksize))]
            precond_err_to_avgiters[error_tolerance] = [blkno_to_avgiters[blkno] for blkno in range(len(blkno_to_blocksize))]
        else:
            nonprecond_err_to_avgtimes[error_tolerance] = [blkno_to_avgtime[blkno] for blkno in range(len(blkno_to_blocksize))]
            nonprecond_err_to_avgiters[error_tolerance] = [blkno_to_avgiters[blkno] for blkno in range(len(blkno_to_blocksize))]

    
    # for the CGM methods, desired pandas dataframes
    # print(precond_err_to_avgtimes)
    
    precond_avgtime = pd.DataFrame(data=precond_err_to_avgtimes, index=blocksizes)
    precond_avgiters = pd.DataFrame(data=precond_err_to_avgiters, index=blocksizes)
    nonprecond_avgtime = pd.DataFrame(data=nonprecond_err_to_avgtimes, index=blocksizes)
    nonprecond_avgiters = pd.DataFrame(data=nonprecond_err_to_avgiters, index=blocksizes)
    
    # since blocks were appended in the order of their block number, they aren't sorted on block size; so we sort them here
    def sort_dataframe(df):
        df = df.sort_index()
        return df.reindex(sorted(df.columns), axis=1)

    precond_avgtime = sort_dataframe(precond_avgtime)
    precond_avgiters = sort_dataframe(precond_avgiters)
    nonprecond_avgtime = sort_dataframe(nonprecond_avgtime)
    nonprecond_avgiters = sort_dataframe(nonprecond_avgiters)

    # data is now in a very convenient form for plotting
    # colormap for being able to scale the lines sensibly
    viridis = mpl.colormaps['viridis'].resampled(4)
    viridis_r = mpl.colormaps['viridis_r'].resampled(4)
    error_to_fraction = lambda x: math.log10(1/x) / 10

    plt.plot(blocksizes, [vanilla_blocksize_to_avgtime[blocksize] for blocksize in blocksizes], '.-', label='Cholesky', color='red')
    plt.plot(precond_avgtime.index, precond_avgtime[0.0], '.-', label='exact preconditioned CGM', color=viridis(0))
    plt.plot(precond_avgtime.index, precond_avgtime[1e-8], '.-', label='1e-8 preconditioned CGM', color=viridis(.8))
    plt.plot(precond_avgtime.index, precond_avgtime[1e-7], '.-', label='1e-7 preconditioned CGM', color=viridis(.7))
    plt.plot(precond_avgtime.index, precond_avgtime[1e-5], '.-', label='1e-5 error precond', color=viridis(.5))
    plt.plot(precond_avgtime.index, precond_avgtime[1e-3], '.-', label='1e-3 error precond', color=viridis(.3))
    plt.plot(nonprecond_avgtime.index, nonprecond_avgtime[0.0], '.--', label='exact nonpreconditioned CGM', color=viridis(0))
    plt.plot(nonprecond_avgtime.index, nonprecond_avgtime[1e-8], '.--', label='1e-8 nonpreconditioned CGM', color=viridis(.8))
    plt.plot(nonprecond_avgtime.index, nonprecond_avgtime[1e-7], '.--', label='1e-7 nonpreconditioned CGM', color=viridis(.7))
    plt.plot(nonprecond_avgtime.index, nonprecond_avgtime[1e-5], '.--', label='1e-5 error nonprecond', color=viridis(.5))
    plt.plot(nonprecond_avgtime.index, nonprecond_avgtime[1e-3], '.--', label='1e-3 error nonprecond', color=viridis(.3))

    # TODO break the x axis potentially?
    # plt.xticks(blocksizes)
    plt.gca().set_xticks(blocksizes, minor=True)
    plt.legend()
    plt.title('Average sampling time (seconds) by blocksize')
    plt.show()

    for err in [1e-3, 1e-5, 1e-7, 1e-8]:
        label = 'exact' if err == 0.0 else str(err) # well, it's not that sensible to think about this as exact obviously is a straight line?
        plt.plot(precond_avgiters.index, precond_avgiters[err], '.-',
                 label=label + ' precond',
                 color=viridis(error_to_fraction(err)))
        plt.plot(nonprecond_avgiters.index, nonprecond_avgiters[err], '.--',
                 label=label + ' nonprecond',
                 color=viridis(error_to_fraction(err)))
    plt.plot(precond_avgiters.index, precond_avgiters[0.0],
             label='exact') # exact sampling is the max number of iterations regardless of whether we precondition
    plt.gca().set_xticks(blocksizes, minor=True)
    plt.legend()
    plt.title('Average number of iterations by blocksize')
    plt.show()


if __name__ == '__main__':
    main()
