"""
Quick script to calculate the time blowup for CGM based sampling (in plaintext) over the vanilla method.
Preliminary numbers (see the TODO below) when evaluated on the test dataset provided in the PRScs repo:
    Vanilla MVN sampling took, on average, 0.0023184492500000022 seconds
    CGM-based MVN sampling took, on average, 0.013257944250000023 seconds
    The blowup is about a factor of 5.718453509387799
"""


VANILLA_LOG = './vanilla.log'
CGM_LOG = './cgm_mvn.log'

def main():
    vanilla_avg = average_time(VANILLA_LOG)
    cgm_avg = average_time(CGM_LOG)
    print("Vanilla MVN sampling took, on average,", vanilla_avg)
    print("CGM-based MVN sampling took, on average,", cgm_avg)
    print("The blowup is about", cgm_avg/vanilla_avg)

def average_time(filepath):
    total_time = 0
    count = 0
    with open(filepath) as f:
        for line in f:
            if "MVN sampling" in line:
                count += 1
                # really scuffed number extraction
                time = float(line.split('took ')[1].split(' seconds')[0])
                total_time += time
    return total_time / count


if __name__ == '__main__':
    # TODO since the order of samplings is deterministic (as it depends only on block sequence) we could actually zip the two together and see what the blowup is on each sample? And also go back and record the block sizes
    # that would be good, as it lets us plot the blowup as a function of the dimension
    main()
