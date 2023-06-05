"""
Quick script to calculate the time blowup for CGM based sampling (in plaintext) over the vanilla method.
Preliminary numbers when evaluated on the test dataset provided in the PRScs repo:
    Vanilla MVN sampling took, on average, 0.0023184492500000022 seconds
    CGM-based MVN sampling took, on average, 0.013257944250000023 seconds
    The blowup is about a factor of 5.718453509387799
---
Vanilla block sizes to times: {135: 0.003336923999999999, 176: 0.0017466090000000014, 511: 0.005111019000000006, 178: 0.0014734180000000004}
CGM block sizes to times: {135: 0.005527726999999996, 176: 0.0038376510000000005, 511: 0.024762743000000014, 178: 0.003913789000000005}
On blocksize 135, the factor is 1.6565336819178376
On blocksize 176, the factor is 2.1972009762917732
On blocksize 511, the factor is 4.844971814818138
On blocksize 178, the factor is 2.6562652281972965
"""


VANILLA_LOG = './vanilla.log'
CGM_LOG = './cgm_mvn.log'

def main():
    van_size_to_times = {}
    cgm_size_to_times = {}

    with open(VANILLA_LOG) as van_f, open(CGM_LOG) as cgm_f:
        for van_line, cgm_line in zip(van_f, cgm_f):
            
            if "MVN sampling" in van_line and "MVN sampling" in cgm_line:
                van_blksize, van_time = process_line(van_line)
                if van_blksize not in van_size_to_times:
                    van_size_to_times[van_blksize] = []
                van_size_to_times[van_blksize].append(van_time)

                cgm_blksize, cgm_time = process_line(cgm_line)
                if cgm_blksize not in cgm_size_to_times:
                    cgm_size_to_times[cgm_blksize] = []
                cgm_size_to_times[cgm_blksize].append(cgm_time)
                assert van_blksize == cgm_blksize
    
    van_size_to_avg_time = {size: (sum(times) / len(times)) for size, times in van_size_to_times.items()}
    cgm_size_to_avg_time = {size: (sum(times) / len(times)) for size, times in cgm_size_to_times.items()}
    print("Vanilla block sizes to times:", van_size_to_avg_time)
    print("CGM block sizes to times:", cgm_size_to_avg_time)

    for size, vantime in van_size_to_avg_time.items():
        ratio = cgm_size_to_avg_time[size] / vantime
        print(f"On blocksize {size}, the factor is", ratio)


def process_line(line):
    # really scuffed extraction of relevant quantities from the log
    time = float(line.split('took ')[1].split(' seconds')[0])
    blocksize = int(line.split('size ')[1].split(' took')[0])
    return blocksize, time

def old_main():
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
    main()
