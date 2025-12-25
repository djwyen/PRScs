import pandas as pd

FILE = './data/ldblk_1kg_eur/ldblk_1kg_chr22.hdf5'

def main():
    hd_file = pd.read_hdf(FILE)
    print(hd_file)


if __name__ == '__main__':
    main()
