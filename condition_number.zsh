python PRScs.py --ref_dir=data/prscs_test_data --bim_prefix=data/prscs_test_data/geno_first_1000_snps --sst_file=prscs_test_test_data/sumstats.txt --n_gwas=2504 --chrom=1 --phi=1e-2 --use_cgm False --n_iter=10000 --n_burnin=5000 --out_dir=output/july11presentation_data --log_file=logs/july11presentation_data

python PRScs.py --ref_dir=data/ldblk_1kg_eur --bim_prefix=test_data/test --sst_file=test_data/sumstats.txt --n_gwas=200000 --chrom=22 --phi=1e-2 --use_cgm False --n_iter=10000 --n_burnin=5000 --out_dir=output/july11presentation_data --log_file=logs/july11presentation_data
