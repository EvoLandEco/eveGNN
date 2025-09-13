#!/bin/bash
#SBATCH --time=1-23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=gnn_eve_spectra
#SBATCH --output=logs/gnn_ddd_pars_free-%j.log
#SBATCH --mem=128GB
#SBATCH --partition=regular

ml R
ml GDAL
ml MPFR
ml tbb
Rscript ../spectra.R