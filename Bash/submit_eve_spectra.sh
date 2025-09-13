#!/bin/bash
#SBATCH --time=1-3:09:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=gnn_eve_spectra
#SBATCH --output=logs/gnn_eve_spectra-%j.log
#SBATCH --mem=256GB
#SBATCH --partition=regular

ml R
ml GDAL
ml MPFR
ml tbb
Rscript ../Script/eve_spectra.R