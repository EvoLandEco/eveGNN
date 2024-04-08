#!/bin/bash
#SBATCH --time=00:09:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=submit_ddd_pars_ott_mle
#SBATCH --mem=3GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/ddd_pars_est_free_ott_mle.R ${name}