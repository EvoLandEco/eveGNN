#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gnn_ddd_pars_free_mle
#SBATCH --mem=3GB
#SBATCH --partition=regular

index=${1}
name=${2}

ml R
Rscript ../Script/ddd_pars_est_free_mle.R ${index} ${name}