#!/bin/bash
#SBATCH --time=00:29:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gnn_ddd_pars_free_mle
#SBATCH --output=logs/mle/gnn_ddd_mle-%j.log
#SBATCH --mem=3GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/ddd_debug.R ${name}