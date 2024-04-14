#!/bin/bash
#SBATCH --time=0-4:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gnn_ddd_stats
#SBATCH --output=logs/gnn_ddd_stats-%j.log
#SBATCH --mem=64GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/ddd_pars_est_free_stats.R ${name}