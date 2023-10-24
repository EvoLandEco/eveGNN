#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=gnn_ddd_pars_free
#SBATCH --output=logs/gnn_ddd_pars_free-%j.log
#SBATCH --mem=32GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/ddd_pars_est_free_data.R ${name}