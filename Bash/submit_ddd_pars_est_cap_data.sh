#!/bin/bash
#SBATCH --time=3-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_ddd_pars_cap
#SBATCH --output=logs/gnn_ddd_pars_cap-%j.log
#SBATCH --mem=16GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/ddd_pars_est_cap_data.R ${name}