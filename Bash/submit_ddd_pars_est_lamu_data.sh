#!/bin/bash
#SBATCH --time=3-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_ddd_pars_lamu
#SBATCH --output=logs/gnn_ddd_pars_lamu-%j.log
#SBATCH --mem=24GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/ddd_pars_est_lamu_data.R ${name}