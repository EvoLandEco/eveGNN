#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=gnn_ddd_pars_lamu
#SBATCH --output=logs/gnn_ddd_pars_lamu-%j.log
#SBATCH --mem=12GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/ddd_pars_est_lamu_data.R ${name}