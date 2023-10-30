#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_eve_pars_free
#SBATCH --output=logs/gnn_eve_pars_free-%j.log
#SBATCH --mem=16GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/eve_pars_est_free_data.R ${name}