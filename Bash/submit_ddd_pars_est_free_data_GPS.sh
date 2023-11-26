#!/bin/bash
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gnn_ddd_pars_free_gps
#SBATCH --output=logs/gnn_ddd_pars_free_gps-%j.log
#SBATCH --mem=16GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/ddd_pars_est_free_data_GPS.R ${name}