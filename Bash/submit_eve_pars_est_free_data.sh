#!/bin/bash
#SBATCH --time=3-23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=gnn_eve_pars_free
#SBATCH --output=logs/gnn_eve_pars_free-%j.log
#SBATCH --mem=80GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/eve_pars_est_free_data.R ${name}