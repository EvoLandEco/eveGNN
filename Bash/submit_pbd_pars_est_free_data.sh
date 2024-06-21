#!/bin/bash
#SBATCH --time=1-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=gnn_pbd_pars_free
#SBATCH --output=logs/gnn_pbd_pars_free-%j.log
#SBATCH --mem=48GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/pbd_pars_est_free_data.R ${name}