#!/bin/bash
#SBATCH --time=6-23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=gnn_bd_pars_free
#SBATCH --output=logs/gnn_bd_pars_free-%j.log
#SBATCH --mem=64GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/bd_pars_est_free_data.R ${name}