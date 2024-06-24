#!/bin/bash
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=gnn_bd_pars_free
#SBATCH --output=logs/gnn_bd_pars_free-%j.log
#SBATCH --mem=48GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/bd_pars_est_free_data_rebalanced.R ${name}