#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gnn_bd_pars_with_ddd
#SBATCH --output=logs/gnn_bd_pars_with_ddd-%j.log
#SBATCH --mem=3GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/bd_pars_est_with_ddd.R ${name}
