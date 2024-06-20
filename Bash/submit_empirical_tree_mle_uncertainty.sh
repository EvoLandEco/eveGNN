#!/bin/bash
#SBATCH --time=00:09:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gnn_emp_mle_un
#SBATCH --output=logs/gnn_emp_mle_un-%j.log
#SBATCH --mem=1GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/empirical_tree_mle_uncertainty.R ${name}