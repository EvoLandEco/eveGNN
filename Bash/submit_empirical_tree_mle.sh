#!/bin/bash
#SBATCH --time=2-23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gnn_emp_mle
#SBATCH --output=logs/gnn_emp_mle-%j.log
#SBATCH --mem=3GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/empirical_tree_mle.R ${name}