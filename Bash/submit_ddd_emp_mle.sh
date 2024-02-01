#!/bin/bash
#SBATCH --time=3:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gnn_emp_mle_ddd
#SBATCH --output=logs/gnn_emp_mle_ddd-%j.log
#SBATCH --mem=3GB
#SBATCH --partition=regular

file_name=${1}
family_name=${2}
tree_name=${3}

ml R
Rscript ../../../../Script/ddd_emp_mle.R ${file_name} ${family_name} ${tree_name}