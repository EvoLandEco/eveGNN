#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gnn_ddd_boot
#SBATCH --output=logs/gnn_ddd_boot-%j.log
#SBATCH --mem=3GB
#SBATCH --partition=regular

lambda=${1}
mu=${2}
cap=${3}
ntips=${4}
family_name=${5}
tree_name=${6}
path=${7}

ml R

Rscript ../Script/ddd_emp_bootstrap.R "${lambda}" "${mu}" "${cap}" "${ntips}" "${family_name}" "${tree_name}" "${path}"
