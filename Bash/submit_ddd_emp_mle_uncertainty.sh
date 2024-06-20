#!/bin/bash
#SBATCH --time=9:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gnn_emp_mle_ddd_uncertainty
#SBATCH --output=logs/gnn_emp_mle_ddd_uncertainty-%j.log
#SBATCH --mem=4GB
#SBATCH --partition=regular

family_name=${1}
tree_name=${2}
lambda=${3}
mu=${4}
cap=${5}

ml R

# Run 100 times and pass the index to the R script
for i in {1..100}
do
    Rscript ../../Script/ddd_emp_mle_uncertainty.R ${family_name} ${tree_name} ${lambda} ${mu} ${cap} ${i}
done
