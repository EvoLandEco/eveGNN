#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gnn_emp_mle_ddd_uncertainty
#SBATCH --output=logs/gnn_emp_mle_ddd_uncertainty-%j.log
#SBATCH --mem=1GB
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
    sbatch ../submit_ddd_emp_mle_uncertainty_worker.sh ${family_name} ${tree_name} ${lambda} ${mu} ${cap} ${i}
done
