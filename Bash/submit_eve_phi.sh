#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_eve_data
#SBATCH --output=logs/gnn_eve_data-%j.log
#SBATCH --mem=500MB
#SBATCH --partition=regular


# Assign command line arguments to variables
name=$1
beta_phi=$2
batch=$3
index=$4

ml R

# Call the R script with the necessary arguments
Rscript ../Script/eve_phi_data.R "$name" "$beta_phi" "$batch" "$index"