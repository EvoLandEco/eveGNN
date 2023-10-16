#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_ddd_data
#SBATCH --output=logs/gnn_ddd_data-%j.log
#SBATCH --mem=4GB
#SBATCH --partition=regular

name=$1
cap=$2
index=$3

# Call the R script test.R with the variables name and cap as arguments
Rscript ddd_data.R "$name" "$cap" "$index"
