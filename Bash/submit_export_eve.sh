#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=export_eve
#SBATCH --output=logs/export_eve-%j.log
#SBATCH --mem=8GB
#SBATCH --partition=regular

ml R

# Capture the arguments
file=$1
counter=$2

# Pass the arguments to the R script
Rscript ../Script/export_eve_data.R "$file" "$counter"