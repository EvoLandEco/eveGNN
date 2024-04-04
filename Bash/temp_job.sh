#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=temp
#SBATCH --output=logs/temp-%j.log
#SBATCH --mem=100MB
#SBATCH --partition=regular

# Rename models GNN depth 2 and 3 stacking
sbatch --dependency=afterok:7867790 7867790.sh
sbatch --dependency=afterok:7867791 7867791.sh

# Rename models GNN depth 1, 2 and 3 original
sbatch --dependency=afterok:7876227 7876227.sh
sbatch --dependency=afterok:7876228 7876228.sh
sbatch --dependency=afterok:7876229 7876229.sh

# Rename models GNN depth 1 bagging
sbatch --dependency=afterok:7919414 7919414.sh