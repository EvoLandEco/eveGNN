#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gnn_boot_start
#SBATCH --output=logs/gnn_boot_start-%j.log
#SBATCH --mem=4GB
#SBATCH --partition=regular

ml R

name=${1}

Rscript ../Script/empirical_tree_gnn_bootstrap.R "${name}"