#!/bin/bash
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gnn_boot_start
#SBATCH --output=logs/gnn_boot_start-%j.log
#SBATCH --mem=4GB
#SBATCH --partition=gpu

ml R

name=${1}

Rscript ../Script/empirical_tree_gnn_bootstrap.R "${name}"