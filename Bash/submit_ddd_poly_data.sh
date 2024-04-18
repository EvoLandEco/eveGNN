#!/bin/bash
#SBATCH --time=9:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=gnn_ddd_poly
#SBATCH --output=logs/gnn_ddd_poly-%j.log
#SBATCH --mem=16GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/ddd_polymorph_data.R ${name}