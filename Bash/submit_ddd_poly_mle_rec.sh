#!/bin/bash
#SBATCH --time=00:29:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gnn_ddd_poly_mle_rec
#SBATCH --output=logs/gnn_ddd_poly_mle_rec-%j.log
#SBATCH --mem=4GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/ddd_polymorph_mle_rec.R ${name}