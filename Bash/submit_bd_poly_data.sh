#!/bin/bash
#SBATCH --time=1-23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gnn_bd_poly
#SBATCH --output=logs/gnn_bd_poly-%j.log
#SBATCH --mem=32GB
#SBATCH --partition=regular

name=${1}

ml R
Rscript ../Script/bd_polymorph_data.R ${name}