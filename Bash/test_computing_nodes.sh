#!/bin/bash
#SBATCH --time=0-01:09:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=test_module
#SBATCH --output=logs/test_module_%j.log
#SBATCH --mem=2GB
#SBATCH --partition=regular

ml R
ml GSL

Rscript ../Script/test_module.R