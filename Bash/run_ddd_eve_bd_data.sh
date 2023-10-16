#!/bin/bash
#SBATCH --time=01:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=ddd_eve_bd_data
#SBATCH --output=logs/ddd_eve_bd_data-%j.log
#SBATCH --mem=32GB
#SBATCH --partition=regular

name=${1}

# Ensure the "logs" directory exists
mkdir -p logs

ml R
Rscript -e "devtools::install_github('EvoLandEco/eve')"
Rscript -e "devtools::install_github('EvoLandEco/eveGNN')"

Rscript ../Script/ddd_eve_bd_data.R ${name}