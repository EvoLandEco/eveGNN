#!/bin/bash
#SBATCH --time=3-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_sim_qt
#SBATCH --output=logs/gnn_sim_qt-%j.log
#SBATCH --mem=32GB
#SBATCH --partition=regular

ml R

Rscript -e "devtools::install_github('EvoLandEco/eve')"

name=${1}
param_set=${2}
nrep=${3}

ml R
Rscript ../Script/qualitative_data.R ${name} \
${param_set} \
${nrep}