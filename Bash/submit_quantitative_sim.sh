#!/bin/bash
#SBATCH --time=3-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_sim_qt
#SBATCH --output=logs/gnn_sim_qt-%j.log
#SBATCH --mem=8GB
#SBATCH --partition=regular

name=${1}
param_set=${2}
nrep=${3}

ml R
Rscript ./Script/quantitative_data.R ${name} \
                                   ${param_set} \
                                   ${nrep}