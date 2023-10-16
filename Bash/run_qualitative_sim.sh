#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_sim_qt_start
#SBATCH --output=logs/gnn_sim_qt_start-%j.log
#SBATCH --mem=1GB
#SBATCH --partition=short

ml R

Rscript -e "devtools::install_github('EvoLandEco/eve')"

name=${1}
nrep=${2}

for (( param_set = 1; param_set <= 21; param_set++ ))
do
sbatch submit_qualitative_sim.sh ${name} \
                                    ${param_set} \
                                    ${nrep}
done