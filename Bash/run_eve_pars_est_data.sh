#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_eve_pars_start
#SBATCH --output=logs/gnn_eve_pars_start-%j.log
#SBATCH --mem=500MB
#SBATCH --partition=regular

# Ensure the script is called with the necessary argument for 'name'
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <name>"
    exit 1
fi

name=${1}

# Ensure the "logs" directory exists
mkdir -p logs

ml R
Rscript -e "devtools::install_github('EvoLandEco/eve')"
Rscript -e "devtools::install_github('EvoLandEco/eveGNN')"
Rscript -e "devtools::install_github('rsetienne/DDD')"

echo "Submitting job for eve free data"
sbatch submit_eve_pars_est_free_data.sh "$name"