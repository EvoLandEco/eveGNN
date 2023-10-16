#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_ddd_data_start
#SBATCH --output=logs/gnn_ddd_data_start-%j.log
#SBATCH --mem=500MB
#SBATCH --partition=regular

name=${1}

# Ensure the "logs" directory exists
mkdir -p logs

ml R
Rscript -e "devtools::install_github('EvoLandEco/eve')"
Rscript -e "devtools::install_github('EvoLandEco/eveGNN')"

sbatch submit_bd.sh "$name"

# Initialize a counter for the iteration index
index=1

# Iterate through the values 300, 400, 500, ..., 1000 for the variable cap
for cap in {300..1000..100}; do
    # Call the shell script submit_ddd.sh with the variables name, cap, and index as arguments
    sbatch submit_ddd.sh "$name" "$cap" "$index"
    # Increment the index for the next iteration
    ((index++))
done