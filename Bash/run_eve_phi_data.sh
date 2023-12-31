#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_eve_data_start
#SBATCH --output=logs/gnn_eve_data_start-%j.log
#SBATCH --mem=500MB
#SBATCH --partition=regular

# Ensure the script is called with the necessary argument for 'name'
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <name>"
    exit 1
fi

# Ensure the "logs" directory exists
mkdir -p logs

ml R
Rscript -e "devtools::install_github('EvoLandEco/eve')"
Rscript -e "devtools::install_github('EvoLandEco/eveGNN')"

name=$1
index=1  # Initialize index

# Loop over beta_n values
for beta_phi in -0.00025 -0.00035 -0.00045 -0.00055 -0.00065 -0.00075; do

    # Loop over batch values
    for batch in {1..100}; do

        # Call sbatch with the current values of name, beta_n, batch, and index
        sbatch submit_eve_phi.sh "$name" "$beta_phi" "$batch" "$index"

    done  # End batch loop

    # Increment index at the end of each beta_n iteration
    index=$((index + 1))

done  # End beta_n loop
