#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=export_eve_start
#SBATCH --output=logs/export_eve_start-%j.log
#SBATCH --mem=500MB
#SBATCH --partition=short

# Ensure the "logs" directory exists
mkdir -p logs

ml R
Rscript -e "devtools::install_github('EvoLandEco/eve')"
Rscript -e "devtools::install_github('EvoLandEco/eveGNN')"

# Get the list of files in the current directory
file_list=$(find . -maxdepth 1 -type f -name "*_*.RData")

# Iterate through the list of files and call submit_export.sh with each file name and index
for file in $file_list; do
    # Extract the index from the file name
    index=$(echo "$file" | sed -n 's/.*_\([0-9]*\)\.RData/\1/p')

    # Call submit_export.sh with the file name and index
    sbatch ../submit_export_eve.sh "$file" "$index"
done