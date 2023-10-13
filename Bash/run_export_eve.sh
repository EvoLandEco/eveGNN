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
file_list=$(find . -type f)

# Initialize a counter
counter=1

# Iterate through the list of files and call submit_export.sh with each file name
for file in $file_list; do
    sbatch ./submit_export.sh "$file" "$counter"
    # Increment the counter
    ((counter++))
done
