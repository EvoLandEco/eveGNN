#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_ddd_pars_model_train_start
#SBATCH --output=logs/gnn_ddd_pars_model_train_start-%j.log
#SBATCH --mem=500MB
#SBATCH --partition=regular

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <name>"
    exit 1
fi

# Get the path from the command line arguments
name=$1

# Iterate through the folders under the specified path
for task_type in "$name"/*; do
    # Check if it's a directory (folder)
    if [ -d "$task_type" ]; then
        # Extract the folder name from the path
        task_type=$(basename "$task_type")
        # Call the other script with "name" and "task_type" as arguments
        sbatch submit_ddd_pars_est_model_training.sh "$name" "$task_type"
    fi
done