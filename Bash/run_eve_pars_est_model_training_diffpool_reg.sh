#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_eve_pars_model_train_diffpool_reg_start
#SBATCH --output=logs/gnn_eve_pars_model_train_diffpool_reg_start-%j.log
#SBATCH --mem=500MB
#SBATCH --partition=regular

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <name>"
    exit 1
fi

# Get the path from the command line arguments
name=$1

# Define an array of possible metric values
metrics=("pd" "ed" "nnd")

# Iterate through the folders under the specified path
for task_type in "$name"/*; do
    # Check if it's a directory (folder)
    if [ -d "$task_type" ]; then
        # Extract the folder name from the path
        task_type=$(basename "$task_type")

        # Iterate over each metric
        for metric in "${metrics[@]}"; do
            echo "Submitting job for $task_type with metric $metric"
            # Call the other script with "name", "task_type", and "metric" as arguments
            sbatch submit_eve_pars_est_model_training_diffpool_reg.sh "$name" "$task_type" "$metric"
        done
    fi
done