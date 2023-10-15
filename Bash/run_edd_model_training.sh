#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_training_start
#SBATCH --output=logs/gnn_training_start-%j.log
#SBATCH --mem=500MB
#SBATCH --partition=short

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_directory>"
    exit 1
fi

# Get the path to the directory from the command line arguments
name=$1

bash split_data.sh $name

# Construct the full path to the txt file
combination="$name/combination.txt"

# Check if the txt file exists
if [ ! -f "$combination" ]; then
    echo "Error: File $combination not found."
    exit 1
fi

# Read each line from the txt file
while IFS= read -r line || [ -n "$line" ]; do
    # Replace commas with spaces to get the columns as separate arguments
    args=$(echo $line | tr ',' ' ')
    # Call the Python script with the columns as arguments
    sbatch submit_edd_model_training.sh $name $args
done < "$combination"
