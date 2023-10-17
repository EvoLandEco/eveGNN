#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_bd_training_start
#SBATCH --output=logs/gnn_bd_training_start-%j.log
#SBATCH --mem=500MB
#SBATCH --partition=short

ml Python/3.8.16-GCCcore-11.2.0
source $HOME/venvs/eve/bin/activate

# Function to submit job
submit_job() {
  local name=$1
  local set_i=$2
  local task_type=$3  # This will be either DDD_TES or EVE_TES

  echo "Submitting job for $name, $set_i, $task_type"
  sbatch submit_ddd_eve_bd_model_training.sh "$name" "$set_i" "$task_type"
}

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <name>"
    exit 1
fi

# Get the path to the directory from the command line arguments
name=$1

# Path to the directories
ddd_path="$name/DDD_TES"
eve_path="$name/EVE_TES"

# Iterate through the sets in DDD_TES and submit jobs
for set_dir in "$ddd_path"/set_*; do
    set_i=$(basename "$set_dir")
    submit_job "$name" "$set_i" "DDD_TES"
done

# Iterate through the sets in EVE_TES and submit jobs
for set_dir in "$eve_path"/set_*; do
    set_i=$(basename "$set_dir")
    submit_job "$name" "$set_i" "EVE_TES"
done
