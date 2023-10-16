#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_bd_training
#SBATCH --output=logs/gnn_bd_training-%j.log
#SBATCH --mem=32GB
#SBATCH --partition=regular

ml Python/3.8.16-GCCcore-11.2.0
source $HOME/venvs/eve/bin/activate

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <name> <set_i> <task_type>"
    exit 1
fi

# Get the arguments from the command line
name=$1
set_i=$2
task_type=$3

# Call the Python script with the provided arguments
python ../Script/train_ddd_eve_bd.py "$name" "$set_i" "$task_type"