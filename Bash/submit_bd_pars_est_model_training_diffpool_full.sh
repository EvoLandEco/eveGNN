#!/bin/bash
#SBATCH --time=5:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gnn_bd_pars_est_diffpool_full
#SBATCH --output=logs/gnn_bd_pars_est_diffpool_full-%j.log
#SBATCH --mem=24GB
#SBATCH --partition=gpu

ml Python/3.8.16-GCCcore-11.2.0
source $HOME/venvs/eve/bin/activate

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <name> <task_type> <gnn_depth>"
    exit 1
fi

# Get the arguments
name=$1
task_type=$2
gnn_depth=$3

# Call the Python script with the arguments
python ../Script/train_bd_pars_est_DiffPool_full.py "$name" "$task_type" "$gnn_depth"

deactivate