#!/bin/bash
#SBATCH --time=12:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gnn_eve_pars_est_diffpool_reg
#SBATCH --output=logs/gnn_eve_pars_est_diffpool_reg-%j.log
#SBATCH --mem=64GB
#SBATCH --partition=gpu

ml Python/3.8.16-GCCcore-11.2.0
source $HOME/venvs/eve/bin/activate

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <name> <task_type> <metric>"
    exit 1
fi

# Get the arguments
name=$1
task_type=$2
metric=$3

# Call the Python script with the arguments
python ../Script/train_eve_pars_est_DiffPool_reg.py "$name" "$task_type" "$metric"

deactivate