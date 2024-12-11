#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gnn_eve_pars_est_diffpool_misspec
#SBATCH --output=logs/gnn_eve_pars_est_diffpool-misspec-%j.log
#SBATCH --mem=64GB
#SBATCH --partition=gpu

ml Python/3.8.16-GCCcore-11.2.0
source $HOME/venvs/eve/bin/activate

# Check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Bash Command Line Error. Usage: $0 <name> <task_type> <gnn_depth> <scenario> <partite>"
    exit 1
fi

# Get the arguments
name=$1
task_type=$2
gnn_depth=$3
scenario=$4
partite=$5

# Call the regression model training Python script with the arguments
python ../Script/train_eve_pars_est_DiffPool_reg_misspec.py "$name" "$task_type" "$gnn_depth" "$scenario" "$partite"

deactivate