#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gnn_ddd_pars_est_val2_diffpool
#SBATCH --output=logs/gnn_ddd_pars_est_val2_diffpool-%j.log
#SBATCH --mem=64GB
#SBATCH --partition=gpu

module --ignore_cache load "Python/3.8.16-GCCcore-11.2.0"
source $HOME/venvs/eve/bin/activate

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
echo "Usage: $0 <name> <task_type>"
exit 1
fi

# Get the arguments
name=$1
gnn_depth=$2

# Call the Python script with the arguments
python ../Script/val_ddd_pars_est_DiffPool2.py "$name" "$gnn_depth"

deactivate