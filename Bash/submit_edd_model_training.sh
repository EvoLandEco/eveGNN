#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_training
#SBATCH --output=logs/gnn_training-%j.log
#SBATCH --mem=32GB
#SBATCH --partition=regular

ml Python/3.8.16-GCCcore-11.2.0
source $HOME/venvs/eve/bin/activate

# Check if at least two arguments are provided (name and at least one arg)
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <name> <arg1> [arg2 arg3 ... argN]"
    exit 1
fi

# Get the name argument
name=$1

# Shift the positional parameters so $@ contains only the args
shift 1

# Call the Python script with all arguments
python ../Script/train_edd_qualitative_model.py $name $@

deactivate