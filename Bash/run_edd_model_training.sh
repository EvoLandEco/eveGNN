#!/bin/bash

ml Python/3.8.16-GCCcore-11.2.0
ml SciPy-bundle/2023.02-gfbf-2022b

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_directory>"
    exit 1
fi

# Get the path to the directory from the command line arguments
dir_path=$1

# Construct the full path to the txt file
txt_file="$dir_path/combination.txt"

# Check if the txt file exists
if [ ! -f "$txt_file" ]; then
    echo "Error: File $txt_file not found."
    exit 1
fi

# Read each line from the txt file
while IFS= read -r line || [ -n "$line" ]; do
    # Replace commas with spaces to get the columns as separate arguments
    args=$(echo $line | tr ',' ' ')
    # Call the Python script with the columns as arguments
    python ../Script/test.py $args
done < "$txt_file"
