#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_directory>"
    exit 1
fi

# Get the directory path argument
path=$1

# Check if the directory exists
if [ ! -d "$path" ]; then
    echo "Error: Directory $path not found."
    exit 1
fi

# Check if params.txt exists in the directory
if [ ! -f "$path/params.txt" ]; then
    echo "Error: params.txt not found in $path."
    exit 1
fi

# Call the Python script
python ../Script/split_data.py "$path"

# Check if the Python script executed successfully
if [ "$?" -ne 0 ]; then
    echo "Error: Failed to execute Python script."
    exit 1
fi

echo "Script executed successfully, output written to $path/output.txt"
