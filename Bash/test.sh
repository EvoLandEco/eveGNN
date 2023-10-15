#!/bin/bash

# Get the name argument
name=$1

# Check if the name argument is provided
if [[ -z $name ]]; then
    echo "Usage: $0 <name>"
    exit 1
fi

# File name
file="${name}/params.txt"

# Check if the file exists
if [[ ! -f $file ]]; then
    echo "Error: File $file not found."
    exit 1
fi

# Temporary file to hold intermediate results
temp_file=$(mktemp)

# Process the data file to get unique groups and their row indices
awk '
  NR > 1 {
    # Create a group identifier
    group = $5" "$6" "$7" "$8

    # Collect row indices per group
    groups[group] = groups[group] ? groups[group]","NR : NR
  }
  END {
    # Print the result
    for (group in groups)
      print group" "groups[group]
  }
' $file > $temp_file

# Sort the result based on the group identifier
sort $temp_file | awk '
  {
    # Split the indices into an array
    split($NF, indices, ",")

    # Print the indices on separate lines
    for (i in indices)
      print indices[i]
  }
' > grouped_indices.txt

# Remove the temporary file
rm $temp_file
