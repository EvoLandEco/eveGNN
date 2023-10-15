#!/bin/bash

# File name
file=params.txt

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
sort $temp_file > grouped_indices.txt

# Remove the temporary file
rm $temp_file
