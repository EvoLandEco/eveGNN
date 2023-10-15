import sys
import os
import pandas as pd


def get_params(name, set_index):
    # Form the filename by concatenating the base directory with 'params.txt'
    filename = os.path.join(name, 'params.txt')

    # Read the table from the text file
    df = pd.read_csv(filename, delim_whitespace=True, header=None)

    # Convert set_index to integer
    set_index = int(set_index) - 1  # subtract 1 because iloc is 0-based

    # Print the specified row
    print(df.iloc[set_index])


def main():
    # The base directory path is passed as the first argument
    name = sys.argv[1]

    print(f'Project: {name}')

    # The set_i folder names are passed as the remaining arguments
    set_paths = sys.argv[2:]

    for set_index in set_paths:
        set_path = f'set_{set_index}'
        # Concatenate the base directory path with the set_i folder name
        full_dir = os.path.join(name, set_path)
        # Call read_rds_to_pytorch with the full directory path
        print(full_dir)  # The set_i folder names are passed as the remaining arguments
        get_params(name, set_index)


if __name__ == '__main__':
    main()
