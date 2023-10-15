import pandas as pd
import sys


def find_unique_combinations(path):
    # Combine path with params.txt
    file_path = f'{path}/params.txt'

    # Read the data from params.txt
    df = pd.read_csv(file_path, sep="\s+", header=0)

    # Group the data by the specified columns
    grouped = df.groupby(['lambda', 'mu', 'beta_n', 'beta_phi'])

    # Initialize an empty list to collect the indices
    index_vectors = []

    # Iterate through each group and collect the indices
    for name, group in grouped:
        index_vector = group.index.tolist()
        index_vectors.append(index_vector)

    return index_vectors


if __name__ == '__main__':
    # Get the path argument from the command line
    if len(sys.argv) != 2:
        print("Usage: python script.py <path>")
        sys.exit(1)

    path = sys.argv[1]

    # Call the function
    combo_indices = find_unique_combinations(path)

    # Combine path with output.txt
    output_path = f'{path}/combination.txt'

    # Write the output to output.txt
    with open(output_path, 'w') as file:
        for index_vector in combo_indices:
            file.write(','.join(map(str, index_vector)) + '\n')
