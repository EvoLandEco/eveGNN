import sys
import os
import pandas as pd
import pyreadr
import torch
import glob
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import InMemoryDataset, Data


def get_params(name, set_index):
    # Form the filename by concatenating the base directory with 'params.txt'
    file_path = os.path.join(name, 'params.txt')

    # Read the data from params.txt
    df = pd.read_csv(file_path, sep="\s+", header=0)

    # Convert set_index to integer
    set_index = int(set_index) - 1  # subtract 1 because iloc is 0-based

    # Print the specified row
    return df.iloc[set_index]


def check_params(params):
    # Check if all the list elements have the same lambda, mu, beta_n, and beta_phi
    first_row = params[0]
    for row in params[1:]:
        if not (first_row[['lambda', 'mu', 'beta_n', 'beta_phi']].equals(row[['lambda', 'mu', 'beta_n', 'beta_phi']])):
            print("Parameter mismatch found. Data is not consistent.")
            sys.exit()
    print("All rows have the same lambda, mu, beta_n, and beta_phi.")


def count_rds_files(path):
    # Get the list of .rds files in the specified path
    rds_files = glob.glob(os.path.join(path, '**/*.rds'), recursive=True)
    return len(rds_files)


def check_rds_files_count(tree_path, el_path):
    # Count the number of .rds files in both paths
    tree_count = count_rds_files(tree_path)
    el_count = count_rds_files(el_path)

    # Check if the counts are equal
    if tree_count == el_count:
        return tree_count
    else:
        raise ValueError("The number of .rds files in the two paths are not equal")


def main():
    # The base directory path is passed as the first argument
    name = sys.argv[1]

    print(f'Project: {name}')

    # The set_i folder names are passed as the remaining arguments
    set_paths = sys.argv[2:]

    params_list = []

    # Iterate through each set_i folder name
    for set_index in set_paths:
        set_path = f'set_{set_index}'
        # Concatenate the base directory path with the set_i folder name
        full_dir = os.path.join(name, set_path)
        full_dir_tree = os.path.join(full_dir, 'GNN', 'tree')
        full_dir_el = os.path.join(full_dir, 'GNN', 'EL')
        # Call read_rds_to_pytorch with the full directory path
        print(full_dir)  # The set_i folder names are passed as the remaining arguments
        params_current = get_params(name, set_index)
        print(params_current.transpose())
        params_list.append(params_current)

        # Check if the number of .rds files in the tree and el paths are equal
        rds_count = check_rds_files_count(full_dir_tree, full_dir_el)
        print(f'There are: {rds_count} trees in the set_{set_index} folder.')

    # Check if all the list elements have the same lambda, mu, beta_n, and beta_phi
    check_params(params_list)


if __name__ == '__main__':
    main()
