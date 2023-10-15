import sys
import os
import pandas as pd
import pyreadr
import torch
import glob
import functools
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
    rds_files = glob.glob(os.path.join(path, '*.rds'))
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


def read_rds_to_pytorch(path, set_index, count):
    params_current = get_params(path, set_index)

    metric = params_current['metric']

    # Map metrics to category values
    metric_to_category = {'pd': 0, 'ed': 1, 'nnd': 2}

    # Check if the provided metric is valid
    if metric not in metric_to_category:
        raise ValueError(f"Unknown metric: {metric}. Expected one of: {', '.join(metric_to_category.keys())}")

    # Get the category value for the provided prefix
    category_value = torch.tensor([metric_to_category[metric]], dtype=torch.long)

    # List to hold the data from each .rds file
    data_list = []

    # Loop through the files for the specified prefix
    for i in range(1, count + 1):
        # Construct the file path using the specified prefix
        file_path = os.path.join(path, 'GNN', 'tree', f"tree_{i}.rds")

        # Read the .rds file
        result = pyreadr.read_r(file_path)

        # The result is a dictionary where keys are the name of objects and the values python dataframes
        # Since RDS can only contain one object, it will be the first item in the dictionary
        data = result[None]

        # Append the data to data_list
        data_list.append(data)

    length_list = []

    for i in range(1, count + 1):
        length_file_path = os.path.join(path, 'GNN', 'tree', "EL", f"EL_{i}.rds")
        length_result = pyreadr.read_r(length_file_path)
        length_data = length_result[None]
        length_list.append(length_data)

    # List to hold the Data objects
    pytorch_geometric_data_list = []

    for i in range(0, count):
        # Ensure the DataFrame is of integer type and convert to a tensor
        edge_index_tensor = torch.tensor(data_list[i].values, dtype=torch.long)

        # Make sure the edge_index tensor is of size [2, num_edges]
        edge_index_tensor = edge_index_tensor.t().contiguous()

        # Determine the number of nodes
        num_nodes = edge_index_tensor.max().item() + 1

        edge_length_tensor = torch.tensor(length_list[i].values, dtype=torch.float)

        # Create a Data object with the edge index, number of nodes, and category value
        data = Data(x=edge_length_tensor,
                    edge_index=edge_index_tensor,
                    num_nodes=num_nodes,
                    y=category_value)

        # Append the Data object to the list
        pytorch_geometric_data_list.append(data)

    return pytorch_geometric_data_list


def get_training_data(data_list):
    # Find the index at which to split the list
    split_index = int(len(data_list) * 0.9)
    # Use list slicing to get the first 90% of the data
    training_data = data_list[:split_index]
    return training_data


def get_testing_data(data_list):
    # Find the index at which to split the list
    split_index = int(len(data_list) * 0.9)
    # Use list slicing to get the last 10% of the data
    testing_data = data_list[split_index:]
    return testing_data


def main():
    # The base directory path is passed as the first argument
    name = sys.argv[1]

    print(f'Project: {name}')

    # The set_i folder names are passed as the remaining arguments
    set_paths = sys.argv[2:]

    params_list = []

    training_dataset_list = []
    testing_dataset_list = []

    # Iterate through each set_i folder name
    for set_index in set_paths:
        set_path = f'set_{set_index}'
        # Concatenate the base directory path with the set_i folder name
        full_dir = os.path.join(name, set_path)
        full_dir_tree = os.path.join(full_dir, 'GNN', 'tree')
        full_dir_el = os.path.join(full_dir, 'GNN', 'tree', 'EL')
        # Call read_rds_to_pytorch with the full directory path
        print(full_dir)  # The set_i folder names are passed as the remaining arguments
        params_current = get_params(name, set_index)
        print(params_current)
        params_list.append(params_current)

        # Check if the number of .rds files in the tree and el paths are equal
        rds_count = check_rds_files_count(full_dir_tree, full_dir_el)
        print(f'There are: {rds_count} trees in the set_{set_index} folder.')

    # Check if all the list elements have the same lambda, mu, beta_n, and beta_phi
    check_params(params_list)


if __name__ == '__main__':
    main()
