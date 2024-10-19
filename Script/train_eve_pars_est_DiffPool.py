import sys
import os
import pandas as pd
import numpy as np
import pyreadr
import torch
import glob
import functools
import random
import torch_geometric.transforms as T
import torch.nn.functional as F
import yaml
from math import ceil
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool
from torch_geometric.nn import DenseSAGEConv as SAGEConv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence, pack_sequence

# Load the global parameters from the config file
global_params = None

with open("../Config/eve_train_diffpool.yaml", "r") as ymlfile:
    global_params = yaml.safe_load(ymlfile)

# Set global variables
epoch_number_gnn = global_params["epoch_number_gnn"]
epoch_number_dnn = global_params["epoch_number_dnn"]
epoch_number_lstm = global_params["epoch_number_lstm"]
diffpool_ratio = global_params["diffpool_ratio"]
dropout_ratio = global_params["dropout_ratio"]
learning_rate = global_params["learning_rate"]
train_batch_size = global_params["train_batch_size"]
test_batch_size = global_params["test_batch_size"]
gcn_layer1_hidden_channels = global_params["gcn_layer1_hidden_channels"]
gcn_layer2_hidden_channels = global_params["gcn_layer2_hidden_channels"]
gcn_layer3_hidden_channels = global_params["gcn_layer3_hidden_channels"]
dnn_hidden_channels = global_params["dnn_hidden_channels"]
dnn_output_channels = global_params["dnn_output_channels"]
dnn_depth = global_params["dnn_depth"]
lstm_hidden_channels = global_params["lstm_hidden_channels"]
lstm_output_channels = global_params["lstm_output_channels"]
lstm_depth = global_params["lstm_depth"]
lin_layer1_hidden_channels = global_params["lin_layer1_hidden_channels"]
lin_layer2_hidden_channels = global_params["lin_layer2_hidden_channels"]
n_predicted_values = global_params["n_predicted_values"]
n_classes = global_params["n_classes"]
batch_size_reduce_factor = global_params["batch_size_reduce_factor"]
max_nodes_limit = global_params["max_nodes_limit"]
normalize_edge_length = global_params["normalize_edge_length"]
normalize_graph_representation = global_params["normalize_graph_representation"]
huber_delta = global_params["huber_delta"]
global_pooling_method = global_params["global_pooling_method"]

alpha= global_params["alpha"]
beta = global_params["beta"]
metric_to_category = {'pd': 0, 'ed': 1, 'nnd': 2}

# Check if metric_to_category is a dictionary with string keys and integer values
assert isinstance(metric_to_category, dict), "metric_to_category should be a dictionary"
for key, value in metric_to_category.items():
    assert isinstance(key, str), "All keys in metric_to_category should be strings"
    assert isinstance(value, int), "All values in metric_to_category should be integers"

# Check if alpha and beta are positive floats
assert isinstance(alpha, float) and alpha > 0, "alpha should be a positive float"
assert isinstance(beta, float) and beta > 0, "beta should be a positive float"

# Check if alpha + beta = 1
assert alpha + beta == 1, "alpha + beta should equal 1"


def read_table(path):
    return pd.read_csv(path, sep="\s+", header=0)  # assuming the tables are tab-delimited


def check_same_across_rows(df):
    return df.apply(lambda x: x.nunique() == 1)


def count_rds_files(path):
    # Get the list of .rds files in the specified path
    rds_files = glob.glob(os.path.join(path, '*.rds'))
    return len(rds_files)


def check_rds_files_count(tree_path, el_path, bt_path):
    # Count the number of .rds files in all four paths
    tree_count = count_rds_files(tree_path)
    el_count = count_rds_files(el_path)
    bt_count = count_rds_files(bt_path)

    # Check if the counts are equal
    if tree_count == el_count == bt_count:
        return tree_count  # Assuming all counts are equal, return one of them
    else:
        raise ValueError("The number of .rds files in the four paths are not equal")


def list_subdirectories(path):
    try:
        # Ensure the given path exists and it's a directory
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} does not exist.")
        if not os.path.isdir(path):
            raise NotADirectoryError(f"The path {path} is not a directory.")

        # List all entries in the directory
        entries = os.listdir(path)

        # Filter out entries that are directories
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]

        return subdirectories

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_params_string(filename):
    # Function to extract parameters from the filename
    params = filename.split('_')[1:-1]  # Exclude the first and last elements
    return "_".join(params)


def get_params(filename):
    parts = filename.split('_')[1:-1]
    params = []
    for part in parts:
        try:
            # Try to convert to float
            params.append(float(part))
        except ValueError:
            # If it's not a float, convert using the metric_to_category mapping
            params.append(metric_to_category.get(part, part))  # default to the string itself if not found

    return params


def get_sort_key(filename):
    # Split the filename by underscores
    parts = filename.split('_')[1:-1]

    params = []
    for part in parts:
        try:
            # Try to convert to float
            params.append(float(part))
        except ValueError:
            # If it's not a float, convert using the metric_to_category mapping
            params.append(metric_to_category.get(part, part))  # default to the string itself if not found

    return tuple(params)


def sort_files(files):
    # Sort the files based on the parameters extracted by the get_sort_key function
    return sorted(files, key=get_sort_key)


def check_file_consistency(files_tree, files_el, files_bt):
    # Check if the four lists have the same length
    if not (len(files_tree) == len(files_el) == len(files_bt)):
        raise ValueError("Mismatched lengths among file lists.")

    # Define a function to extract parameters from filename
    def get_params_tuple(filename):
        parts = filename.split('_')[1:-1]  # Exclude first one and last three elements in split strings
        params = []
        for part in parts:
            try:
                # Try to convert to float
                params.append(float(part))
            except ValueError:
                # If it's not a float, convert using the metric_to_category mapping
                params.append(metric_to_category.get(part, part))  # default to the string itself if not found
        return tuple(params)

    # Check each pair of files for matching parameters
    for tree_file, el_file, bt_file in zip(files_tree, files_el, files_bt):
        tree_params = get_params_tuple(tree_file)
        el_params = get_params_tuple(el_file)
        bt_params = get_params_tuple(bt_file)

        if not (tree_params == el_params == bt_params):
            raise ValueError(f"Mismatched parameters among files: {tree_file}, {el_file}, {bt_file}")

    # If we get here, all checks passed
    print("File lists consistency check passed across tree, EL, and BT datasets.")


def check_params_consistency(params_tree_list, params_el_list, params_bt_list):
    # Check if all corresponding elements in the four lists are equal
    is_consistent = all(
        a == b == c for a, b, c in zip(params_tree_list, params_el_list, params_bt_list))

    if is_consistent:
        print("Parameters are consistent across the tree, EL and BT datasets.")
    else:
        raise ValueError("Mismatch in parameters between the tree, EL and BT datasets.")

    return is_consistent


def check_list_count(count, data_list, length_list, params_list, brts_list):
    # Get the number of elements in each list
    data_count = len(data_list)
    length_count = len(length_list)
    params_count = len(params_list)
    brts_count = len(brts_list)  # Calculate the count for the new brts_list

    # Check if the count matches the number of elements in each list
    if count != data_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, data_list has {data_count} elements.")

    if count != length_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, length_list has {length_count} elements.")

    if count != params_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, params_list has {params_count} elements.")

    if count != brts_count:  # Check for brts_list
        raise ValueError(f"Count mismatch: input argument count is {count}, brts_list has {brts_count} elements.")

    # If all checks pass, print a success message
    print("Count check passed")


def read_rds_to_pytorch(path, count, normalize=False):
    # List all files in the directory
    files_tree = [f for f in os.listdir(os.path.join(path, 'GNN', 'tree'))
                  if f.startswith('tree_') and f.endswith('.rds')]
    files_el = [f for f in os.listdir(os.path.join(path, 'GNN', 'tree', 'EL'))
                if f.startswith('EL_') and f.endswith('.rds')]
    files_bt = [f for f in os.listdir(os.path.join(path, 'GNN', 'tree', 'BT'))
                if f.startswith('BT_') and f.endswith('.rds')]  # Get the list of files in the new BT directory

    # Sort the files based on the parameters
    files_tree = sort_files(files_tree)
    files_el = sort_files(files_el)
    files_bt = sort_files(files_bt)  # Sort the files in the new BT directory

    # Check if the files are consistent
    check_file_consistency(files_tree, files_el, files_bt)

    # List to hold the data from each .rds file
    data_list = []
    params_tree_list = []

    # Loop through the files with the prefix 'tree_'
    for filename in files_tree:
        file_path = os.path.join(path, 'GNN', 'tree', filename)
        result = pyreadr.read_r(file_path)
        data = result[None]
        data_list.append(data)
        params_tree_list.append(get_params_string(filename))

    length_list = []
    params_el_list = []

    # Loop through the files with the prefix 'EL_'
    for filename in files_el:
        length_file_path = os.path.join(path, 'GNN', 'tree', 'EL', filename)
        length_result = pyreadr.read_r(length_file_path)
        length_data = length_result[None]
        length_list.append(length_data)
        params_el_list.append(get_params_string(filename))

    brts_list = []
    params_bt_list = []

    # Loop through the files with the prefix 'BT_'
    for filename in files_bt:
        brts_file_path = os.path.join(path, 'GNN', 'tree', 'BT', filename)
        brts_result = pyreadr.read_r(brts_file_path)
        brts_data = brts_result[None]
        brts_list.append(brts_data)
        params_bt_list.append(get_params_string(filename))

    check_params_consistency(params_tree_list, params_el_list, params_bt_list)

    params_list = []

    for filename in files_tree:
        params = get_params(filename)
        params_list.append(params)

    # # Normalize beta_n and beta_phi
    # for vector in params_list:
    #     vector[2] = vector[2] * beta_n_norm_factor
    #     vector[3] = vector[3] * beta_phi_norm_factor

    check_list_count(count, data_list, length_list, params_list, brts_list)

    # List to hold the Data objects
    pytorch_geometric_data_list = []

    for i in range(0, count):
        # Ensure the DataFrame is of integer type and convert to a tensor
        edge_index_tensor = torch.tensor(data_list[i].values, dtype=torch.long)

        # Make sure the edge_index tensor is of size [2, num_edges]
        edge_index_tensor = edge_index_tensor.t().contiguous()

        # Determine the number of nodes
        num_nodes = edge_index_tensor.max().item() + 1

        if normalize:
            # Normalize node features by dividing edge lengths by log transformed number of nodes
            norm_length_list = length_list[i] / np.log10(len(length_list[i]))
            edge_length_tensor = torch.tensor(norm_length_list.values, dtype=torch.float)
        else:
            edge_length_tensor = torch.tensor(length_list[i].values, dtype=torch.float)

        params_current = params_list[i]

        params_current_tensor = torch.tensor(params_current[0:(n_predicted_values)], dtype=torch.float)

        class_current_tensor = torch.tensor(params_current[n_predicted_values + 1], dtype=torch.long)

        brts_tensor = torch.tensor(brts_list[i].values, dtype=torch.float)

        brts_tensor = brts_tensor.squeeze(1)  # Remove the extra dimension

        brts_length = torch.tensor([len(brts_list[i].values)], dtype=torch.long)

        # Create a Data object with the edge index, number of nodes, and category value
        data = Data(x=edge_length_tensor,
                    edge_index=edge_index_tensor,
                    num_nodes=num_nodes,
                    y_re=params_current_tensor,
                    y_cl=class_current_tensor,
                    brts=brts_tensor,
                    brts_len=brts_length)

        # Append the Data object to the list
        pytorch_geometric_data_list.append(data)

    print("Finished reading data")

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

def shuffle_data(data_list):
    # Create a copy of the data list to shuffle
    shuffled_list = data_list.copy()
    # Shuffle the copied list in place
    random.shuffle(shuffled_list)
    return shuffled_list


def main():
    if len(sys.argv) != 4:
        print(f"Python Command Line Error. Usage: {sys.argv[0]} <name> <task_type> <gnn_depth>")
        sys.exit(1)

    name = sys.argv[1]
    task_type = sys.argv[2]
    gnn_depth = int(sys.argv[3])

    # Now you can use the variables name and set_i in your code
    print(f'Name: {name}, Task Type: {task_type}', f'GNN Depth: {gnn_depth}')
    print("Now on branch Multimodal-Stacking-Boosting")

    training_dataset_list = []
    testing_dataset_list = []

    # Concatenate the base directory path with the set_i folder name
    full_dir = os.path.join(name, task_type)
    full_dir_tree = os.path.join(full_dir, 'GNN', 'tree')
    full_dir_el = os.path.join(full_dir, 'GNN', 'tree', 'EL')
    full_dir_bt = os.path.join(full_dir, 'GNN', 'tree', 'BT')  # Add the full path for the new BT directory
    # Call read_rds_to_pytorch with the full directory path
    print(full_dir)
    # Check if the number of .rds files in the tree and el paths are equal
    rds_count = check_rds_files_count(full_dir_tree, full_dir_el, full_dir_bt)
    print(f'There are: {rds_count} trees in the {task_type} folder.')
    print(f"Now reading {task_type}...")
    # Read the .rds files into a list of PyTorch Geometric Data objects
    current_dataset = read_rds_to_pytorch(full_dir, rds_count, normalize_edge_length)
    # Shuffle the data
    current_dataset = shuffle_data(current_dataset)
    current_training_data = get_training_data(current_dataset)
    current_testing_data = get_testing_data(current_dataset)
    training_dataset_list.append(current_training_data)
    testing_dataset_list.append(current_testing_data)

    validation_dataset_list = []
    val_dir = ""

    train_batch_size_adjusted = None
    test_batch_size_adjusted = None

    if task_type == "EVE_FREE_TES":
        val_dir = os.path.join(name, "EVE_VAL_TES")
        train_batch_size_adjusted = int(train_batch_size)
        test_batch_size_adjusted = int(test_batch_size)
    elif task_type == "EVE_FREE_TAS":
        val_dir = os.path.join(name, "EVE_VAL_TAS")
        train_batch_size_adjusted = int(ceil(train_batch_size * batch_size_reduce_factor))
        test_batch_size_adjusted = int(ceil(test_batch_size * batch_size_reduce_factor))
    else:
        raise ValueError("Invalid task type.")

    full_val_dir_tree = os.path.join(val_dir, 'GNN', 'tree')
    full_val_dir_el = os.path.join(val_dir, 'GNN', 'tree', 'EL')
    full_val_dir_bt = os.path.join(val_dir, 'GNN', 'tree', 'BT')
    val_rds_count = check_rds_files_count(full_val_dir_tree, full_val_dir_el, full_val_dir_bt)
    print(f'There are: {val_rds_count} trees in the validation folder.')
    print(f"Now reading validation data...")
    # current_val_dataset = read_rds_to_pytorch(val_dir, val_rds_count, normalize_edge_length)
    current_val_dataset = []
    validation_dataset_list.append(current_val_dataset)

    sum_training_data = functools.reduce(lambda x, y: x + y, training_dataset_list)
    sum_testing_data = functools.reduce(lambda x, y: x + y, testing_dataset_list)
    sum_validation_data = functools.reduce(lambda x, y: x + y, validation_dataset_list)

    # Filtering out trees with only 3 nodes
    # They might cause problems with ToDense
    filtered_training_data = [data for data in sum_training_data if data.edge_index.shape != torch.Size([2, 2])]
    filtered_testing_data = [data for data in sum_testing_data if data.edge_index.shape != torch.Size([2, 2])]
    filtered_validation_data = [data for data in sum_validation_data if data.edge_index.shape != torch.Size([2, 2])]

    # Filtering out trees with more than 3000 nodes
    filtered_training_data = [data for data in filtered_training_data if data.num_nodes <= max_nodes_limit]
    filtered_testing_data = [data for data in filtered_testing_data if data.num_nodes <= max_nodes_limit]
    filtered_validation_data = [data for data in filtered_validation_data if data.num_nodes <= max_nodes_limit]

    # Get the maximum number of nodes, for padding the matrices of the graphs
    max_nodes_train = max([data.num_nodes for data in filtered_training_data])
    max_nodes_test = max([data.num_nodes for data in filtered_testing_data])

    # Get the maximum number of nodes in the validation set
    # Check if the validation set is empty
    max_nodes_val = 0
    if len(filtered_validation_data) == 0:
        max_nodes_val = 0
    else:
        max_nodes_val = max([data.num_nodes for data in filtered_validation_data])
    max_nodes = max(max_nodes_train, max_nodes_test, max_nodes_val)
    print(f"Max nodes: {max_nodes} for {task_type}")

    # Similarly, get the maximum lengths of brts sequences, for padding
    max_brts_len_train = max([len(data.brts) for data in filtered_training_data])
    max_brts_len_test = max([len(data.brts) for data in filtered_testing_data])

    # Get the maximum length of brts sequences in the validation set
    # Check if the validation set is empty
    max_brts_len_val = 0
    if len(filtered_validation_data) == 0:
        max_brts_len_val = 0
    else:
        max_brts_len_val = max([len(data.brts) for data in filtered_validation_data])
    max_brts_len = max(max_brts_len_train, max_brts_len_test, max_brts_len_val)
    print(f"Max brts length: {max_brts_len} for {task_type}")

    class TreeData(InMemoryDataset):
        def __init__(self, root, data_list, transform=None, pre_transform=None):
            # Calculate the maximum length of brts across all graphs
            max_length = max_brts_len

            # Pad the brts attribute for each graph
            for data in data_list:
                pad_size = max_length - len(data.brts)
                data.brts = torch.cat([data.brts, data.brts.new_full((pad_size,), fill_value=0)], dim=0)

            super(TreeData, self).__init__(root, transform, pre_transform)
            self.data, self.slices = self.collate(data_list)

        def _download(self):
            pass  # No download required

        def _process(self):
            pass  # No processing required

    training_dataset = TreeData(root=None, data_list=filtered_training_data, transform=T.ToDense(max_nodes))
    testing_dataset = TreeData(root=None, data_list=filtered_testing_data, transform=T.ToDense(max_nodes))

    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels,
                     normalize=False):
            super(GNN, self).__init__()

            self.convs = torch.nn.ModuleList()
            self.bns = torch.nn.ModuleList()

            for i in range(gnn_depth):
                first_index = 0
                last_index = gnn_depth - 1

                if gnn_depth == 1:
                    self.convs.append(SAGEConv(in_channels, out_channels, normalize))
                    self.bns.append(torch.nn.BatchNorm1d(out_channels))
                else:
                    if i == first_index:
                        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize))
                        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
                    elif i == last_index:
                        self.convs.append(SAGEConv(hidden_channels, out_channels, normalize))
                        self.bns.append(torch.nn.BatchNorm1d(out_channels))
                    else:
                        self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize))
                        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        def forward(self, x, adj, mask=None):
            outputs = []  # Initialize a list to store outputs at each step
            for step in range(len(self.convs)):
                x = F.gelu(self.convs[step](x, adj, mask))
                x = torch.permute(x, (0, 2, 1))
                x = self.bns[step](x)
                x = torch.permute(x, (0, 2, 1))
                outputs.append(x)  # Store the current x

            x_concatenated = torch.cat(outputs, dim=-1)
            return x_concatenated

    class LSTM(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, lstm_depth=1, dropout_rate=0.5, num_classes=n_classes):
            super(LSTM, self).__init__()

            self.lstm = torch.nn.LSTM(input_size=in_channels, hidden_size=hidden_channels, num_layers=lstm_depth,
                                      batch_first=True, dropout=dropout_rate if lstm_depth > 1 else 0)
            self.dropout = torch.nn.Dropout(dropout_rate)

            # Regression output layers
            self.lin_reg = torch.nn.Linear(hidden_channels, out_channels)
            self.readout_reg = torch.nn.Linear(out_channels, n_predicted_values)  # Assuming regression output is a single value

            # Classification output layers
            self.lin_class = torch.nn.Linear(hidden_channels, out_channels)
            self.readout_class = torch.nn.Linear(out_channels, num_classes)  # Number of classes for classification

        def forward(self, x):
            out, (h_n, c_n) = self.lstm(x)

            # Get the final hidden state from the last LSTM layer
            final_hidden_state = h_n[-1, :, :]

            # Apply activation function to the final hidden state
            x = F.gelu(final_hidden_state)

            # Regression branch
            reg_out = self.dropout(x)
            reg_out = self.lin_reg(reg_out)
            reg_out = F.gelu(reg_out)
            reg_out = self.dropout(reg_out)
            reg_out = self.readout_reg(reg_out)

            # Classification branch
            class_out = self.dropout(x)
            class_out = self.lin_class(class_out)
            class_out = F.gelu(class_out)
            class_out = self.dropout(class_out)
            class_out = self.readout_class(class_out)
            class_out = torch.softmax(class_out, dim=-1)

            return reg_out, class_out


    class DiffPool(torch.nn.Module):
        def __init__(self, verbose=False):
            super(DiffPool, self).__init__()
            self.verbose = verbose
            if self.verbose:
                print("Initializing DiffPool model...")

            # Read in augmented data
            self.graph_sizes = torch.tensor([], dtype=torch.long)

            if self.verbose:
                print("Graph sizes loaded...")

            # DiffPool Layer 1
            num_nodes1 = ceil(diffpool_ratio * max_nodes)
            self.gnn1_pool = GNN(training_dataset.num_node_features, gcn_layer1_hidden_channels, num_nodes1)
            self.gnn1_embed = GNN(training_dataset.num_node_features, gcn_layer1_hidden_channels,
                                  gcn_layer2_hidden_channels)

            # DiffPool Layer 2
            num_nodes2 = ceil(diffpool_ratio * num_nodes1)
            gnn1_out_channels = gcn_layer1_hidden_channels * (gnn_depth - 1) + gcn_layer2_hidden_channels
            self.gnn2_pool = GNN(gnn1_out_channels, gcn_layer2_hidden_channels, num_nodes2)
            self.gnn2_embed = GNN(gnn1_out_channels, gcn_layer2_hidden_channels, gcn_layer3_hidden_channels)

            # DiffPool Layer 3
            gnn2_out_channels = gcn_layer2_hidden_channels * (gnn_depth - 1) + gcn_layer3_hidden_channels
            self.gnn3_embed = GNN(gnn2_out_channels, gcn_layer3_hidden_channels, lin_layer1_hidden_channels)
            gnn3_out_channels = gcn_layer3_hidden_channels * (gnn_depth - 1) + lin_layer1_hidden_channels

            # Final Readout Layers
            self.lin1 = torch.nn.Linear(gnn3_out_channels, lin_layer2_hidden_channels)
            self.lin2 = torch.nn.Linear(lin_layer2_hidden_channels, n_predicted_values)

            self.lin_class1 = torch.nn.Linear(gnn3_out_channels, lin_layer2_hidden_channels)
            self.lin_class2 = torch.nn.Linear(lin_layer2_hidden_channels, n_classes)

        def forward(self, x, adj, mask=None):
            # Forward pass through the DiffPool layer 1
            s = self.gnn1_pool(x, adj, mask)
            x = self.gnn1_embed(x, adj, mask)
            x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

            if self.verbose:
                print("DiffPool Layer 1 Completed...")

            # Forward pass through the DiffPool layer 2
            s = self.gnn2_pool(x, adj)
            x = self.gnn2_embed(x, adj)
            x, adj, l2, e2 = dense_diff_pool(x, adj, s)

            if self.verbose:
                print("DiffPool Layer 2 Completed...")

            # Forward pass through the DiffPool layer 3
            x = self.gnn3_embed(x, adj)

            if self.verbose:
                print("DiffPool Layer 3 Completed...")

            # Global pooling after the final GNN layer
            if global_pooling_method == "mean":
                x = x.mean(dim=1)
            elif global_pooling_method == "max":
                x = x.max(dim=1).values
            elif global_pooling_method == "sum":
                x = x.sum(dim=1)
            else:
                raise ValueError("Invalid global pooling method.")

            if self.verbose:
                print("Global Pooling Completed...")

            # Regression branch
            reg_out = F.dropout(x, p=dropout_ratio, training=self.training)
            reg_out = self.lin1(reg_out)
            reg_out = F.gelu(reg_out)
            reg_out = F.dropout(reg_out, p=dropout_ratio, training=self.training)
            reg_out = self.lin2(reg_out)

            # Classification branch
            class_out = F.dropout(x, p=dropout_ratio, training=self.training)
            class_out = self.lin_class1(class_out)
            class_out = F.gelu(class_out)
            class_out = F.dropout(class_out, p=dropout_ratio, training=self.training)
            class_out = self.lin_class2(class_out)
            class_out = torch.softmax(class_out, dim=-1)

            return reg_out, class_out, l1 + l2, e1 + e2

    def combined_loss(output_regression, target_regression, output_classification, target_classification, link_loss, entropy_loss):
        loss_regression = criterion(output_regression, target_regression)
        loss_classification = F.cross_entropy(output_classification, target_classification)
        loss_all = alpha * loss_regression + beta * loss_classification + link_loss + entropy_loss

        return loss_all, loss_regression, loss_classification

    def train_gnn():
        model_gnn.train()

        loss_all = 0  # Keep track of the loss
        loss_reg = 0  # Keep track of the regression loss
        loss_cls = 0  # Keep track of the classification loss

        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()
            out_re, out_cl, l, e = model_gnn(data.x, data.adj, data.mask)
            target_re = data.y_re.view(data.num_nodes.__len__(), n_predicted_values).to(device)
            target_cl = data.y_cl.view(-1).to(device)
            assert out_re.device == target_re.device == out_cl.device == target_cl.device, \
                "Error: Device mismatch between output and target tensors."
            loss, loss_reg, loss_cls = combined_loss(out_re, target_re, out_cl, target_cl, l, e)
            loss.backward()
            loss_all += loss.item() * data.num_nodes.__len__()
            loss_reg += loss_reg.item() * data.num_nodes.__len__()
            loss_cls += loss_cls.item() * data.num_nodes.__len__()
            optimizer.step()

        out_loss_all = loss_all / len(train_loader.dataset)
        out_loss_reg = loss_reg / len(train_loader.dataset)
        out_loss_cls = loss_cls / len(train_loader.dataset)

        return out_loss_all, out_loss_reg, out_loss_cls

    @torch.no_grad()
    def test_diff_gnn(loader):
        model_gnn.eval()

        diffs_all = torch.tensor([], dtype=torch.float, device=device)
        outputs_all = torch.tensor([], dtype=torch.float, device=device)  # To store all outputs
        y_all = torch.tensor([], dtype=torch.float, device=device)  # To store all y
        nodes_all = torch.tensor([], dtype=torch.long, device=device)

        for data in loader:
            data.to(device)
            out_re, _, _, _ = model_gnn(data.x, data.adj, data.mask)
            diffs = torch.abs(out_re - data.y_re.view(data.num_nodes.__len__(), n_predicted_values))
            diffs_all = torch.cat((diffs_all, diffs), dim=0)
            outputs_all = torch.cat((outputs_all, out_re), dim=0)
            y_all = torch.cat((y_all, data.y_re.view(data.num_nodes.__len__(), n_predicted_values)), dim=0)
            nodes_all = torch.cat((nodes_all, data.num_nodes), dim=0)

        print(f"diffs_all length: {len(diffs_all)}; test_loader.dataset length: {len(test_loader.dataset)}; Equal: {len(diffs_all) == len(test_loader.dataset)}")
        mean_diffs = torch.sum(diffs_all, dim=0) / len(test_loader.dataset)
        return mean_diffs.cpu().detach().numpy(), diffs_all.cpu().detach().numpy(), outputs_all.cpu().detach().numpy(), y_all.cpu().detach().numpy(), nodes_all.cpu().detach().numpy()

    @torch.no_grad()
    def compute_test_loss_gnn():
        model_gnn.eval()  # Set the model to evaluation mode

        loss_all = 0  # Keep track of the loss
        loss_reg = 0  # Keep track of the regression loss
        loss_cls = 0  # Keep track of the classification loss

        for data in test_loader:
            data.to(device)
            graph_sizes = data.num_nodes
            out_re, out_cl, l, e = model_gnn(data.x, data.adj, data.mask)
            target_re = data.y_re.view(data.num_nodes.__len__(), n_predicted_values).to(device)
            target_cl = data.y_cl.view(-1).to(device)
            loss, loss_reg, loss_cls = combined_loss(out_re, target_re, out_cl, target_cl, l, e)
            loss_all += loss.item() * data.num_nodes.__len__()
            loss_reg += loss_reg.item() * data.num_nodes.__len__()
            loss_cls += loss_cls.item() * data.num_nodes.__len__()

        out_loss_all = loss_all / len(train_loader.dataset)
        out_loss_reg = loss_reg / len(train_loader.dataset)
        out_loss_cls = loss_cls / len(train_loader.dataset)

        return out_loss_all, out_loss_reg, out_loss_cls

    @torch.no_grad()
    def test_accu_gnn(loader, num_classes):
        model_gnn.eval()
        correct = 0
        total_samples = len(loader.dataset)

        # Initialize counters for per-class correct predictions and total samples per class
        correct_per_class = torch.zeros(num_classes, dtype=torch.long).to(device)
        total_per_class = torch.zeros(num_classes, dtype=torch.long).to(device)
        outputs_all = torch.tensor([], dtype=torch.float, device=device)  # To store all outputs
        y_all = torch.tensor([], dtype=torch.float, device=device)  # To store all y

        for data in loader:
            data = data.to(device)
            _, out_cl, _, _ = model_gnn(data.x, data.adj, data.mask)

            outputs_all = torch.cat((outputs_all, out_cl), dim=0)
            y_all = torch.cat((y_all, data.y_cl.view(-1)), dim=0)

            # Get the predicted class by taking the class with the max score
            pred = out_cl.max(dim=1)[1]

            # Update the total correct predictions
            correct += pred.eq(data.y_cl.view(-1)).sum().item()

            # Update the correct count and total count per class
            for i in range(num_classes):
                mask = data.y_cl.view(-1) == i  # Find all samples belonging to class i
                correct_per_class[i] += pred[mask].eq(data.y_cl.view(-1)[mask]).sum().item()
                total_per_class[i] += mask.sum().item()

        # Calculate overall accuracy
        overall_accuracy = correct / total_samples

        # Calculate per-class accuracy
        class_accuracy = correct_per_class.float() / total_per_class.float()

        # Handle cases where a class might not be present in the batch
        class_accuracy[total_per_class == 0] = float('nan')  # Assign NaN to avoid division by zero

        return overall_accuracy, class_accuracy, outputs_all.cpu().detach().numpy(), y_all.cpu().detach().numpy()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training using {device}")

    model_gnn = DiffPool()
    model_gnn = model_gnn.to(device)
    optimizer = torch.optim.AdamW(model_gnn.parameters(), lr=learning_rate)
    criterion = torch.nn.HuberLoss(delta=huber_delta).to(device)

    def shape_check(dataset, max_nodes):
        incorrect_shapes = []  # List to store indices of data elements with incorrect shapes
        for i in range(len(dataset)):
            data = dataset[i]
            # Check the shapes of data.x, data.adj, and data.mask
            if data.x.shape != torch.Size([max_nodes, 3]) or \
                    data.y_re.shape != torch.Size([n_predicted_values]) or \
                    data.y_cl.shape != torch.Size([1]) or \
                    data.adj.shape != torch.Size([max_nodes, max_nodes]) or \
                    data.mask.shape != torch.Size([max_nodes]):
                incorrect_shapes.append(i)  # Add index to the list if any shape is incorrect

        # Print the indices of incorrect data elements or a message if all shapes are correct
        if incorrect_shapes:
            print(f"Incorrect shapes found at indices: {incorrect_shapes}")
        else:
            print("No incorrect shapes found.")

    # Check the shapes of the training and testing datasets
    # Be aware that ToDense will pad the data with zeros to the max_nodes value
    # However, ToDense may create malformed data.y when the number of nodes is 3 (2 tips)
    shape_check(training_dataset, max_nodes)
    shape_check(testing_dataset, max_nodes)

    train_loader = DenseDataLoader(training_dataset, batch_size=train_batch_size_adjusted, shuffle=False)
    test_loader = DenseDataLoader(testing_dataset, batch_size=test_batch_size_adjusted, shuffle=False)
    print(f"Training dataset length: {len(train_loader.dataset)}")
    print(f"Testing dataset length: {len(test_loader.dataset)}")
    print(train_loader.dataset.transform)
    print(test_loader.dataset.transform)

    print(model_gnn)

    test_mean_diffs_history = []
    train_loss_all_history = []
    train_loss_regression_history = []
    train_loss_classification_history = []
    test_loss_all_history = []
    test_loss_regression_history = []
    test_loss_classification_history = []
    test_overall_accuracy_history = []
    test_per_class_accuracy_history = []
    final_test_diffs = []
    final_test_predictions = []
    final_test_y = []
    final_test_nodes = []
    final_test_label_pred = []
    final_test_label_true = []
    final_overall_accuracy = []
    final_per_class_accuracy = []

    # Set up the early stopper
    # early_stopper = EarlyStopper(patience=3, min_delta=0.05)
    actual_epoch_gnn = 0
    # The losses are summed over each data point in the batch, thus we should normalize the losses accordingly
    train_test_ratio = len(train_loader.dataset) / len(test_loader.dataset)

    print("Now training GNN model...")

    for epoch in range(1, epoch_number_gnn):
        actual_epoch_gnn = epoch
        train_loss_all, train_loss_reg, train_loss_cls = train_gnn()
        test_loss_all, test_loss_reg, test_loss_cls = compute_test_loss_gnn()
        test_loss_all = test_loss_all * train_test_ratio
        test_loss_reg = test_loss_reg * train_test_ratio
        test_loss_cls = test_loss_cls * train_test_ratio
        test_mean_diffs, test_diffs_all, test_predictions, test_y, test_nodes_all = test_diff_gnn(test_loader)
        test_accuracy_all, test_accuracy_class, test_label_pred, test_label_true = test_accu_gnn(test_loader, n_classes)
        print(f'Epoch: {epoch:03d}, Train Regression Loss: {train_loss_reg:.4f}, Train Classification Loss: {train_loss_cls:.4f}, Train Combined Loss: {train_loss_all:.4f}')
        print(f'Epoch: {epoch:03d}, Test Regression Loss: {test_loss_reg:.4f}, Test Classification Loss: {test_loss_cls:.4f}, Test Combined Loss: {test_loss_all:.4f}')
        print(f'Epoch: {epoch:03d}, Overall Classification Accuracy: {test_accuracy_all:.4f}')
        # Convert the tensors to arrayss
        train_loss_reg = train_loss_reg.cpu().detach().numpy()
        train_loss_cls = train_loss_cls.cpu().detach().numpy()
        test_loss_reg = test_loss_reg.cpu().detach().numpy()
        test_loss_cls = test_loss_cls.cpu().detach().numpy()
        test_accuracy_class = test_accuracy_class.cpu().detach().numpy()
        print(f'Epoch: {epoch:03d}, Class 0 Accuracy: {test_accuracy_class[0]:.4f}, Class 1 Accuracy: {test_accuracy_class[1]:.4f}, Class 2 Accuracy: {test_accuracy_class[2]:.4f}')
        # Record the values
        test_mean_diffs_history.append(test_mean_diffs)
        train_loss_all_history.append(train_loss_all)
        train_loss_regression_history.append(train_loss_reg)
        train_loss_classification_history.append(train_loss_cls)
        test_loss_all_history.append(test_loss_all)
        test_loss_regression_history.append(test_loss_reg)
        test_loss_classification_history.append(test_loss_cls)
        test_overall_accuracy_history.append(test_accuracy_all)
        test_per_class_accuracy_history.append(test_accuracy_class)
        final_test_diffs = test_diffs_all
        final_test_predictions = test_predictions
        final_test_y = test_y
        final_test_nodes = test_nodes_all
        final_test_label_pred = test_label_pred
        final_test_label_true = test_label_true
        final_overall_accuracy = test_accuracy_all
        final_per_class_accuracy = test_accuracy_class
        print(f"Final test diffs length: {len(final_test_diffs)}")
        print(f"Final predictions length: {len(final_test_predictions)}")
        print(f"Final y length: {len(final_test_y)}")
        print(f"Final nodes length: {len(final_test_nodes)}")

    # Save the model
    print("Saving GNN model...")
    # Create the directory if it doesn't exist
    if not os.path.exists(os.path.join(name, task_type, "STBO")):
        os.makedirs(os.path.join(name, task_type, "STBO"))
    torch.save(model_gnn.state_dict(),
               os.path.join(name, task_type, "STBO", f"{task_type}_model_diffpool_{gnn_depth}_gnn.pt"))

    # Safe append in case of missing values
    def safe_append(array, index, default_value=0):
        try:
            return array[index]
        except IndexError:
            return default_value

    # After the loop, create a dictionary to hold the data
    data_dict = {"lambda_diff": [], "mu_diff": [], "beta_n_diff": [], "beta_phi_diff": [], "gamma_n_diff": [], "gamma_phi_diff": []}

    # Iterate through test_mean_diffs_history
    for array in test_mean_diffs_history:
        # Safely append each value, filling with 0 if missing
        data_dict["lambda_diff"].append(safe_append(array, 0))
        data_dict["mu_diff"].append(safe_append(array, 1))
        data_dict["beta_n_diff"].append(safe_append(array, 2))
        data_dict["beta_phi_diff"].append(safe_append(array, 3))
        data_dict["gamma_n_diff"].append(safe_append(array, 4))
        data_dict["gamma_phi_diff"].append(safe_append(array, 5))

    # Ensure the length of other lists matches actual_epoch_gnn, filling missing values with 0
    actual_epoch_gnn = len(train_loss_all_history)  # Ensure epoch count matches available data

    data_dict["Epoch"] = list(range(1, actual_epoch_gnn + 1))
    data_dict["Train_Loss_ALL"] = train_loss_all_history[:actual_epoch_gnn] + [0] * (actual_epoch_gnn - len(train_loss_all_history))
    data_dict["Train_Loss_Regression"] = train_loss_regression_history[:actual_epoch_gnn] + [0] * (actual_epoch_gnn - len(train_loss_regression_history))
    data_dict["Train_Loss_Classification"] = train_loss_classification_history[:actual_epoch_gnn] + [0] * (actual_epoch_gnn - len(train_loss_classification_history))
    data_dict["Test_Loss_ALL"] = test_loss_all_history[:actual_epoch_gnn] + [0] * (actual_epoch_gnn - len(test_loss_all_history))
    data_dict["Test_Loss_Regression"] = test_loss_regression_history[:actual_epoch_gnn] + [0] * (actual_epoch_gnn - len(test_loss_regression_history))
    data_dict["Test_Loss_Classification"] = test_loss_classification_history[:actual_epoch_gnn] + [0] * (actual_epoch_gnn - len(test_loss_classification_history))

    # Convert the dictionary to a pandas DataFrame
    model_performance = pd.DataFrame(data_dict)

    # For final_test_diffs, final_test_predictions, final_test_y, final_test_nodes, final_test_label_pred, and final_test_label_true
    # Use fillna to ensure missing values are replaced with 0 in DataFrames

    # Define a function to safely create DataFrames, checking for missing columns and filling with zeros
    def safe_create_dataframe(data, columns):
        # Convert data to a DataFrame, handling case where data might have fewer columns than expected
        df = pd.DataFrame(data)

        # If there are fewer columns in the data than expected, fill missing columns with zeros
        if df.shape[1] < len(columns):
            # Add missing columns with zeros
            for col in range(len(columns) - df.shape[1]):
                df[f"missing_col_{col}"] = 0

        # Now set the correct column names, making sure the DataFrame has the correct structure
        df.columns = columns[:df.shape[1]]

        # Check for any missing expected columns (by name) and fill them with zeros
        for col in columns:
            if col not in df.columns:
                df[col] = 0  # Add the missing column and fill with zeros

        return df

    # Using the safe_create_dataframe function for each DataFrame
    final_differences = safe_create_dataframe(final_test_diffs, ["lambda_diff", "mu_diff", "beta_n_diff", "beta_phi_diff", "gamma_n_diff", "gamma_phi_diff"])
    final_predictions = safe_create_dataframe(final_test_predictions, ["lambda_pred", "mu_pred", "beta_n_pred", "beta_phi_pred", "gamma_n_pred", "gamma_phi_pred"])
    final_y = safe_create_dataframe(final_test_y, ["lambda", "mu", "beta_n", "beta_phi", "gamma_n", "gamma_phi"])
    final_nodes = safe_create_dataframe(final_test_nodes, ["num_nodes"])
    final_label_prob = safe_create_dataframe(final_test_label_pred, ["pd_prob", "ed_prob", "nnd_prob"])
    final_label_true = safe_create_dataframe(final_test_label_true, ["true_class"])

    # Column-wise combine all final DataFrames
    final_result = pd.concat([final_differences, final_predictions, final_y, final_nodes, final_label_prob, final_label_true], axis=1)

    print("Final differences:")
    print(abs(final_differences[["lambda_diff", "mu_diff", "beta_n_diff", "beta_phi_diff", "gamma_n_diff", "gamma_phi_diff"]]).mean())
    print("Final overall accuracy:", final_overall_accuracy)
    print("Final per-class accuracy:", final_per_class_accuracy)

    # Workaround to get rid of the dtype incompatible issue
    model_performance = model_performance.astype(object)
    final_result = final_result.astype(object)

    # Save the data to a file using pyreadr
    pyreadr.write_rds(os.path.join(name, task_type, "STBO", f"{task_type}_diffpool_{gnn_depth}_gnn.rds"), model_performance)
    pyreadr.write_rds(os.path.join(name, task_type, "STBO", f"{task_type}_final_diffpool_{gnn_depth}_gnn.rds"), final_result)


    # Now the functions and logics for lstm
    def combined_loss_lstm(output_regression, target_regression, output_classification, target_classification):
        loss_regression = criterion(output_regression, target_regression)
        loss_classification = F.cross_entropy(output_classification, target_classification)
        loss_all = alpha * loss_regression + beta * loss_classification

        return loss_all, loss_regression, loss_classification

    def train_lstm():
        model_lstm.train()

        loss_all = 0  # Keep track of the loss
        loss_reg = 0  # Keep track of the regression loss
        loss_cls = 0  # Keep track of the classification loss

        for data in train_loader:
            lengths_brts = torch.sum(data.brts != 0, dim=1).cpu().tolist()
            brts_cpu = data.brts.cpu()
            brts_cpu = brts_cpu.unsqueeze(-1)
            packed_brts = pack_padded_sequence(brts_cpu, lengths_brts, batch_first=True, enforce_sorted=False).to(device)
            data.to(device)
            optimizer_lstm.zero_grad()
            out_re, out_cl = model_lstm(packed_brts)
            target_re = data.y_re.view(data.num_nodes.__len__(), n_predicted_values).to(device)
            target_cl = data.y_cl.view(-1).to(device)
            assert out_re.device == target_re.device == out_cl.device == target_cl.device, \
                "Error: Device mismatch between output and target tensors."
            loss, loss_reg, loss_cls = combined_loss_lstm(out_re, target_re, out_cl, target_cl)
            loss.backward()
            loss_all += loss.item() * data.num_nodes.__len__()
            loss_reg += loss_reg.item() * data.num_nodes.__len__()
            loss_cls += loss_cls.item() * data.num_nodes.__len__()
            optimizer_lstm.step()

        out_loss_all = loss_all / len(train_loader.dataset)
        out_loss_reg = loss_reg / len(train_loader.dataset)
        out_loss_cls = loss_cls / len(train_loader.dataset)

        return out_loss_all, out_loss_reg, out_loss_cls

    @torch.no_grad()
    def test_diff_lstm(loader):
        model_lstm.eval()

        diffs_all = torch.tensor([], dtype=torch.float, device=device)
        outputs_all = torch.tensor([], dtype=torch.float, device=device)  # To store all outputs
        y_all = torch.tensor([], dtype=torch.float, device=device)  # To store all y
        nodes_all = torch.tensor([], dtype=torch.long, device=device)

        for data in loader:
            lengths_brts = torch.sum(data.brts != 0, dim=1).cpu().tolist()
            brts_cpu = data.brts.cpu()
            brts_cpu = brts_cpu.unsqueeze(-1)
            packed_brts = pack_padded_sequence(brts_cpu, lengths_brts, batch_first=True, enforce_sorted=False).to(device)
            data.to(device)
            out_re, _ = model_lstm(packed_brts)
            diffs = torch.abs(out_re - data.y_re.view(data.num_nodes.__len__(), n_predicted_values))
            diffs_all = torch.cat((diffs_all, diffs), dim=0)
            outputs_all = torch.cat((outputs_all, out_re), dim=0)
            y_all = torch.cat((y_all, data.y_re.view(data.num_nodes.__len__(), n_predicted_values)), dim=0)
            nodes_all = torch.cat((nodes_all, data.num_nodes), dim=0)

        print(f"diffs_all length: {len(diffs_all)}; test_loader.dataset length: {len(test_loader.dataset)}; Equal: {len(diffs_all) == len(test_loader.dataset)}")
        mean_diffs = torch.sum(diffs_all, dim=0) / len(test_loader.dataset)
        return mean_diffs.cpu().detach().numpy(), diffs_all.cpu().detach().numpy(), outputs_all.cpu().detach().numpy(), y_all.cpu().detach().numpy(), nodes_all.cpu().detach().numpy()

    @torch.no_grad()
    def compute_test_loss_lstm():
        model_lstm.eval()  # Set the model to evaluation mode

        loss_all = 0  # Keep track of the loss
        loss_reg = 0  # Keep track of the regression loss
        loss_cls = 0  # Keep track of the classification loss

        for data in test_loader:
            lengths_brts = torch.sum(data.brts != 0, dim=1).cpu().tolist()
            brts_cpu = data.brts.cpu()
            brts_cpu = brts_cpu.unsqueeze(-1)
            packed_brts = pack_padded_sequence(brts_cpu, lengths_brts, batch_first=True, enforce_sorted=False).to(device)
            data.to(device)
            graph_sizes = data.num_nodes
            out_re, out_cl = model_lstm(packed_brts)
            target_re = data.y_re.view(data.num_nodes.__len__(), n_predicted_values).to(device)
            target_cl = data.y_cl.view(-1).to(device)
            loss, loss_reg, loss_cls = combined_loss_lstm(out_re, target_re, out_cl, target_cl)
            loss_all += loss.item() * data.num_nodes.__len__()
            loss_reg += loss_reg.item() * data.num_nodes.__len__()
            loss_cls += loss_cls.item() * data.num_nodes.__len__()

        out_loss_all = loss_all / len(train_loader.dataset)
        out_loss_reg = loss_reg / len(train_loader.dataset)
        out_loss_cls = loss_cls / len(train_loader.dataset)

        return out_loss_all, out_loss_reg, out_loss_cls

    @torch.no_grad()
    def test_accu_lstm(loader, num_classes):
        model_lstm.eval()
        correct = 0
        total_samples = len(loader.dataset)

        # Initialize counters for per-class correct predictions and total samples per class
        correct_per_class = torch.zeros(num_classes, dtype=torch.long).to(device)
        total_per_class = torch.zeros(num_classes, dtype=torch.long).to(device)
        outputs_all = torch.tensor([], dtype=torch.float, device=device)  # To store all outputs
        y_all = torch.tensor([], dtype=torch.float, device=device)  # To store all y

        for data in loader:
            lengths_brts = torch.sum(data.brts != 0, dim=1).cpu().tolist()
            brts_cpu = data.brts.cpu()
            brts_cpu = brts_cpu.unsqueeze(-1)
            packed_brts = pack_padded_sequence(brts_cpu, lengths_brts, batch_first=True, enforce_sorted=False).to(device)

            data = data.to(device)
            _, out_cl = model_lstm(packed_brts)

            outputs_all = torch.cat((outputs_all, out_cl), dim=0)
            y_all = torch.cat((y_all, data.y_cl.view(-1)), dim=0)

            # Get the predicted class by taking the class with the max score
            pred = out_cl.max(dim=1)[1]

            # Update the total correct predictions
            correct += pred.eq(data.y_cl.view(-1)).sum().item()

            # Update the correct count and total count per class
            for i in range(num_classes):
                mask = data.y_cl.view(-1) == i  # Find all samples belonging to class i
                correct_per_class[i] += pred[mask].eq(data.y_cl.view(-1)[mask]).sum().item()
                total_per_class[i] += mask.sum().item()

        # Calculate overall accuracy
        overall_accuracy = correct / total_samples

        # Calculate per-class accuracy
        class_accuracy = correct_per_class.float() / total_per_class.float()

        # Handle cases where a class might not be present in the batch
        class_accuracy[total_per_class == 0] = float('nan')  # Assign NaN to avoid division by zero

        return overall_accuracy, class_accuracy, outputs_all.cpu().detach().numpy(), y_all.cpu().detach().numpy()

    model_lstm = LSTM(in_channels=1, hidden_channels=lstm_hidden_channels,
                      out_channels=lstm_output_channels, lstm_depth=lstm_depth).to(device)
    optimizer_lstm = torch.optim.AdamW(model_lstm.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training using {device}")
    print(model_lstm)

    test_mean_diffs_history = []
    train_loss_all_history = []
    train_loss_regression_history = []
    train_loss_classification_history = []
    test_loss_all_history = []
    test_loss_regression_history = []
    test_loss_classification_history = []
    test_overall_accuracy_history = []
    test_per_class_accuracy_history = []
    final_test_diffs = []
    final_test_predictions = []
    final_test_y = []
    final_test_nodes = []
    final_test_label_pred = []
    final_test_label_true = []
    final_overall_accuracy = []
    final_per_class_accuracy = []

    # Set up the early stopper
    # early_stopper = EarlyStopper(patience=3, min_delta=0.05)
    actual_epoch_lstm = 0
    # The losses are summed over each data point in the batch, thus we should normalize the losses accordingly
    train_test_ratio = len(train_loader.dataset) / len(test_loader.dataset)

    print("Now training LSTM model...")

    for epoch in range(1, epoch_number_lstm):
        actual_epoch_lstm = epoch
        train_loss_all, train_loss_reg, train_loss_cls = train_lstm()
        test_loss_all, test_loss_reg, test_loss_cls = compute_test_loss_lstm()
        test_loss_all = test_loss_all * train_test_ratio
        test_loss_reg = test_loss_reg * train_test_ratio
        test_loss_cls = test_loss_cls * train_test_ratio
        test_mean_diffs, test_diffs_all, test_predictions, test_y, test_nodes_all = test_diff_lstm(test_loader)
        test_accuracy_all, test_accuracy_class, test_label_pred, test_label_true = test_accu_lstm(test_loader, n_classes)
        print(f'Epoch: {epoch:03d}, Train Regression Loss: {train_loss_reg:.4f}, Train Classification Loss: {train_loss_cls:.4f}, Train Combined Loss: {train_loss_all:.4f}')
        print(f'Epoch: {epoch:03d}, Test Regression Loss: {test_loss_reg:.4f}, Test Classification Loss: {test_loss_cls:.4f}, Test Combined Loss: {test_loss_all:.4f}')
        print(f'Epoch: {epoch:03d}, Overall Classification Accuracy: {test_accuracy_all:.4f}')
        # Convert the tensors to arrayss
        train_loss_reg = train_loss_reg.cpu().detach().numpy()
        train_loss_cls = train_loss_cls.cpu().detach().numpy()
        test_loss_reg = test_loss_reg.cpu().detach().numpy()
        test_loss_cls = test_loss_cls.cpu().detach().numpy()
        test_accuracy_class = test_accuracy_class.cpu().detach().numpy()
        print(f'Epoch: {epoch:03d}, Class 0 Accuracy: {test_accuracy_class[0]:.4f}, Class 1 Accuracy: {test_accuracy_class[1]:.4f}, Class 2 Accuracy: {test_accuracy_class[2]:.4f}')
        # Record the values
        test_mean_diffs_history.append(test_mean_diffs)
        train_loss_all_history.append(train_loss_all)
        train_loss_regression_history.append(train_loss_reg)
        train_loss_classification_history.append(train_loss_cls)
        test_loss_all_history.append(test_loss_all)
        test_loss_regression_history.append(test_loss_reg)
        test_loss_classification_history.append(test_loss_cls)
        test_overall_accuracy_history.append(test_accuracy_all)
        test_per_class_accuracy_history.append(test_accuracy_class)
        final_test_diffs = test_diffs_all
        final_test_predictions = test_predictions
        final_test_y = test_y
        final_test_nodes = test_nodes_all
        final_test_label_pred = test_label_pred
        final_test_label_true = test_label_true
        final_overall_accuracy = test_accuracy_all
        final_per_class_accuracy = test_accuracy_class
        print(f"Final test diffs length: {len(final_test_diffs)}")
        print(f"Final predictions length: {len(final_test_predictions)}")
        print(f"Final y length: {len(final_test_y)}")
        print(f"Final nodes length: {len(final_test_nodes)}")

    # Save the model
    print("Saving LSTM model...")
    # Create the directory if it doesn't exist
    if not os.path.exists(os.path.join(name, task_type, "STBO")):
        os.makedirs(os.path.join(name, task_type, "STBO"))
    torch.save(model_lstm.state_dict(),
               os.path.join(name, task_type, "STBO", f"{task_type}_model_diffpool_{gnn_depth}_lstm.pt"))


    # After the loop, create a dictionary to hold the data
    data_dict = {"lambda_diff": [], "mu_diff": [], "beta_n_diff": [], "beta_phi_diff": [], "gamma_n_diff": [], "gamma_phi_diff": []}

    # Iterate through test_mean_diffs_history
    for array in test_mean_diffs_history:
        # Safely append each value, filling with 0 if missing
        data_dict["lambda_diff"].append(safe_append(array, 0))
        data_dict["mu_diff"].append(safe_append(array, 1))
        data_dict["beta_n_diff"].append(safe_append(array, 2))
        data_dict["beta_phi_diff"].append(safe_append(array, 3))
        data_dict["gamma_n_diff"].append(safe_append(array, 4))
        data_dict["gamma_phi_diff"].append(safe_append(array, 5))

    # Ensure the length of other lists matches actual_epoch_lstm, filling missing values with 0
    actual_epoch_lstm = len(train_loss_all_history)  # Ensure epoch count matches available data

    data_dict["Epoch"] = list(range(1, actual_epoch_lstm + 1))
    data_dict["Train_Loss_ALL"] = train_loss_all_history[:actual_epoch_lstm] + [0] * (actual_epoch_lstm - len(train_loss_all_history))
    data_dict["Train_Loss_Regression"] = train_loss_regression_history[:actual_epoch_lstm] + [0] * (actual_epoch_lstm - len(train_loss_regression_history))
    data_dict["Train_Loss_Classification"] = train_loss_classification_history[:actual_epoch_lstm] + [0] * (actual_epoch_lstm - len(train_loss_classification_history))
    data_dict["Test_Loss_ALL"] = test_loss_all_history[:actual_epoch_lstm] + [0] * (actual_epoch_lstm - len(test_loss_all_history))
    data_dict["Test_Loss_Regression"] = test_loss_regression_history[:actual_epoch_lstm] + [0] * (actual_epoch_lstm - len(test_loss_regression_history))
    data_dict["Test_Loss_Classification"] = test_loss_classification_history[:actual_epoch_lstm] + [0] * (actual_epoch_lstm - len(test_loss_classification_history))

    # Convert the dictionary to a pandas DataFrame
    model_performance = pd.DataFrame(data_dict)

    # Using the safe_create_dataframe function for each DataFrame
    final_differences = safe_create_dataframe(final_test_diffs, ["lambda_diff", "mu_diff", "beta_n_diff", "beta_phi_diff", "gamma_n_diff", "gamma_phi_diff"])
    final_predictions = safe_create_dataframe(final_test_predictions, ["lambda_pred", "mu_pred", "beta_n_pred", "beta_phi_pred", "gamma_n_pred", "gamma_phi_pred"])
    final_y = safe_create_dataframe(final_test_y, ["lambda", "mu", "beta_n", "beta_phi", "gamma_n", "gamma_phi"])
    final_nodes = safe_create_dataframe(final_test_nodes, ["num_nodes"])
    final_label_prob = safe_create_dataframe(final_test_label_pred, ["pd_prob", "ed_prob", "nnd_prob"])
    final_label_true = safe_create_dataframe(final_test_label_true, ["true_class"])

    # Column-wise combine all final DataFrames
    final_result = pd.concat([final_differences, final_predictions, final_y, final_nodes, final_label_prob, final_label_true], axis=1)

    print("Final differences:")
    print(abs(final_differences[["lambda_diff", "mu_diff", "beta_n_diff", "beta_phi_diff", "gamma_n_diff", "gamma_phi_diff"]]).mean())
    print("Final overall accuracy:", final_overall_accuracy)
    print("Final per-class accuracy:", final_per_class_accuracy)

    # Workaround to get rid of the dtype incompatible issue
    model_performance = model_performance.astype(object)
    final_result = final_result.astype(object)

    # Save the data to a file using pyreadr
    pyreadr.write_rds(os.path.join(name, task_type, "STBO", f"{task_type}_diffpool_{gnn_depth}_lstm.rds"), model_performance)
    pyreadr.write_rds(os.path.join(name, task_type, "STBO", f"{task_type}_final_diffpool_{gnn_depth}_lstm.rds"), final_result)


if __name__ == '__main__':
    main()
