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
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DenseDataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool
from torch_geometric.nn import DenseSAGEConv as SAGEConv
from torch_geometric.nn import AntiSymmetricConv as ASConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

# Load the global parameters from the config file
global_params = None

with open("../Config/ddd_train_diffpool.yaml", "r") as ymlfile:
    global_params = yaml.safe_load(ymlfile)

# Set global variables
cap_norm_factor = global_params["cap_norm_factor"]
epoch_number = global_params["epoch_number"]
diffpool_ratio = global_params["diffpool_ratio"]
dropout_ratio = global_params["dropout_ratio"]
learning_rate = global_params["learning_rate"]
train_batch_size = global_params["train_batch_size"]
test_batch_size = global_params["test_batch_size"]
gcn_layer1_hidden_channels = global_params["gcn_layer1_hidden_channels"]
gcn_layer2_hidden_channels = global_params["gcn_layer2_hidden_channels"]
gcn_layer3_hidden_channels = global_params["gcn_layer3_hidden_channels"]
lin_layer1_hidden_channels = global_params["lin_layer1_hidden_channels"]
lin_layer2_hidden_channels = global_params["lin_layer2_hidden_channels"]
lin_layer3_hidden_channels = global_params["lin_layer3_hidden_channels"]
lin_layer4_hidden_channels = global_params["lin_layer4_hidden_channels"]
lin_layer5_hidden_channels = global_params["lin_layer5_hidden_channels"]
n_predicted_values = global_params["n_predicted_values"]
batch_size_reduce_factor = global_params["batch_size_reduce_factor"]
max_nodes_limit = global_params["max_nodes_limit"]
normalize_edge_length = global_params["normalize_edge_length"]
normalize_graph_representation = global_params["normalize_graph_representation"]
huber_delta = global_params["huber_delta"]
global_pooling_method = global_params["global_pooling_method"]
minimum_nodes_limit = global_params["minimum_nodes_limit"]


def read_table(path):
    return pd.read_csv(path, sep="\s+", header=0)  # assuming the tables are tab-delimited


def check_same_across_rows(df):
    return df.apply(lambda x: x.nunique() == 1)


def count_rds_files(path):
    # Get the list of .rds files in the specified path
    rds_files = glob.glob(os.path.join(path, '*.rds'))
    return len(rds_files)


def check_rds_files_count(tree_path, el_path, st_path):
    # Count the number of .rds files in all three paths
    tree_count = count_rds_files(tree_path)
    el_count = count_rds_files(el_path)
    st_count = count_rds_files(st_path)

    # Check if the counts are equal
    if tree_count == el_count == st_count:
        return tree_count  # Assuming all counts are equal, return one of them
    else:
        raise ValueError("The number of .rds files in the three paths are not equal")

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
    params = filename.split('_')[1:-1]
    params = list(map(float, params))  # Convert string to float
    return params


def get_sort_key(filename):
    # Split the filename by underscores, convert the parameter values to floats, and return them as a tuple
    params = tuple(map(float, filename.split('_')[1:-1]))
    return params


def sort_files(files):
    # Sort the files based on the parameters extracted by the get_sort_key function
    return sorted(files, key=get_sort_key)


def check_file_consistency(files_tree, files_el, files_st):
    # Check if the three lists have the same length
    if not (len(files_tree) == len(files_el) == len(files_st)):
        raise ValueError("Mismatched lengths among file lists.")

    # Define a function to extract parameters from filename
    def get_params_tuple(filename):
        return tuple(map(float, filename.split('_')[1:-1]))

    # Check each trio of files for matching parameters
    for tree_file, el_file, st_file in zip(files_tree, files_el, files_st):
        tree_params = get_params_tuple(tree_file)
        el_params = get_params_tuple(el_file)
        st_params = get_params_tuple(st_file)

        if not (tree_params == el_params == st_params):
            raise ValueError(f"Mismatched parameters among files: {tree_file}, {el_file}, and {st_file}")

    # If we get here, all checks passed
    print("File lists consistency check passed across tree, EL, and ST datasets.")


def check_params_consistency(params_tree_list, params_el_list, params_st_list):
    # Check if all corresponding elements in the three lists are equal
    is_consistent = all(a == b == c for a, b, c in zip(params_tree_list, params_el_list, params_st_list))

    if is_consistent:
        print("Parameters are consistent across the tree, EL, and ST datasets.")
    else:
        raise ValueError("Mismatch in parameters between the tree, EL, and ST datasets.")

    return is_consistent


def check_list_count(count, data_list, length_list, params_list, stats_list):
    # Get the number of elements in each list
    data_count = len(data_list)
    length_count = len(length_list)
    params_count = len(params_list)
    stats_count = len(stats_list)  # Get the count for the new stats_list

    # Check if the count matches the number of elements in each list
    if count != data_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, data_list has {data_count} elements.")

    if count != length_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, length_list has {length_count} elements.")

    if count != params_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, params_list has {params_count} elements.")

    if count != stats_count:  # New check for stats_list
        raise ValueError(f"Count mismatch: input argument count is {count}, stats_list has {stats_count} elements.")

    # If all checks pass, print a success message
    print("Count check passed")


def read_rds_to_pytorch(path, count, normalize=False):
    # List all files in the directory
    files_tree = [f for f in os.listdir(os.path.join(path, 'GNN', 'tree'))
                  if f.startswith('tree_') and f.endswith('.rds')]
    files_el = [f for f in os.listdir(os.path.join(path, 'GNN', 'tree', 'EL'))
                if f.startswith('EL_') and f.endswith('.rds')]
    files_st = [f for f in os.listdir(os.path.join(path, 'GNN', 'tree', 'ST'))
                if f.startswith('ST_') and f.endswith('.rds')]

    # Sort the files based on the parameters
    files_tree = sort_files(files_tree)
    files_el = sort_files(files_el)
    files_st = sort_files(files_st)

    # Check if the files are consistent
    check_file_consistency(files_tree, files_el, files_st)

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

    stats_list = []
    params_st_list = []

    # Loop through the files with the prefix 'ST_'
    for filename in files_st:
        stats_file_path = os.path.join(path, 'GNN', 'tree', 'ST', filename)
        stats_result = pyreadr.read_r(stats_file_path)
        stats_data = stats_result[None]
        stats_list.append(stats_data)
        params_st_list.append(get_params_string(filename))

    check_params_consistency(params_tree_list, params_el_list, params_st_list)

    params_list = []

    for filename in files_tree:
        params = get_params(filename)
        params_list.append(params)

    # Normalize carrying capacity by dividing by a factor
    for vector in params_list:
        vector[2] = vector[2] / cap_norm_factor

    check_list_count(count, data_list, length_list, params_list, stats_list)

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

        params_current_tensor = torch.tensor(params_current[0:n_predicted_values], dtype=torch.float)

        stats_tensor = torch.tensor(stats_list[i].values, dtype=torch.float)

        # Create a Data object with the edge index, number of nodes, and category value
        data = Data(x=edge_length_tensor,
                    edge_index=edge_index_tensor,
                    num_nodes=num_nodes,
                    y=params_current_tensor,
                    stats=stats_tensor)

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


def shuffle_data(data_list):
    # Create a copy of the data list to shuffle
    shuffled_list = data_list.copy()
    # Shuffle the copied list in place
    random.shuffle(shuffled_list)
    return shuffled_list


def export_to_rds(embeddings, epoch, name, task_type, which_set):
    # Convert to DataFrame
    df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])

    # Export to RDS
    rds_path = os.path.join(name, task_type, "umap")
    if not os.path.exists(rds_path):
        os.makedirs(rds_path)
    rds_filename = os.path.join(rds_path, f'{which_set}_umap_epoch_{epoch}.rds')
    pyreadr.write_rds(rds_filename, df)


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <name> <task_type> <gnn_depth>")
        sys.exit(1)

    name = sys.argv[1]
    task_type = sys.argv[2]
    gnn_depth = int(sys.argv[3])

    # Now you can use the variables name and set_i in your code
    print(f'Name: {name}, Task Type: {task_type}', f'GNN Depth: {gnn_depth}')

    training_dataset_list = []
    testing_dataset_list = []

    # Concatenate the base directory path with the set_i folder name
    full_dir = os.path.join(name, task_type)
    full_dir_tree = os.path.join(full_dir, 'GNN', 'tree')
    full_dir_el = os.path.join(full_dir, 'GNN', 'tree', 'EL')
    full_dir_st = os.path.join(full_dir, 'GNN', 'tree', 'ST')
    # Call read_rds_to_pytorch with the full directory path
    print(full_dir)
    # Check if the number of .rds files in the tree and el paths are equal
    rds_count = check_rds_files_count(full_dir_tree, full_dir_el, full_dir_st)
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

    if task_type == "DDD_FREE_TES":
        val_dir = os.path.join(name, "DDD_VAL_TES")
        train_batch_size_adjusted = int(train_batch_size)
        test_batch_size_adjusted = int(test_batch_size)
    elif task_type == "DDD_FREE_TAS":
        val_dir = os.path.join(name, "DDD_VAL_TAS")
        train_batch_size_adjusted = int(ceil(train_batch_size * batch_size_reduce_factor))
        test_batch_size_adjusted = int(ceil(test_batch_size * batch_size_reduce_factor))
    else:
        raise ValueError("Invalid task type.")

    full_val_dir_tree = os.path.join(val_dir, 'GNN', 'tree')
    full_val_dir_el = os.path.join(val_dir, 'GNN', 'tree', 'EL')
    full_val_dir_st = os.path.join(val_dir, 'GNN', 'tree', 'ST')
    val_rds_count = check_rds_files_count(full_val_dir_tree, full_val_dir_el, full_val_dir_st)
    print(f'There are: {val_rds_count} trees in the validation folder.')
    print(f"Now reading validation data...")
    current_val_dataset = read_rds_to_pytorch(val_dir, val_rds_count, normalize_edge_length)
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

    # Filtering out trees with less than 200 nodes for the training set
    # TODO: Add switch to decide which sets to filter
    # filtered_training_data = [data for data in filtered_training_data if data.num_nodes >= minimum_nodes_limit]
    # filtered_testing_data = [data for data in filtered_testing_data if data.num_nodes >= max_nodes_limit]
    # filtered_validation_data = [data for data in filtered_validation_data if data.num_nodes >= max_nodes_limit]

    class TreeData(InMemoryDataset):
        def __init__(self, root, data_list, transform=None, pre_transform=None):
            super(TreeData, self).__init__(root, transform, pre_transform)
            self.data, self.slices = self.collate(data_list)

        def _download(self):
            pass  # No download required

        def _process(self):
            pass  # No processing required

    max_nodes_train = max([data.num_nodes for data in filtered_training_data])
    max_nodes_test = max([data.num_nodes for data in filtered_testing_data])
    max_nodes_val = max([data.num_nodes for data in filtered_validation_data])
    max_nodes = max(max_nodes_train, max_nodes_test, max_nodes_val)
    print(f"Max nodes: {max_nodes} for {task_type}")

    #training_dataset = TreeData(root=None, data_list=filtered_training_data, transform=T.ToDense(max_nodes))
    training_dataset = TreeData(root=None, data_list=filtered_training_data, transform=None)

    #testing_dataset = TreeData(root=None, data_list=filtered_testing_data, transform=T.ToDense(max_nodes))
    testing_dataset = TreeData(root=None, data_list=filtered_testing_data, transform=None)

    num_stats = training_dataset[0].stats.shape[0]

    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels,
                     normalize=False, lin=True):
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

    class DiffPool(torch.nn.Module):
        def __init__(self):
            super(DiffPool, self).__init__()

            self.graph_sizes = torch.tensor([], dtype=torch.long)
            self.stats = torch.tensor([], dtype=torch.float)

            num_nodes1 = ceil(diffpool_ratio * max_nodes)
            self.gnn1_pool = GNN(training_dataset.num_node_features, gcn_layer1_hidden_channels, num_nodes1)
            self.gnn1_embed = GNN(training_dataset.num_node_features, gcn_layer1_hidden_channels, gcn_layer2_hidden_channels)

            num_nodes2 = ceil(diffpool_ratio * num_nodes1)
            gnn1_out_channels = gcn_layer1_hidden_channels * (gnn_depth - 1) + gcn_layer2_hidden_channels
            self.gnn2_pool = GNN(gnn1_out_channels, gcn_layer2_hidden_channels, num_nodes2)
            self.gnn2_embed = GNN(gnn1_out_channels, gcn_layer2_hidden_channels, gcn_layer3_hidden_channels, lin=False)

            gnn2_out_channels = gcn_layer2_hidden_channels * (gnn_depth - 1) + gcn_layer3_hidden_channels
            self.gnn3_embed = GNN(gnn2_out_channels, gcn_layer3_hidden_channels, lin_layer1_hidden_channels, lin=False)

            gnn3_out_channels = gcn_layer3_hidden_channels * (gnn_depth - 1) + lin_layer1_hidden_channels
            self.lin1 = torch.nn.Linear(gnn3_out_channels + num_stats, lin_layer2_hidden_channels)
            self.lin2 = torch.nn.Linear(lin_layer2_hidden_channels, lin_layer3_hidden_channels)
            self.lin3 = torch.nn.Linear(lin_layer3_hidden_channels, lin_layer4_hidden_channels)
            self.lin4 = torch.nn.Linear(lin_layer4_hidden_channels, lin_layer5_hidden_channels)
            self.lin5 = torch.nn.Linear(lin_layer5_hidden_channels, n_predicted_values)

        def forward(self, x, adj, mask=None, graph_sizes=None, stats=None):
            s = self.gnn1_pool(x, adj, mask)
            x = self.gnn1_embed(x, adj, mask)

            x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

            s = self.gnn2_pool(x, adj)
            x = self.gnn2_embed(x, adj)

            x, adj, l2, e2 = dense_diff_pool(x, adj, s)

            x = self.gnn3_embed(x, adj)

            if global_pooling_method == "mean":
                x = x.mean(dim=1)
            elif global_pooling_method == "max":
                x = x.max(dim=1).values
            elif global_pooling_method == "sum":
                x = x.sum(dim=1)
            else:
                raise ValueError("Invalid global pooling method.")

            if normalize_graph_representation:
                self.graph_sizes = graph_sizes.view(-1, 1).to(device)
                x = x / self.graph_sizes

            self.stats = stats
            self.stats = torch.squeeze(self.stats, -1).to(device)
            x = torch.cat((x, self.stats), dim=1)

            x = F.dropout(x, p=dropout_ratio, training=self.training)
            x = self.lin1(x)
            x = F.gelu(x)
            x = F.dropout(x, p=dropout_ratio, training=self.training)
            x = self.lin2(x)
            x = F.gelu(x)
            x = F.dropout(x, p=dropout_ratio, training=self.training)
            x = self.lin3(x)
            x = F.gelu(x)
            x = F.dropout(x, p=dropout_ratio, training=self.training)
            x = self.lin4(x)
            x = F.gelu(x)
            x = F.dropout(x, p=dropout_ratio, training=self.training)
            x = self.lin5(x)
            # x = F.relu(x)

            return x, l1 + l2, e1 + e2

    class AntiSymmetricConv(torch.nn.Module):
        def __init__(self):
            super(AntiSymmetricConv, self).__init__()

            self.asconv = ASConv(in_channels=training_dataset.num_node_features, num_iters=2 * gnn_depth, act="relu")
            self.lin1 = torch.nn.Linear(training_dataset.num_node_features * 3, training_dataset.num_node_features * 3 // 2)
            self.lin2 = torch.nn.Linear(training_dataset.num_node_features * 3 // 2, n_predicted_values)

        def forward(self, data: Data) -> torch.Tensor:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.asconv(x, edge_index)

            x = torch.cat([global_add_pool(x, batch), global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

            x = F.dropout(x, p=dropout_ratio, training=self.training)
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, p=dropout_ratio, training=self.training)
            x = self.lin2(x)

            return x

    class EarlyStopper:
        def __init__(self, patience=3, min_delta=0.1):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_loss = float('inf')

        def early_stop(self, validation_loss):
            if self.min_validation_loss - validation_loss >= self.min_delta:
                self.min_validation_loss = validation_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False

    def train():
        model.train()

        loss_all = 0  # Keep track of the loss
        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.view(data.num_graphs, n_predicted_values))
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

        return loss_all / len(train_loader.dataset)

    @torch.no_grad()
    def test_diff(loader):
        model.eval()

        diffs_all = torch.tensor([], dtype=torch.float, device=device)
        outputs_all = torch.tensor([], dtype=torch.float, device=device)  # To store all outputs
        y_all = torch.tensor([], dtype=torch.float, device=device)  # To store all y
        nodes_all = torch.tensor([], dtype=torch.long, device=device)

        for data in loader:
            data.to(device)
            out = model(data)
            diffs = torch.abs(out - data.y.view(data.num_graphs, n_predicted_values))
            diffs_all = torch.cat((diffs_all, diffs), dim=0)
            outputs_all = torch.cat((outputs_all, out), dim=0)
            y_all = torch.cat((y_all, data.y.view(data.num_graphs, n_predicted_values)), dim=0)
            for i in range(data.num_graphs):
                current_num_nodes = torch.tensor([data[i].num_nodes], dtype=torch.long, device=device)
                nodes_all = torch.cat((nodes_all, current_num_nodes), dim=0)

        print(f"diffs_all length: {len(diffs_all)}; test_loader.dataset length: {len(test_loader.dataset)}; Equal: {len(diffs_all) == len(test_loader.dataset)}")
        mean_diffs = torch.sum(diffs_all, dim=0) / len(test_loader.dataset)
        return mean_diffs.cpu().detach().numpy(), diffs_all.cpu().detach().numpy(), outputs_all.cpu().detach().numpy(), y_all.cpu().detach().numpy(), nodes_all.cpu().detach().numpy()

    @torch.no_grad()
    def compute_test_loss():
        model.eval()  # Set the model to evaluation mode
        loss_all = 0  # Keep track of the loss
        for data in test_loader:
            data.to(device)
            out = model(data)
            loss = criterion(out, data.y.view(data.num_graphs, n_predicted_values))
            loss_all += loss.item() * data.num_graphs

        return loss_all / len(train_loader.dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training using {device}")

    #model = DiffPool()
    model = AntiSymmetricConv()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.HuberLoss(delta=huber_delta).to(device)

    #train_loader = DenseDataLoader(training_dataset, batch_size=train_batch_size_adjusted, shuffle=False)
    train_loader = DataLoader(training_dataset, batch_size=train_batch_size_adjusted, shuffle=False)
    #test_loader = DenseDataLoader(testing_dataset, batch_size=test_batch_size_adjusted, shuffle=False)
    test_loader = DataLoader(testing_dataset, batch_size=test_batch_size_adjusted, shuffle=False)

    print(f"Training dataset length: {len(train_loader.dataset)}")
    print(f"Testing dataset length: {len(test_loader.dataset)}")

    print(model)

    test_mean_diffs_history = []
    train_loss_history = []
    test_loss_history = []
    final_test_diffs = []
    final_test_predictions = []
    final_test_y = []
    final_test_nodes = []

    # Set up the early stopper
    # early_stopper = EarlyStopper(patience=3, min_delta=0.05)
    actual_epoch = 0
    # The losses are summed over each data point in the batch, thus we should normalize the losses accordingly
    train_test_ratio = len(train_loader.dataset) / len(test_loader.dataset)

    for epoch in range(1, epoch_number):
        actual_epoch = epoch
        train_loss_all = train()
        test_loss_all = compute_test_loss()
        test_loss_all = test_loss_all * train_test_ratio
        test_mean_diffs, test_diffs_all, test_predictions, test_y, test_nodes_all = test_diff(test_loader)
        test_mean_diffs[2] = test_mean_diffs[2] * cap_norm_factor
        print(f'Epoch: {epoch:03d}, Par 1 Mean Diff: {test_mean_diffs[0]:.4f}, Par 2 Mean Diff: {test_mean_diffs[1]:.4f}, Par 3 Mean Diff: {test_mean_diffs[2]:.4f}, Train Loss: {train_loss_all:.4f}, Test Loss: {test_loss_all:.4f}')

        # Record the values
        test_mean_diffs_history.append(test_mean_diffs)
        train_loss_history.append(train_loss_all)
        test_loss_history.append(test_loss_all)
        final_test_diffs = test_diffs_all
        final_test_predictions = test_predictions
        final_test_y = test_y
        final_test_nodes = test_nodes_all
        print(f"Final test diffs length: {len(final_test_diffs)}")
        print(f"Final predictions length: {len(final_test_predictions)}")
        print(f"Final y length: {len(final_test_y)}")
        print(f"Final nodes length: {len(final_test_nodes)}")

        #  if early_stopper.early_stop(test_loss_all):
        #      print(f"Early stopping at epoch {epoch}")
        #      break

    # Save the model
    print(f"Saving model to {os.path.join(name, task_type, f'{task_type}_model_diffpool_{gnn_depth}.pt')}")
    torch.save(model.state_dict(), os.path.join(name, task_type, f"{task_type}_model_diffpool_{gnn_depth}.pt"))

    # After the loop, create a dictionary to hold the data
    data_dict = {"lambda_diff": [], "mu_diff": [], "cap_diff": []}
    # Iterate through test_mean_diffs_history
    for array in test_mean_diffs_history:
        # It's assumed that the order of elements in the array corresponds to the keys in data_dict
        data_dict["lambda_diff"].append(array[0])
        data_dict["mu_diff"].append(array[1])
        data_dict["cap_diff"].append(array[2])
    data_dict["Epoch"] = list(range(1, actual_epoch + 1))
    data_dict["Train_Loss"] = train_loss_history
    data_dict["Test_Loss"] = test_loss_history

    # Convert the dictionary to a pandas DataFrame
    model_performance = pd.DataFrame(data_dict)
    final_differences = pd.DataFrame(final_test_diffs, columns=["lambda_diff", "mu_diff", "cap_diff"])
    final_predictions = pd.DataFrame(final_test_predictions, columns=["lambda_pred", "mu_pred", "cap_pred"])
    final_y = pd.DataFrame(final_test_y, columns=["lambda", "mu", "cap"])
    final_differences["num_nodes"] = final_test_nodes
    # Save the data to a file using pyreadr
    pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_diffpool_{gnn_depth}.rds"), model_performance)
    pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_final_diffs_diffpool_{gnn_depth}.rds"), final_differences)
    pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_final_predictions_diffpool_{gnn_depth}.rds"), final_predictions)
    pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_final_y_diffpool_{gnn_depth}.rds"), final_y)


if __name__ == '__main__':
    main()
