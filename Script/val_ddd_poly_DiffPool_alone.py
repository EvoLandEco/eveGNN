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
max_nodes = 2147
max_brts_len = 1073

with open("../Config/ddd_train_diffpool.yaml", "r") as ymlfile:
    global_params = yaml.safe_load(ymlfile)

# Set global variables
cap_norm_factor = global_params["cap_norm_factor"]
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
batch_size_reduce_factor = global_params["batch_size_reduce_factor"]
max_nodes_limit = global_params["max_nodes_limit"]
normalize_edge_length = global_params["normalize_edge_length"]
normalize_graph_representation = global_params["normalize_graph_representation"]
huber_delta = global_params["huber_delta"]
global_pooling_method = global_params["global_pooling_method"]


def read_table(path):
    return pd.read_csv(path, sep="\s+", header=0)  # assuming the tables are tab-delimited


def check_same_across_rows(df):
    return df.apply(lambda x: x.nunique() == 1)


def count_rds_files(path):
    # Get the list of .rds files in the specified path
    rds_files = glob.glob(os.path.join(path, '*.rds'))
    return len(rds_files)


def check_rds_files_count(tree_path, el_path, st_path, bt_path):
    # Count the number of .rds files in all four paths
    tree_count = count_rds_files(tree_path)
    el_count = count_rds_files(el_path)
    st_count = count_rds_files(st_path)
    bt_count = count_rds_files(bt_path)  # Count for the new bt_path

    # Check if the counts are equal
    if tree_count == el_count == st_count == bt_count:
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


def check_file_consistency(files_tree, files_el, files_st, files_bt):
    # Check if the four lists have the same length
    if not (len(files_tree) == len(files_el) == len(files_st) == len(files_bt)):
        raise ValueError("Mismatched lengths among file lists.")

    # Define a function to extract parameters from filename
    def get_params_tuple(filename):
        return tuple(map(float, filename.split('_')[1:-1]))

    # Check each quartet of files for matching parameters
    for tree_file, el_file, st_file, bt_file in zip(files_tree, files_el, files_st, files_bt):
        tree_params = get_params_tuple(tree_file)
        el_params = get_params_tuple(el_file)
        st_params = get_params_tuple(st_file)
        bt_params = get_params_tuple(bt_file)

        if not (tree_params == el_params == st_params == bt_params):
            raise ValueError(f"Mismatched parameters among files: {tree_file}, {el_file}, {st_file}, {bt_file}")

    # If we get here, all checks passed
    print("File lists consistency check passed across tree, EL, ST, and BT datasets.")


def check_params_consistency(params_tree_list, params_el_list, params_st_list, params_bt_list):
    # Check if all corresponding elements in the four lists are equal
    is_consistent = all(
        a == b == c == d for a, b, c, d in zip(params_tree_list, params_el_list, params_st_list, params_bt_list))

    if is_consistent:
        print("Parameters are consistent across the tree, EL, ST, and BT datasets.")
    else:
        raise ValueError("Mismatch in parameters between the tree, EL, ST, and BT datasets.")

    return is_consistent


def check_list_count(count, data_list, length_list, params_list, stats_list, brts_list):
    # Get the number of elements in each list
    data_count = len(data_list)
    length_count = len(length_list)
    params_count = len(params_list)
    stats_count = len(stats_list)
    brts_count = len(brts_list)  # Calculate the count for the new brts_list

    # Check if the count matches the number of elements in each list
    if count != data_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, data_list has {data_count} elements.")

    if count != length_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, length_list has {length_count} elements.")

    if count != params_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, params_list has {params_count} elements.")

    if count != stats_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, stats_list has {stats_count} elements.")

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
    files_st = [f for f in os.listdir(os.path.join(path, 'GNN', 'tree', 'ST'))
                if f.startswith('ST_') and f.endswith('.rds')]
    files_bt = [f for f in os.listdir(os.path.join(path, 'GNN', 'tree', 'BT'))
                if f.startswith('BT_') and f.endswith('.rds')]  # Get the list of files in the new BT directory

    # Sort the files based on the parameters
    files_tree = sort_files(files_tree)
    files_el = sort_files(files_el)
    files_st = sort_files(files_st)
    files_bt = sort_files(files_bt)  # Sort the files in the new BT directory

    # Check if the files are consistent
    check_file_consistency(files_tree, files_el, files_st, files_bt)

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

    brts_list = []
    params_bt_list = []

    # Loop through the files with the prefix 'BT_'
    for filename in files_bt:
        brts_file_path = os.path.join(path, 'GNN', 'tree', 'BT', filename)
        brts_result = pyreadr.read_r(brts_file_path)
        brts_data = brts_result[None]
        brts_list.append(brts_data)
        params_bt_list.append(get_params_string(filename))

    check_params_consistency(params_tree_list, params_el_list, params_st_list, params_bt_list)

    params_list = []

    for filename in files_tree:
        params = get_params(filename)
        params_list.append(params)

    # Normalize carrying capacity by dividing by a factor
    for vector in params_list:
        vector[2] = vector[2] / cap_norm_factor

    check_list_count(count, data_list, length_list, params_list, stats_list, brts_list)

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

        stats_tensor = stats_tensor.squeeze(1)  # Remove the extra dimension

        brts_tensor = torch.tensor(brts_list[i].values, dtype=torch.float)

        brts_tensor = brts_tensor.squeeze(1)  # Remove the extra dimension

        brts_length = torch.tensor([len(brts_list[i].values)], dtype=torch.long)

        # Create a Data object with the edge index, number of nodes, and category value
        data = Data(x=edge_length_tensor,
                    edge_index=edge_index_tensor,
                    num_nodes=num_nodes,
                    y=params_current_tensor,
                    stats=stats_tensor,
                    brts=brts_tensor,
                    brts_len=brts_length)

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
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <name> <gnn_depth>")
        sys.exit(1)

    name = sys.argv[1]
    gnn_depth = int(sys.argv[2])

    # Now you can use the variables name and set_i in your code
    print(f'Name: {name}, Task Type: DDD POLY', f'GNN Depth: {gnn_depth}')
    print("Now on branch Multimodal-Stacking-Boosting")

    validation_dataset_list = []
    val_dir = os.path.join(name, "DDD_POLY_TES")

    full_val_dir_tree = os.path.join(val_dir, 'GNN', 'tree')
    full_val_dir_el = os.path.join(val_dir, 'GNN', 'tree', 'EL')
    full_val_dir_st = os.path.join(val_dir, 'GNN', 'tree', 'ST')
    full_val_dir_bt = os.path.join(val_dir, 'GNN', 'tree', 'BT')  # Add the full path for the new BT directory
    val_rds_count = check_rds_files_count(full_val_dir_tree, full_val_dir_el, full_val_dir_st, full_val_dir_bt)
    print(f'There are: {val_rds_count} trees in the validation folder.')
    print(f"Now reading validation data...")
    current_val_dataset = read_rds_to_pytorch(val_dir, val_rds_count, normalize_edge_length)
    validation_dataset_list.append(current_val_dataset)

    sum_validation_data = functools.reduce(lambda x, y: x + y, validation_dataset_list)

    # Filtering out trees with only 3 nodes
    # They might cause problems with ToDense
    filtered_validation_data = [data for data in sum_validation_data if data.edge_index.shape != torch.Size([2, 2])]

    # Filtering out trees with more than 3000 nodes
    filtered_validation_data = [data for data in filtered_validation_data if data.num_nodes <= max_nodes_limit]

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

    validation_dataset = TreeData(root=None, data_list=filtered_validation_data, transform=T.ToDense(max_nodes))

    # The dense neural network model class with variable number of linear layers
    class DNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
            super(DNN, self).__init__()

            self.lins = torch.nn.ModuleList()
            self.bns = torch.nn.ModuleList()
            self.dropouts = torch.nn.ModuleList()  # Add a list for dropout modules

            for i in range(dnn_depth):
                first_index = 0
                last_index = dnn_depth - 1

                if dnn_depth == 1:
                    self.lins.append(torch.nn.Linear(in_channels, out_channels))
                    self.bns.append(torch.nn.BatchNorm1d(out_channels))
                else:
                    if i == first_index:
                        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
                        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
                    elif i == last_index:
                        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
                        self.bns.append(torch.nn.BatchNorm1d(out_channels))
                    else:
                        self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
                        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

                # Initialize dropout modules
                self.dropouts.append(torch.nn.Dropout(dropout_rate))

            # Initialize readout layers
            readout_in_channels = hidden_channels * (dnn_depth - 1) + out_channels
            self.readout = torch.nn.Linear(readout_in_channels, n_predicted_values)

        def forward(self, x):
            outputs = []  # Initialize a list to store outputs at each step
            for step in range(len(self.lins)):
                x = F.gelu(self.lins[step](x))
                x = self.dropouts[step](x)  # Apply dropout after activation
                x = self.bns[step](x)
                outputs.append(x)

            x_concatenated = torch.cat(outputs, dim=-1)
            x_readout = self.readout(x_concatenated)
            return x_readout

    # The LSTM model class with variable number of layers to process variable-length branch time sequences
    class LSTM(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, lstm_depth=1, dropout_rate=0.5):
            super(LSTM, self).__init__()

            self.lstm = torch.nn.LSTM(input_size = in_channels, hidden_size = hidden_channels, num_layers=lstm_depth, batch_first=True, dropout=dropout_ratio if lstm_depth > 1 else 0)
            self.dropout = torch.nn.Dropout(dropout_rate)  # Dropout layer after LSTM
            self.lin = torch.nn.Linear(hidden_channels, out_channels)
            self.readout = torch.nn.Linear(out_channels, n_predicted_values)

        def forward(self, x):
            out, (h_n, c_n) = self.lstm(x)

            # Unpack the sequences
            # x, _ = pad_packed_sequence(x, batch_first=True)

            # Get the final hidden state
            final_hidden_state = h_n[-1, :, :]

            x = F.gelu(final_hidden_state)

            x = self.dropout(x)  # Apply dropout after LSTM and activation
            x = self.lin(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = self.readout(x)
            return x

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training using {device}")

    def shape_check(dataset, max_nodes):
        incorrect_shapes = []  # List to store indices of data elements with incorrect shapes
        for i in range(len(dataset)):
            data = dataset[i]
            # Check the shapes of data.x, data.adj, and data.mask
            if data.x.shape != torch.Size([max_nodes, 3]) or \
                    data.y.shape != torch.Size([n_predicted_values]) or \
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
    shape_check(validation_dataset, max_nodes)

    validation_loader = DenseDataLoader(validation_dataset, batch_size=64, shuffle=False)
    print(f"Validation dataset length: {len(validation_loader.dataset)}")

    # Create dir for saving data
    if not os.path.exists(os.path.join(name, "DDD_FREE_TES", "POLY")):
        os.makedirs(os.path.join(name, "DDD_FREE_TES", "POLY"))

    @torch.no_grad()
    def test_diff(loader):
        model_dnn.eval()
        model_lstm.eval()

        # Initialize tensors to store the differences, outputs, and y values for DNN model
        diffs_all_dnn = torch.tensor([], dtype=torch.float, device=device)
        outputs_all_dnn = torch.tensor([], dtype=torch.float, device=device)  # To store all outputs

        # Initialize tensors to store the differences, outputs, and y values for LSTM model
        diffs_all_lstm = torch.tensor([], dtype=torch.float, device=device)
        outputs_all_lstm = torch.tensor([], dtype=torch.float, device=device)  # To store all outputs

        # Initialize tensors to store common y values and node indices
        y_all = torch.tensor([], dtype=torch.float, device=device)  # To store all y
        nodes_all = torch.tensor([], dtype=torch.long, device=device)

        for data in loader:
            data.to(device)

            # Record common y values and node indices
            y_all = torch.cat((y_all, data.y.view(data.num_nodes.__len__(), n_predicted_values)), dim=0)
            nodes_all = torch.cat((nodes_all, data.num_nodes), dim=0)

            # Compute the DNN model outputs and differences
            out_dnn = model_dnn(data.stats)
            diffs_dnn = torch.abs(out_dnn - data.y.view(data.num_nodes.__len__(), n_predicted_values))
            diffs_all_dnn = torch.cat((diffs_all_dnn, diffs_dnn), dim=0)
            outputs_all_dnn = torch.cat((outputs_all_dnn, out_dnn), dim=0)

            # Compute the LSTM model outputs and differences
            lengths_brts = torch.sum(data.brts != 0, dim=1).cpu().tolist()
            brts_cpu = data.brts.cpu()
            brts_cpu = brts_cpu.unsqueeze(-1)
            packed_brts = pack_padded_sequence(brts_cpu, lengths_brts, batch_first=True, enforce_sorted=False).to(device)
            out_lstm = model_lstm(packed_brts)
            diffs_lstm = torch.abs(out_lstm - data.y.view(data.num_nodes.__len__(), n_predicted_values))
            diffs_all_lstm = torch.cat((diffs_all_lstm, diffs_lstm), dim=0)
            outputs_all_lstm = torch.cat((outputs_all_lstm, out_lstm), dim=0)

        print(f"diffs_all_dnn length: {len(diffs_all_dnn)}; test_loader.dataset length: {len(loader.dataset)}; Equal: {len(diffs_all_dnn) == len(loader.dataset)}")
        print(f"diffs_all_lstm length: {len(diffs_all_lstm)}; test_loader.dataset length: {len(loader.dataset)}; Equal: {len(diffs_all_lstm) == len(loader.dataset)}")

        mean_diffs_dnn = torch.sum(diffs_all_dnn, dim=0) / len(loader.dataset)
        mean_diffs_lstm = torch.sum(diffs_all_lstm, dim=0) / len(loader.dataset)

        return (y_all.cpu().detach().numpy(), nodes_all.cpu().detach().numpy(),
                mean_diffs_dnn.cpu().detach().numpy(),
                diffs_all_dnn.cpu().detach().numpy(),
                outputs_all_dnn.cpu().detach().numpy(),
                mean_diffs_lstm.cpu().detach().numpy(),
                diffs_all_lstm.cpu().detach().numpy(),
                outputs_all_lstm.cpu().detach().numpy())

    # Test trained DNN model on the same training dataset
    num_stats = validation_dataset[0].stats.shape[0]
    model_dnn = DNN(num_stats, dnn_hidden_channels, dnn_output_channels)
    model_dnn = model_dnn.to(device)

    model_lstm = LSTM(in_channels=1, hidden_channels=lstm_hidden_channels, out_channels=lstm_output_channels)
    model_lstm = model_lstm.to(device)

    # Load the models
    model_dnn.load_state_dict(torch.load(os.path.join(name, "DDD_FREE_TES", "Bagging", f"DDD_FREE_TES_model_dnn_diffpool_{gnn_depth}.pt")))
    model_lstm.load_state_dict(torch.load(os.path.join(name, "DDD_FREE_TES", "Bagging", f"DDD_FREE_TES_model_lstm_diffpool_{gnn_depth}.pt")))

    print(model_dnn)
    print(model_lstm)

    # Initialize common history lists
    final_y = []
    final_nodes = []

    # Initialize the history lists for the DNN differences and losses
    final_predictions_dnn = []

    # Initialize the history lists for the LSTM differences and losses
    final_predictions_lstm = []

    test_y_all, test_nodes_all, \
        test_mean_diffs_dnn, test_diffs_all_dnn, test_predictions_dnn, \
        test_mean_diffs_lstm, test_diffs_all_lstm, test_predictions_lstm = test_diff(validation_loader)

    test_mean_diffs_dnn[2] = test_mean_diffs_dnn[2] * cap_norm_factor
    test_mean_diffs_lstm[2] = test_mean_diffs_lstm[2] * cap_norm_factor

    # Record the values
    final_y = test_y_all
    final_nodes = test_nodes_all

    final_predictions_dnn = test_predictions_dnn
    final_predictions_lstm = test_predictions_lstm

    print(f"Final y length: {len(final_y)}")
    print(f"Final nodes length: {len(final_nodes)}")
    print(f"Final DNN predictions length: {len(final_predictions_dnn)}")
    print(f"Final LSTM predictions length: {len(final_predictions_lstm)}")

    # Convert the dictionary to a pandas DataFrame
    final_y = pd.DataFrame(final_y, columns=["lambda", "mu", "cap"])
    final_predictions_dnn["num_nodes"] = final_nodes
    final_predictions_dnn = pd.DataFrame(final_predictions_dnn, columns=["lambda_pred_dnn", "mu_pred_dnn", "cap_pred_dnn"])
    final_predictions_lstm = pd.DataFrame(final_predictions_lstm, columns=["lambda_pred_lstm", "mu_pred_lstm", "cap_pred_lstm"])
    # Combine the dataframes except for model_performance
    final_data = pd.concat([final_y, final_predictions_dnn, final_predictions_lstm], axis=1)

    # Save the data to a file using pyreadr
    pyreadr.write_rds(os.path.join(name, "DDD_FREE_TES", "Bagging", f"DDD_POLY_TES_dnn_lstm_{gnn_depth}.rds"), final_data)


if __name__ == '__main__':
    main()
