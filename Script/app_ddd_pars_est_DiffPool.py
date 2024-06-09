import sys
import os
import pandas as pd
import pyreadr
import torch
import glob
import torch_geometric.transforms as T
import torch.nn.functional as F
import yaml
from math import ceil
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn import DenseSAGEConv as SAGEConv
from torch.nn.utils.rnn import pack_padded_sequence

# Load the global parameters from the config file
global_params = None

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

# Set max node for DDD_TES
model_max_node = 2147
max_brts_len = 1073


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
    name, _ = filename.rsplit('.', 1)  # Split at the last dot to separate the extension
    params = name.split('_')[1:]
    return params


def check_params_consistency(params_tree_list, params_el_list, params_st_list, params_bt_list):
    # Check if all corresponding elements in the four lists are equal
    is_consistent = all(
        a == b == c == d for a, b, c, d in zip(params_tree_list, params_el_list, params_st_list, params_bt_list))

    if is_consistent:
        print("Parameters are consistent across the tree, EL, ST, and BT datasets.")
    else:
        raise ValueError("Mismatch in parameters between the tree, EL, ST, and BT datasets.")

    return is_consistent


def check_file_consistency(files_tree, files_el, files_st, files_bt):
    # Check if the four lists have the same length
    if not (len(files_tree) == len(files_el) == len(files_st) == len(files_bt)):
        raise ValueError("Mismatched lengths among file lists.")

    # Define a function to extract parameters from filename
    def get_params_tuple(filename):
        # Extract the parameters as strings, not floats
        return tuple(filename.split('_')[1:-1])

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

        edge_length_tensor = torch.tensor(length_list[i].values, dtype=torch.float)

        params_current = params_list[i]

        stats_tensor = torch.tensor(stats_list[i].values, dtype=torch.float)

        stats_tensor = stats_tensor.squeeze(1)  # Remove the extra dimension

        brts_tensor = torch.tensor(brts_list[i].values, dtype=torch.float)

        brts_tensor = brts_tensor.squeeze(1)  # Remove the extra dimension

        brts_length = torch.tensor([len(brts_list[i].values)], dtype=torch.long)

        # Create a Data object with the edge index, number of nodes, and category value
        data = Data(x=edge_length_tensor,
                    edge_index=edge_index_tensor,
                    num_nodes=num_nodes,
                    stats=stats_tensor,
                    brts=brts_tensor,
                    brts_len=brts_length,
                    family=params_current[0],
                    tree=params_current[1])

        # Append the Data object to the list
        pytorch_geometric_data_list.append(data)

    # Exclude data with stat or brts == 0, as they are not ultrametric or binary trees
    filtered_data_list = [
        data for data in pytorch_geometric_data_list
        if data.stats.shape != torch.Size([1])
    ]

    return filtered_data_list


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <name> <depth>")
        sys.exit(1)

    name = sys.argv[1]
    depth = int(sys.argv[2])

    print("Applying pre-trained DiffPool + LSTM Boosting model to empirical trees...")

    full_dir = os.path.join(name, "EMP_DATA", "EXPORT")

    # Concatenate the base directory path with the set_i folder name
    full_dir_tree = os.path.join(full_dir, 'GNN', 'tree')
    full_dir_el = os.path.join(full_dir, 'GNN', 'tree', 'EL')
    full_dir_st = os.path.join(full_dir, 'GNN', 'tree', 'ST')
    full_dir_bt = os.path.join(full_dir, 'GNN', 'tree', 'BT')
    # Call read_rds_to_pytorch with the full directory path
    print(full_dir)
    # Check if the number of .rds files in the tree and el paths are equal
    rds_count = check_rds_files_count(full_dir_tree, full_dir_el, full_dir_st, full_dir_bt)
    print(f'There are: {rds_count} trees in the EMP folder.')
    print(f"Now reading the trees in EMP_DATA...")
    # Read the .rds files into a list of PyTorch Geometric Data objects
    current_dataset = read_rds_to_pytorch(full_dir, rds_count)
    filtered_emp_data = [data for data in current_dataset if data.edge_index.shape != torch.Size([2, 2])]
    filtered_emp_data = [data for data in filtered_emp_data if data.num_nodes <= max_nodes_limit]
    filtered_emp_data = [data for data in filtered_emp_data if data.edge_index.shape != torch.Size([2, 1])]

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

    emp_dataset = TreeData(root=None, data_list=filtered_emp_data, transform=T.ToDense(model_max_node))

    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels,
                     normalize=False):
            super(GNN, self).__init__()

            self.convs = torch.nn.ModuleList()
            self.bns = torch.nn.ModuleList()

            for i in range(depth):
                first_index = 0
                last_index = depth - 1

                if depth == 1:
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

    # The LSTM model class with variable number of layers to process variable-length branch time sequences
    class LSTM(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, lstm_depth=1, dropout_rate=0.5):
            super(LSTM, self).__init__()

            self.lstm = torch.nn.LSTM(input_size=in_channels, hidden_size=hidden_channels, num_layers=lstm_depth,
                                      batch_first=True, dropout=dropout_ratio if lstm_depth > 1 else 0)
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

    # Differential pooling model class
    # Multimodal architecture with GNNs, LSTMs, and DNNs
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
            num_nodes1 = ceil(diffpool_ratio * model_max_node)
            self.gnn1_pool = GNN(emp_dataset.num_node_features, gcn_layer1_hidden_channels, num_nodes1)
            self.gnn1_embed = GNN(emp_dataset.num_node_features, gcn_layer1_hidden_channels,
                                  gcn_layer2_hidden_channels)

            # DiffPool Layer 2
            num_nodes2 = ceil(diffpool_ratio * num_nodes1)
            gnn1_out_channels = gcn_layer1_hidden_channels * (depth - 1) + gcn_layer2_hidden_channels
            self.gnn2_pool = GNN(gnn1_out_channels, gcn_layer2_hidden_channels, num_nodes2)
            self.gnn2_embed = GNN(gnn1_out_channels, gcn_layer2_hidden_channels, gcn_layer3_hidden_channels)

            # DiffPool Layer 3
            gnn2_out_channels = gcn_layer2_hidden_channels * (depth - 1) + gcn_layer3_hidden_channels
            self.gnn3_embed = GNN(gnn2_out_channels, gcn_layer3_hidden_channels, lin_layer1_hidden_channels)
            gnn3_out_channels = gcn_layer3_hidden_channels * (depth - 1) + lin_layer1_hidden_channels

            # Final Readout Layers
            self.lin1 = torch.nn.Linear(gnn3_out_channels, lin_layer2_hidden_channels)
            self.lin2 = torch.nn.Linear(lin_layer2_hidden_channels, n_predicted_values)

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

            # Forward pass through the readout layers
            x = F.dropout(x, p=dropout_ratio, training=self.training)
            x = self.lin1(x)
            x = F.gelu(x)
            x = F.dropout(x, p=dropout_ratio, training=self.training)
            x = self.lin2(x)
            # x = F.relu(x)
            if self.verbose:
                print("Readout Layers Completed...")
                print("Forward Pass Completed...")
                print("Epoch Completed...")

            # Return the final output along with the DiffPool layer losses
            return x, l1 + l2, e1 + e2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating using {device}")

    # Load pre-trained GNN model state
    model_gnn = DiffPool()
    model_gnn = model_gnn.to(device)
    path_to_gnn_model = os.path.join(name, "DDD_FREE_TES", "STBO", f"DDD_FREE_TES_model_diffpool_{depth}_gnn.pt")
    model_gnn.load_state_dict(torch.load(path_to_gnn_model, map_location=device))

    # Load pre_trained LSTM model state
    model_lstm = LSTM(in_channels=1, hidden_channels=lstm_hidden_channels,
                      out_channels=lstm_output_channels, lstm_depth=lstm_depth).to(device)
    model_lstm = model_lstm.to(device)
    path_to_lstm_model = os.path.join(name, "DDD_FREE_TES", "STBO", f"DDD_FREE_TES_gnn_{depth}_model_lstm.pt")
    model_lstm.load_state_dict(torch.load(path_to_lstm_model, map_location=device))

    emp_loader = DenseDataLoader(emp_dataset, batch_size=1, shuffle=False)
    print(f"Empirical dataset length: {len(emp_loader.dataset)}")
    print(emp_loader.dataset.transform)

    print(model_gnn)
    print(model_lstm)

    print("Evaluating empirical trees using pre-trained models")

    predictions_before_lstm = torch.tensor([], dtype=torch.float, device=device)
    predictions_after_lstm = torch.tensor([], dtype=torch.float, device=device)
    num_nodes_original = torch.tensor([], dtype=torch.long, device=device)
    family_name = []
    tree_name = []

    with torch.no_grad():
        model_gnn.eval()
        model_lstm.eval()
        for data in emp_loader:
            data.to(device)
            predictions, _, _ = model_gnn(data.x, data.adj, data.mask)
            num_nodes_original = torch.cat((num_nodes_original, data.num_nodes), dim=0)
            family_name.append(data.family)
            tree_name.append(data.tree)
            predictions_before_lstm = torch.cat((predictions_before_lstm, predictions), dim=0)
            lengths_brts = torch.sum(data.brts != 0, dim=1).cpu().tolist()
            brts_cpu = data.brts.cpu()
            brts_cpu = brts_cpu.unsqueeze(-1)
            packed_brts = pack_padded_sequence(brts_cpu, lengths_brts, batch_first=True, enforce_sorted=False).to(
                device)
            predicted_residual = model_lstm(packed_brts)
            predictions_after_lstm = torch.cat((predictions_after_lstm, predictions + predicted_residual), dim=0)

    # Save the data for the LSTM compensation
    predictions_before_lstm = predictions_before_lstm.cpu().detach().numpy()
    predictions_after_lstm = predictions_after_lstm.cpu().detach().numpy()
    num_nodes_original = num_nodes_original.cpu().detach().numpy()

    emp_data_dict = {"pred_lambda_before": predictions_before_lstm[:, 0],
                       "pred_mu_before": predictions_before_lstm[:, 1],
                       "pred_cap_before": predictions_before_lstm[:, 2],
                       "pred_lambda_after": predictions_after_lstm[:, 0],
                       "pred_mu_after": predictions_after_lstm[:, 1],
                       "pred_cap_after": predictions_after_lstm[:, 2],
                        "family": family_name,
                        "tree": tree_name,
                       "num_nodes": num_nodes_original}

    emp_data_df = pd.DataFrame(emp_data_dict)

    # Workaround to get rid of the dtype incompatible issue
    emp_data_df = emp_data_df.astype(object)

    pyreadr.write_rds(os.path.join(name, "EMP_DATA", f"ddd_empirical_gnn_{depth}_lstm_result.rds"),
                      emp_data_df)


if __name__ == '__main__':
    main()
