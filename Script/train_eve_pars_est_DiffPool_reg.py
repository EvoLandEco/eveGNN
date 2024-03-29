import sys
import os
import pandas as pd
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
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool

# Load the global parameters from the config file
global_params = None

with open("../Config/eve_train_diffpool.yaml", "r") as ymlfile:
    global_params = yaml.safe_load(ymlfile)

# Set global variables
metric_to_category = {'pd': 0, 'ed': 1, 'nnd': 2}
beta_n_norm_factor = global_params["beta_n_norm_factor"]
beta_phi_norm_factor = global_params["beta_phi_norm_factor"]
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
n_predicted_values = global_params["n_predicted_values"]
batch_size_reduce_factor = global_params["batch_size_reduce_factor"]
max_nodes_limit = global_params["max_nodes_limit"]

# Check if metric_to_category is a dictionary with string keys and integer values
assert isinstance(metric_to_category, dict), "metric_to_category should be a dictionary"
for key, value in metric_to_category.items():
    assert isinstance(key, str), "All keys in metric_to_category should be strings"
    assert isinstance(value, int), "All values in metric_to_category should be integers"

# Check if beta_n_norm_factor and beta_phi_norm_factor are positive integers
assert isinstance(beta_n_norm_factor, int) and beta_n_norm_factor > 0, "beta_n_norm_factor should be a positive integer"
assert isinstance(beta_phi_norm_factor, int) and beta_phi_norm_factor > 0, "beta_phi_norm_factor should be a positive integer"

# Check if epoch_number is a positive integer
assert isinstance(epoch_number, int) and epoch_number > 0, "epoch_number should be a positive integer"


def read_table(path):
    return pd.read_csv(path, sep="\s+", header=0)  # assuming the tables are tab-delimited


def check_same_across_rows(df):
    return df.apply(lambda x: x.nunique() == 1)


def count_rds_files(path, metric):
    # Define a function to check if the file name matches the desired metric
    def is_metric_in_filename(filename):
        parts = filename.split('_')
        return parts[6] == metric

    # Get the list of .rds files that match the metric
    rds_files = [f for f in glob.glob(os.path.join(path, '*.rds')) if is_metric_in_filename(os.path.basename(f))]
    return len(rds_files)


def check_rds_files_count(tree_path, el_path, metric):
    # Count the number of .rds files in both paths that match the metric
    tree_count = count_rds_files(tree_path, metric)
    el_count = count_rds_files(el_path, metric)

    # Check if the counts are equal
    if tree_count == el_count:
        return tree_count
    else:
        raise ValueError("The number of .rds files in the two paths are not equal")


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


def check_file_consistency(files_tree, files_el):
    # Check if the two lists have the same length
    if len(files_tree) != len(files_el):
        raise ValueError("Mismatched lengths")

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
    for tree_file, el_file in zip(files_tree, files_el):
        tree_params = get_params_tuple(tree_file)
        el_params = get_params_tuple(el_file)
        if tree_params != el_params:
            raise ValueError(f"Mismatched parameters: {tree_file} vs {el_file}")

    # If we get here, all checks passed
    print("File lists consistency check passed")


def check_params_consistency(params_tree_list, params_el_list):
    is_consistent = all(a == b for a, b in zip(params_tree_list, params_el_list))
    if is_consistent:
        print("Parameters are consistent across the tree and EL datasets.")
    else:
        raise ValueError("Mismatch in parameters between the tree and EL datasets.")
    return is_consistent


def check_list_count(count, data_list, length_list, params_list):
    # Get the number of elements in each list
    data_count = len(data_list)
    length_count = len(length_list)
    params_count = len(params_list)

    # Check if the count matches the number of elements in each list
    if count != data_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, data_list has {data_count} elements.")

    if count != length_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, length_list has {length_count} elements.")

    if count != params_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, params_list has {params_count} elements.")

    # If all checks pass, print a success message
    print("Count check passed")


def read_rds_to_pytorch(path, count, metric):
    # Function to check if the filename contains the metric at the correct position
    def is_metric_in_filename(filename):
        parts = filename.split('_')
        # Check if the metric is at the expected position (index 6, 0-based indexing)
        return parts[6] == metric

    # Adjust the file selection logic
    files_tree = [f for f in os.listdir(os.path.join(path, 'GNN', 'tree'))
                  if f.startswith('tree_') and is_metric_in_filename(f) and f.endswith('.rds')]
    files_el = [f for f in os.listdir(os.path.join(path, 'GNN', 'tree', 'EL'))
                if f.startswith('EL_') and is_metric_in_filename(f) and f.endswith('.rds')]

    # Sort the files based on the parameters
    files_tree = sort_files(files_tree)
    files_el = sort_files(files_el)

    # Check if the files are consistent
    check_file_consistency(files_tree, files_el)

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

    check_params_consistency(params_tree_list, params_el_list)

    params_list = []

    for filename in files_tree:
        params = get_params(filename)
        params_list.append(params)

    # Normalize beta_n and beta_phi
    for vector in params_list:
        vector[2] = vector[2] * beta_n_norm_factor
        vector[3] = vector[3] * beta_phi_norm_factor

    check_list_count(count, data_list, length_list, params_list)

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

        params_current_tensor = torch.tensor(params_current[0:n_predicted_values], dtype=torch.float)

        # Create a Data object with the edge index, number of nodes, and category value
        data = Data(x=edge_length_tensor,
                    edge_index=edge_index_tensor,
                    num_nodes=num_nodes,
                    y=params_current_tensor)

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
        print(f"Usage: {sys.argv[0]} <name> <task_type> <metric>")
        sys.exit(1)

    name = sys.argv[1]
    task_type = sys.argv[2]
    metric = sys.argv[3]

    # Now you can use the variables name and set_i in your code
    print(f'Name: {name}, Task Type: {task_type}, Metric: {metric}')

    training_dataset_list = []
    testing_dataset_list = []

    # Concatenate the base directory path with the set_i folder name
    full_dir = os.path.join(name, task_type)
    full_dir_tree = os.path.join(full_dir, 'GNN', 'tree')
    full_dir_el = os.path.join(full_dir, 'GNN', 'tree', 'EL')
    # Call read_rds_to_pytorch with the full directory path
    print(full_dir)
    # Check if the number of .rds files in the tree and el paths are equal
    rds_count = check_rds_files_count(full_dir_tree, full_dir_el, metric)
    print(f'There are: {rds_count} trees in the {task_type} folder.')
    print(f"Now reading {task_type}...")
    # Read the .rds files into a list of PyTorch Geometric Data objects
    current_dataset = read_rds_to_pytorch(full_dir, rds_count, metric)
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
        raise ValueError("Invalid task type")

    full_val_dir_tree = os.path.join(val_dir, 'GNN', 'tree')
    full_val_dir_el = os.path.join(val_dir, 'GNN', 'tree', 'EL')
    val_rds_count = check_rds_files_count(full_val_dir_tree, full_val_dir_el, metric)
    print(f'There are: {val_rds_count} trees in the validation folder.')
    print(f"Now reading validation data...")
    current_val_dataset = read_rds_to_pytorch(val_dir, val_rds_count, metric)
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

    training_dataset = TreeData(root=None, data_list=filtered_training_data, transform=T.ToDense(max_nodes))
    testing_dataset = TreeData(root=None, data_list=filtered_testing_data, transform=T.ToDense(max_nodes))

    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels,
                     normalize=False, lin=True):
            super(GNN, self).__init__()

            self.convs = torch.nn.ModuleList()
            self.bns = torch.nn.ModuleList()

            self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

            self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
            self.bns.append(torch.nn.BatchNorm1d(out_channels))

        def forward(self, x, adj, mask=None):
            for step in range(len(self.convs)):
                x = F.relu(self.convs[step](x, adj, mask))
                x = torch.permute(x, (0, 2, 1))
                x = self.bns[step](x)
                x = torch.permute(x, (0, 2, 1))

            return x

    class DiffPool(torch.nn.Module):
        def __init__(self):
            super(DiffPool, self).__init__()

            num_nodes = ceil(diffpool_ratio * max_nodes)
            self.gnn1_pool = GNN(training_dataset.num_node_features, gcn_layer1_hidden_channels, num_nodes)
            self.gnn1_embed = GNN(training_dataset.num_node_features, gcn_layer1_hidden_channels, gcn_layer2_hidden_channels)

            num_nodes = ceil(diffpool_ratio * num_nodes)
            self.gnn2_pool = GNN(gcn_layer2_hidden_channels, gcn_layer2_hidden_channels, num_nodes)
            self.gnn2_embed = GNN(gcn_layer2_hidden_channels, gcn_layer2_hidden_channels, gcn_layer3_hidden_channels, lin=False)

            self.gnn3_embed = GNN(gcn_layer3_hidden_channels, gcn_layer3_hidden_channels, lin_layer1_hidden_channels, lin=False)

            # Layers for regression
            self.lin1 = torch.nn.Linear(lin_layer1_hidden_channels, lin_layer2_hidden_channels)
            self.lin2 = torch.nn.Linear(lin_layer2_hidden_channels, n_predicted_values)

        def forward(self, x, adj, mask=None):
            s = self.gnn1_pool(x, adj, mask)
            x = self.gnn1_embed(x, adj, mask)

            x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

            s = self.gnn2_pool(x, adj)
            x = self.gnn2_embed(x, adj)

            x, adj, l2, e2 = dense_diff_pool(x, adj, s)

            x = self.gnn3_embed(x, adj)

            x = x.mean(dim=1)

            xre = F.dropout(x, p=dropout_ratio, training=self.training)
            xre = self.lin1(xre)
            xre = F.relu(xre)
            xre = F.dropout(xre, p=dropout_ratio, training=self.training)
            xre = self.lin2(xre)
            # xre = F.relu(xre)

            return xre, l1 + l2, e1 + e2

    def train():
        model.train()

        loss_all = 0  # Keep track of the loss

        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()
            out_re, _, _ = model(data.x, data.adj, data.mask)
            target_re = data.y.view(data.num_nodes.__len__(), n_predicted_values).to(device)
            assert out_re.device == target_re.device, \
                "Error: Device mismatch between output and target tensors."
            loss = F.mse_loss(out_re, target_re)
            loss.backward()
            loss_all += loss.item() * data.num_nodes.__len__()
            optimizer.step()

        out_loss_all = loss_all / len(train_loader.dataset)

        return out_loss_all

    @torch.no_grad()
    def test_diff(loader):
        model.eval()

        diffs_all = torch.tensor([], dtype=torch.float, device=device)
        outputs_all = torch.tensor([], dtype=torch.float, device=device)  # To store all outputs
        y_all = torch.tensor([], dtype=torch.float, device=device)  # To store all y
        nodes_all = torch.tensor([], dtype=torch.long, device=device)  # To store all node numbers

        for data in loader:
            data.to(device)
            out_re, _, _ = model(data.x, data.adj, data.mask)
            diffs = torch.abs(out_re - data.y.view(data.num_nodes.__len__(), n_predicted_values))
            diffs_all = torch.cat((diffs_all, diffs), dim=0)
            outputs_all = torch.cat((outputs_all, out_re), dim=0)  # Concatenate the outputs
            y_all = torch.cat((y_all, data.y.view(data.num_nodes.__len__(), n_predicted_values)), dim=0)  # Concatenate the y
            nodes_all = torch.cat((nodes_all, data.num_nodes), dim=0)  # Concatenate the node numbers

        print(f"diffs_all length: {len(diffs_all)}; test_loader.dataset length: {len(test_loader.dataset)}; Equal: {len(diffs_all) == len(test_loader.dataset)}")
        mean_diffs = torch.sum(diffs_all, dim=0) / len(test_loader.dataset)

        return mean_diffs.cpu().detach().numpy(), diffs_all.cpu().detach().numpy(), outputs_all.cpu().detach().numpy(), y_all.cpu().detach().numpy(), nodes_all.cpu().detach().numpy()

    @torch.no_grad()
    def compute_test_loss():
        model.eval()  # Set the model to evaluation mode
        loss_all = 0  # Keep track of the loss
        for data in test_loader:
            data.to(device)
            out_re, _, _ = model(data.x, data.adj, data.mask)
            target_re = data.y.view(data.num_nodes.__len__(), n_predicted_values).to(device)
            loss = F.mse_loss(out_re, target_re)
            loss_all += loss.item() * data.num_nodes.__len__()

        return loss_all / len(train_loader.dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training using {device}")

    model = DiffPool()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    shape_check(training_dataset, max_nodes)
    shape_check(testing_dataset, max_nodes)

    train_loader = DenseDataLoader(training_dataset, batch_size=train_batch_size_adjusted, shuffle=False)
    test_loader = DenseDataLoader(testing_dataset, batch_size=test_batch_size_adjusted, shuffle=False)
    print(f"Training dataset length: {len(train_loader.dataset)}")
    print(f"Testing dataset length: {len(test_loader.dataset)}")
    print(train_loader.dataset.transform)
    print(test_loader.dataset.transform)

    print(model)

    train_loss_history = []
    test_loss_history = []
    test_mean_diffs_history = []
    final_test_diffs = []
    final_test_predictions = []
    final_test_y = []
    final_test_nodes = []

    # Paths for saving embeddings
    train_dir = os.path.join(name, task_type, "training")
    test_dir = os.path.join(name, task_type, "testing")

    # Check and create directories if not exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    train_test_ratio = len(train_loader.dataset) / len(test_loader.dataset)

    # Training loop
    for epoch in range(1, epoch_number):
        train_loss_all = train()
        test_loss_all = compute_test_loss()
        test_loss_all = test_loss_all * train_test_ratio
        test_mean_diffs, test_diffs_all, test_predictions, test_y, test_nodes_all = test_diff(test_loader)
        test_mean_diffs[2] = test_mean_diffs[2] / beta_n_norm_factor
        test_mean_diffs[3] = test_mean_diffs[3] / beta_phi_norm_factor
        print(f'Epoch: {epoch:03d}, Par 1 Mean Diff: {test_mean_diffs[0]:.4f}, Par 2 Mean Diff: {test_mean_diffs[1]:.4f}, Par 3 Mean Diff: {test_mean_diffs[2]:.4f}, Par 4 Mean Diff: {test_mean_diffs[3]:.4f}')
        print(f'Epoch: {epoch:03d}, Training Loss: {train_loss_all:.4f}, Testing Loss: {test_loss_all:.4f}')

        # Record the values
        train_loss_history.append(train_loss_all)
        test_loss_history.append(test_loss_all)
        test_mean_diffs_history.append(test_mean_diffs)
        final_test_diffs = test_diffs_all
        final_test_predictions = test_predictions
        final_test_y = test_y
        final_test_nodes = test_nodes_all
        print(f"Final test diffs length: {len(final_test_diffs)}")
        print(f"Final predictions length: {len(final_test_predictions)}")
        print(f"Final y length: {len(final_test_y)}")
        print(f"Final nodes length: {len(final_test_nodes)}")

    print("Finished training, saving model...")
    torch.save(model.state_dict(), os.path.join(name, task_type, f"{task_type}_{metric}_model_diffpool_reg.pt"))
    print("Model successfully saved to:")
    print(f"{task_type}_{metric}_model_diffpool_reg.pt")

    print("Saving training and testing performance...")
    # After the loop, create a dictionary to hold the data
    data_dict = {"lambda_diff": [], "mu_diff": [], "beta_n_diff": [], "beta_phi_diff": []}

    printed_vars = set()

    def move_to_cpu(data, var_name="unknown"):
        if isinstance(data, torch.Tensor) and data.device.type == "cuda":
            if var_name not in printed_vars:
                print(f"Moving tensor in {var_name} to CPU from:", data.device)
                printed_vars.add(var_name)
            return data.cpu().detach()
        return data

    for array in test_mean_diffs_history:
        # Use the helper function to ensure data is on the CPU
        array = [move_to_cpu(item, var_name=name) for item, name in zip(array, ["lambda_diff", "mu_diff", "beta_n_diff", "beta_phi_diff"])]

        # It's assumed that the order of elements in the array corresponds to the keys in data_dict
        data_dict["lambda_diff"].append(array[0])
        data_dict["mu_diff"].append(array[1])
        data_dict["beta_n_diff"].append(array[2])
        data_dict["beta_phi_diff"].append(array[3])

    # Similarly, ensure that the other lists are on the CPU before assigning to the dictionary
    data_dict["Epoch"] = list(range(1, epoch_number))
    data_dict["Train_Loss_All"] = [move_to_cpu(item, var_name="Train_Loss_All") for item in train_loss_history]
    data_dict["Test_Loss_All"] = [move_to_cpu(item, var_name="Test_Loss_All") for item in test_loss_history]

    def check_col_lengths(data_dict):
        # Get the lengths of all columns
        lengths = [len(v) for v in data_dict.values()]

        # Check if all lengths are the same
        if len(set(lengths)) != 1:
            inconsistent_cols = [key for key, value in data_dict.items() if len(value) != lengths[0]]
            return False, inconsistent_cols
        return True, []

    model_performance = None
    final_differences = None
    final_predictions = None
    final_y = None

    # Convert the dictionary to a pandas DataFrame
    try:
        model_performance = pd.DataFrame(data_dict)
    except Exception as e:
        consistent, cols = check_col_lengths(data_dict)
        if not consistent:
            print("Error due to inconsistent lengths in columns:", ", ".join(cols))
        else:
            print("An unknown error occurred:", str(e))

    try:
        final_differences = pd.DataFrame(final_test_diffs, columns=["lambda_diff", "mu_diff", "beta_n_diff", "beta_phi_diff"])
        final_differences["num_nodes"] = final_test_nodes
    except Exception as e:
        print("Error occurred while creating the final_differences DataFrame:", str(e))

    try:
        final_predictions = pd.DataFrame(final_test_predictions, columns=["lambda_pred", "mu_pred", "beta_n_pred", "beta_phi_pred"])
    except Exception as e:
        print("Error occurred while creating the final_predictions DataFrame:", str(e))

    try:
        final_y = pd.DataFrame(final_test_y, columns=["lambda", "mu", "beta_n", "beta_phi"])
    except Exception as e:
        print("Error occurred while creating the final_y DataFrame:", str(e))

    # Check if variables exist before saving
    if 'model_performance' in locals():
        try:
            pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_{metric}_diffpool_reg.rds"), model_performance)
            print(f"Successfully saved training and testing performance to files:")
            print(f"{task_type}_{metric}_diffpool_reg.rds")
        except Exception as e:
            print(f"Error occurred while saving model_performance: {str(e)}")
    else:
        print("model_performance has not been assigned!")

    if 'final_differences' in locals():
        try:
            pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_{metric}_final_diffs_diffpool_reg.rds"), final_differences)
            print(f"Successfully saved final mean differences to files:")
            print(f"{task_type}_{metric}_final_diffs_diffpool_reg.rds")
        except Exception as e:
            print(f"Error occurred while saving final_differences: {str(e)}")
    else:
        print("final_differences has not been assigned!")

    if 'final_predictions' in locals():
        try:
            pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_{metric}_final_predictions_diffpool_reg.rds"), final_predictions)
            print(f"Successfully saved final predictions to files:")
            print(f"{task_type}_{metric}_final_predictions_diffpool_reg.rds")
        except Exception as e:
            print(f"Error occurred while saving final_predictions: {str(e)}")

    if 'final_y' in locals():
        try:
            pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_{metric}_final_y_diffpool_reg.rds"), final_y)
            print(f"Successfully saved final y to files:")
            print(f"{task_type}_{metric}_final_y_diffpool_reg.rds")
        except Exception as e:
            print(f"Error occurred while saving final_y: {str(e)}")


if __name__ == '__main__':
    main()
