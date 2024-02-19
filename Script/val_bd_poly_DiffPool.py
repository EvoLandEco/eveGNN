import sys
import os
import pandas as pd
import pyreadr
import torch
import glob
import functools
import random
import yaml
import torch_geometric.transforms as T
import torch.nn.functional as F
from math import ceil
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool


global_params = None
global_params_train = None

with open("../Config/bd_val_diffpool.yaml", "r") as ymlfile:
    global_params = yaml.safe_load(ymlfile)

with open("../Config/bd_train_diffpool.yaml", "r") as ymlfile:
    global_params_train = yaml.safe_load(ymlfile)

# Set global variables
epoch_number = global_params["epoch_number"]
diffpool_ratio = global_params["diffpool_ratio"]
dropout_ratio = global_params["dropout_ratio"]
learning_rate = global_params["learning_rate"]
val_batch_size = global_params["val_batch_size"]
gcn_layer1_hidden_channels = global_params["gcn_layer1_hidden_channels"]
gcn_layer2_hidden_channels = global_params["gcn_layer2_hidden_channels"]
gcn_layer3_hidden_channels = global_params["gcn_layer3_hidden_channels"]
lin_layer1_hidden_channels = global_params["lin_layer1_hidden_channels"]
lin_layer2_hidden_channels = global_params["lin_layer2_hidden_channels"]
n_predicted_values = global_params["n_predicted_values"]
batch_size_reduce_factor = global_params["batch_size_reduce_factor"]
max_nodes_limit = global_params["max_nodes_limit"]

# Set global variables from training configuration
max_gnn_depth = int(global_params_train["max_gnn_depth"])


def read_table(path):
    return pd.read_csv(path, sep="\s+", header=0)  # assuming the tables are tab-delimited


def check_same_across_rows(df):
    return df.apply(lambda x: x.nunique() == 1)


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


def check_file_consistency(files_tree, files_el):
    # Check if the two lists have the same length
    if len(files_tree) != len(files_el):
        raise ValueError("Mismatched lengths")

    # Define a function to extract parameters from filename
    def get_params_tuple(filename):
        return tuple(map(float, filename.split('_')[1:-1]))

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


def read_rds_to_pytorch(path, count):
    # List all files in the directory
    files_tree = [f for f in os.listdir(os.path.join(path, 'GNN', 'tree'))
                  if f.startswith('tree_') and f.endswith('.rds')]
    files_el = [f for f in os.listdir(os.path.join(path, 'GNN', 'tree', 'EL'))
                if f.startswith('EL_') and f.endswith('.rds')]

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
        print(f"Usage: {sys.argv[0]} <name> <task_type>")
        sys.exit(1)

    name = sys.argv[1]
    task_type = sys.argv[2]

    # Now you can use the variables name and set_i in your code
    print(f'Name: {name}, Task Type: {task_type}')

    training_dataset_list = []
    testing_dataset_list = []

    full_dir = os.path.join(name, "BD_FREE_TES")
    val_dir = os.path.join(name, "BD_VAL_TES")
    poly_dir = os.path.join(name, "BD_POLY_TES")
    model_type = "BD_FREE_TES"
    val_batch_size_adjusted = int(val_batch_size)

    # Concatenate the base directory path with the set_i folder name
    full_dir_tree = os.path.join(full_dir, 'GNN', 'tree')
    full_dir_el = os.path.join(full_dir, 'GNN', 'tree', 'EL')
    # Call read_rds_to_pytorch with the full directory path
    print(full_dir)
    # Check if the number of .rds files in the tree and el paths are equal
    rds_count = check_rds_files_count(full_dir_tree, full_dir_el)
    print(f'There are: {rds_count} trees in the {task_type} folder.')
    print(f"Now reading {task_type}...")
    # Read the .rds files into a list of PyTorch Geometric Data objects
    current_dataset = read_rds_to_pytorch(full_dir, rds_count)
    # Shuffle the data
    current_dataset = shuffle_data(current_dataset)
    current_training_data = get_training_data(current_dataset)
    current_testing_data = get_testing_data(current_dataset)
    training_dataset_list.append(current_training_data)
    testing_dataset_list.append(current_testing_data)

    validation_dataset_list = []

    full_val_dir_tree = os.path.join(val_dir, 'GNN', 'tree')
    full_val_dir_el = os.path.join(val_dir, 'GNN', 'tree', 'EL')
    val_rds_count = check_rds_files_count(full_val_dir_tree, full_val_dir_el)
    print(f'There are: {val_rds_count} trees in the validation folder.')
    print(f"Now reading validation data...")
    current_val_dataset = read_rds_to_pytorch(val_dir, val_rds_count)
    validation_dataset_list.append(current_val_dataset)

    polymorph_dataset_list = []

    full_poly_dir_tree = os.path.join(poly_dir, 'GNN', 'tree')
    full_poly_dir_el = os.path.join(poly_dir, 'GNN', 'tree', 'EL')
    poly_rds_count = check_rds_files_count(full_poly_dir_tree, full_poly_dir_el)
    print(f'There are: {poly_rds_count} trees in the polymorph folder.')
    print(f"Now reading polymorph data...")
    current_poly_dataset = read_rds_to_pytorch(poly_dir, poly_rds_count)
    polymorph_dataset_list.append(current_poly_dataset)

    sum_training_data = functools.reduce(lambda x, y: x + y, training_dataset_list)
    sum_testing_data = functools.reduce(lambda x, y: x + y, testing_dataset_list)
    sum_validation_data = functools.reduce(lambda x, y: x + y, validation_dataset_list)
    sum_polymorph_data = functools.reduce(lambda x, y: x + y, polymorph_dataset_list)

    # Filtering out trees with only 3 nodes
    # They might cause problems with ToDense
    filtered_training_data = [data for data in sum_training_data if data.edge_index.shape != torch.Size([2, 2])]
    filtered_testing_data = [data for data in sum_testing_data if data.edge_index.shape != torch.Size([2, 2])]
    filtered_validation_data = [data for data in sum_validation_data if data.edge_index.shape != torch.Size([2, 2])]
    filtered_polymorph_data = [data for data in sum_polymorph_data if data.edge_index.shape != torch.Size([2, 2])]

    # Filtering out trees with more than 3000 nodes
    filtered_training_data = [data for data in filtered_training_data if data.num_nodes <= max_nodes_limit]
    filtered_testing_data = [data for data in filtered_testing_data if data.num_nodes <= max_nodes_limit]
    filtered_validation_data = [data for data in filtered_validation_data if data.num_nodes <= max_nodes_limit]
    filtered_polymorph_data = [data for data in filtered_polymorph_data if data.num_nodes <= max_nodes_limit]

    # For BD trees, there might be possibility that they only have one edge. Should filter them out.
    filtered_training_data = [data for data in filtered_training_data if data.edge_index.shape != torch.Size([2, 1])]
    filtered_testing_data = [data for data in filtered_testing_data if data.edge_index.shape != torch.Size([2, 1])]
    filtered_validation_data = [data for data in filtered_validation_data if data.edge_index.shape != torch.Size([2, 1])]
    filtered_polymorph_data = [data for data in filtered_polymorph_data if data.edge_index.shape != torch.Size([2, 1])]

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
    max_nodes_poly = max([data.num_nodes for data in filtered_polymorph_data])
    max_nodes = max(max_nodes_train, max_nodes_test, max_nodes_val)
    max_nodes2 = max(max_nodes, max_nodes_poly)

    if max_nodes2 != max_nodes:
        raise ValueError("Max node number allowed exceeded.")

    print(f"Max nodes: {max_nodes} for {task_type}")
    polymorph_dataset = TreeData(root=None, data_list=filtered_polymorph_data, transform=T.ToDense(max_nodes))

    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels,
                     normalize=False, gnn_depth=1):
            super(GNN, self).__init__()

            self.convs = torch.nn.ModuleList()
            self.bns = torch.nn.ModuleList()

            for i in range(gnn_depth):
                first_index = 0
                last_index = gnn_depth - 1

                if gnn_depth == 1:
                    self.convs.append(GCNConv(in_channels, out_channels, normalize))
                    self.bns.append(torch.nn.BatchNorm1d(out_channels))
                else:
                    if i == first_index:
                        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
                        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
                    elif i == last_index:
                        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
                        self.bns.append(torch.nn.BatchNorm1d(out_channels))
                    else:
                        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
                        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        def forward(self, x, adj, mask=None):
            for step in range(len(self.convs)):
                x = F.relu(self.convs[step](x, adj, mask))
                x = torch.permute(x, (0, 2, 1))
                x = self.bns[step](x)
                x = torch.permute(x, (0, 2, 1))

            return x

    class DiffPool(torch.nn.Module):
        def __init__(self, gnn_depth=1):
            super(DiffPool, self).__init__()

            num_nodes = ceil(diffpool_ratio * max_nodes)
            self.gnn1_pool = GNN(polymorph_dataset.num_node_features, gcn_layer1_hidden_channels, num_nodes, gnn_depth=gnn_depth)
            self.gnn1_embed = GNN(polymorph_dataset.num_node_features, gcn_layer1_hidden_channels, gcn_layer2_hidden_channels, gnn_depth=gnn_depth)

            num_nodes = ceil(diffpool_ratio * num_nodes)
            self.gnn2_pool = GNN(gcn_layer2_hidden_channels, gcn_layer2_hidden_channels, num_nodes, gnn_depth=gnn_depth)
            self.gnn2_embed = GNN(gcn_layer2_hidden_channels, gcn_layer2_hidden_channels, gcn_layer3_hidden_channels, gnn_depth=gnn_depth)

            self.gnn3_embed = GNN(gcn_layer3_hidden_channels, gcn_layer3_hidden_channels, lin_layer1_hidden_channels, gnn_depth=gnn_depth)

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

            x = F.dropout(x, p=dropout_ratio, training=self.training)
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, p=dropout_ratio, training=self.training)
            x = self.lin2(x)
            # x = F.relu(x)

            return x, l1 + l2, e1 + e2

    @torch.no_grad()
    def test_diff(loader):
        model.eval()

        diffs_all = torch.tensor([], dtype=torch.float, device=device)
        outputs_all = torch.tensor([], dtype=torch.float, device=device)  # To store all outputs
        y_all = torch.tensor([], dtype=torch.float, device=device)  # To store all y
        nodes_all = torch.tensor([], dtype=torch.long, device=device)

        for data in loader:
            data.to(device)
            out, _, _ = model(data.x, data.adj, data.mask)
            diffs = torch.abs(out - data.y.view(data.num_nodes.__len__(), n_predicted_values))
            print(out.shape)
            print(data.y.view(data.num_nodes.__len__(), n_predicted_values).shape)
            diffs_all = torch.cat((diffs_all, diffs), dim=0)
            outputs_all = torch.cat((outputs_all, out), dim=0)
            y_all = torch.cat((y_all, data.y.view(data.num_nodes.__len__(), n_predicted_values)), dim=0)
            nodes_all = torch.cat((nodes_all, data.num_nodes), dim=0)

        print(f"diffs_all length: {len(diffs_all)}; test_loader.dataset length: {len(loader.dataset)}; Equal: {len(diffs_all) == len(loader.dataset)}")
        mean_diffs = torch.sum(diffs_all, dim=0) / len(loader.dataset)
        return mean_diffs.cpu().detach().numpy(), diffs_all.cpu().detach().numpy(), outputs_all.cpu().detach().numpy(), y_all.cpu().detach().numpy(), nodes_all.cpu().detach().numpy()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Validation using {device}")

    for i in range(1, max_gnn_depth + 1):
        model = DiffPool(gnn_depth=i)
        path_to_saved_model = os.path.join(name, model_type, f"{model_type}_model_diffpool_{i}.pt")
        model.load_state_dict(torch.load(path_to_saved_model))

        model = model.to(device)

        poly_loader = DenseDataLoader(polymorph_dataset, batch_size=val_batch_size_adjusted, shuffle=False)
        print(f"Validation dataset length: {len(poly_loader.dataset)}")
        print(poly_loader.dataset.transform)

        print(model)

        test_mean_diffs, test_diffs_all, test_predictions, test_y, test_nodes_all = test_diff(poly_loader)
        print(f'Par 1 Mean Diff: {test_mean_diffs[0]:.4f}, Par 2 Mean Diff: {test_mean_diffs[1]:.4f}')
        print(f"Final test diffs length: {len(test_diffs_all)}")

        # Convert the dictionary to a pandas DataFrame
        final_differences = pd.DataFrame(test_diffs_all, columns=["lambda_diff", "mu_diff"])
        final_predictions = pd.DataFrame(test_predictions, columns=["lambda_pred", "mu_pred"])
        final_y = pd.DataFrame(test_y, columns=["lambda", "mu"])
        final_differences["nodes"] = test_nodes_all
        # Save the data to a file using pyreadr
        pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_final_diffs_diffpool_{i}.rds"), final_differences)
        pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_final_predictions_diffpool_{i}.rds"), final_predictions)
        pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_final_y_diffpool_{i}.rds"), final_y)


if __name__ == '__main__':
    main()
