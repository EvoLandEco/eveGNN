import sys
import os
import pandas as pd
import numpy as np
import pyreadr
import torch
import glob
import functools
import torch_geometric.transforms as T
import torch.nn.functional as F
from math import ceil
from torch.nn import Linear
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool

# Defining a mapping from metric names to category values
metric_to_category = {'pd': 0, 'ed': 1, 'nnd': 2}
beta_n_norm_factor = 100
beta_phi_norm_factor = 1000


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

        params_current_tensor = torch.tensor(params_current[0:4], dtype=torch.float)

        class_current_tensor = torch.tensor(params_current[5], dtype=torch.long)

        # Create a Data object with the edge index, number of nodes, and category value
        data = Data(x=edge_length_tensor,
                    edge_index=edge_index_tensor,
                    num_nodes=num_nodes,
                    y_re=params_current_tensor,
                    y_cl=class_current_tensor)

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

    # Concatenate the base directory path with the set_i folder name
    full_dir = os.path.join(name, task_type)
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
    current_training_data = get_training_data(current_dataset)
    current_testing_data = get_testing_data(current_dataset)
    training_dataset_list.append(current_training_data)
    testing_dataset_list.append(current_testing_data)

    sum_training_data = functools.reduce(lambda x, y: x + y, training_dataset_list)
    sum_testing_data = functools.reduce(lambda x, y: x + y, testing_dataset_list)
    # Filtering out elements with None in edge_index
    filtered_training_data = [data for data in sum_training_data if data.edge_index.shape != torch.Size([2, 2])]
    filtered_testing_data = [data for data in sum_testing_data if data.edge_index.shape != torch.Size([2, 2])]

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
    max_nodes = max(max_nodes_train, max_nodes_test)

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
            batch_size, num_nodes, in_channels = x.size()

            for step in range(len(self.convs)):
                x = F.relu(self.convs[step](x, adj, mask))
                x = torch.permute(x, (0, 2, 1))
                x = self.bns[step](x)
                x = torch.permute(x, (0, 2, 1))

            return x

    class DiffPool(torch.nn.Module):
        def __init__(self):
            super(DiffPool, self).__init__()

            num_nodes = ceil(0.25 * max_nodes)
            self.gnn1_pool = GNN(training_dataset.num_node_features, 64, num_nodes)
            self.gnn1_embed = GNN(training_dataset.num_node_features, 64, 64)

            num_nodes = ceil(0.25 * num_nodes)
            self.gnn2_pool = GNN(64, 64, num_nodes)
            self.gnn2_embed = GNN(64, 64, 64, lin=False)

            self.gnn3_embed = GNN(64, 64, 64, lin=False)

            # Layers for regression
            self.lin1re = torch.nn.Linear(64, 64)
            self.lin2re = torch.nn.Linear(64, 4)

            # Layers for classification
            self.lin1cl = torch.nn.Linear(64, 64)
            self.lin2cl = torch.nn.Linear(64, 3)

        def forward(self, x, adj, mask=None):
            s = self.gnn1_pool(x, adj, mask)
            x = self.gnn1_embed(x, adj, mask)

            x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

            s = self.gnn2_pool(x, adj)
            x = self.gnn2_embed(x, adj)

            x, adj, l2, e2 = dense_diff_pool(x, adj, s)

            x = self.gnn3_embed(x, adj)

            x = x.mean(dim=1)

            xre = F.relu(self.lin1re(x))
            xre = self.lin2re(xre)

            xcl = F.relu(self.lin1cl(x))
            xcl = self.lin2cl(xcl)

            return xre, F.log_softmax(xcl, dim=-1)

    def combined_loss(output_regression, target_regression, output_classification, target_classification):
        loss_regression = F.mse_loss(output_regression, target_regression)
        loss_classification = F.cross_entropy(output_classification, target_classification)
        return loss_regression + loss_classification

    def train():
        model.train()

        loss_all = 0  # Keep track of the loss
        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()
            out_re, out_cl = model(data.x, data.adj, data.mask)
            loss = combined_loss(out_re, data.y_re.view(data.num_nodes.__len__(), 4), out_cl, data.y_cl.argmax(dim=1))
            loss.backward()
            loss_all += loss.item() * data.num_nodes.__len__()
            optimizer.step()

        return loss_all / len(train_loader.dataset)

    @torch.no_grad()
    def test_diff(loader):
        model.eval()

        diffs_all = torch.tensor([], dtype=torch.float, device=device)

        for data in loader:
            data.to(device)
            out_re, _ = model(data.x, data.adj, data.mask)
            diffs = torch.abs(out_re - data.y_re.view(data.num_nodes.__len__(), 4))
            diffs_all = torch.cat((diffs_all, diffs), dim=0)

        print(f"diffs_all length: {len(diffs_all)}; test_loader.dataset length: {len(test_loader.dataset)}; Equal: {len(diffs_all) == len(test_loader.dataset)}")
        mean_diffs = torch.sum(diffs_all, dim=0) / len(test_loader.dataset)
        return mean_diffs.cpu().detach().numpy(), diffs_all.cpu().detach().numpy()

    @torch.no_grad()
    def test_accu(loader):
        model.eval()
        correct = 0

        for data in loader:
            data = data.to(device)
            _, out_cl = model(data.x, data.adj, data.mask)
            pred = out_cl.max(dim=1)[1]
            correct += pred.eq(data.y_cl.view(-1)).sum().item()

        accuracy = correct / len(loader.dataset)
        return accuracy

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training using {device}")

    model = DiffPool()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def shape_check(dataset, max_nodes):
        incorrect_shapes = []  # List to store indices of data elements with incorrect shapes
        for i in range(len(dataset)):
            data = dataset[i]
            # Check the shapes of data.x, data.adj, and data.mask
            if data.x.shape != torch.Size([max_nodes, 3]) or \
                    data.y_re.shape != torch.Size([4]) or \
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

    train_loader = DenseDataLoader(training_dataset, batch_size=64, shuffle=False)
    test_loader = DenseDataLoader(testing_dataset, batch_size=64, shuffle=False)
    print(f"Training dataset length: {len(train_loader.dataset)}")
    print(f"Testing dataset length: {len(test_loader.dataset)}")
    print(train_loader.dataset.transform)
    print(test_loader.dataset.transform)

    print(model)

    test_mean_diffs_history = []
    train_loss_history = []
    test_accuracy_history = []
    final_test_diffs = []

    train_dir = os.path.join(name, task_type, "training")
    test_dir = os.path.join(name, task_type, "testing")

    # Check and create directories if not exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for epoch in range(1, 100):
        train_loss_all = train()
        test_mean_diffs, test_diffs_all = test_diff(test_loader)
        test_accuracy = test_accu(test_loader)
        test_mean_diffs[2] = test_mean_diffs[2] / beta_n_norm_factor
        test_mean_diffs[3] = test_mean_diffs[3] / beta_phi_norm_factor
        print(f'Epoch: {epoch:03d}, Par 1 Mean Diff: {test_mean_diffs[0]:.4f}, Par 2 Mean Diff: {test_mean_diffs[1]:.4f}, Par 3 Mean Diff: {test_mean_diffs[2]:.4f}, Par 4 Mean Diff: {test_mean_diffs[3]:.4f}, Train Loss: {train_loss_all:.4f}')
        print(f'Epoch: {epoch:03d}, Classification Accuracy: {test_accuracy:.4f}')

        # Record the values
        test_mean_diffs_history.append(test_mean_diffs)
        train_loss_history.append(train_loss_all)
        test_accuracy_history.append(test_accuracy)
        final_test_diffs = test_diffs_all
        print(f"Final test diffs length: {len(final_test_diffs)}")

    # After the loop, create a dictionary to hold the data
    data_dict = {"lambda_diff": [], "mu_diff": [], "beta_n_diff": [], "beta_phi_diff": []}
    # Iterate through test_mean_diffs_history
    for array in test_mean_diffs_history:
        # It's assumed that the order of elements in the array corresponds to the keys in data_dict
        data_dict["lambda_diff"].append(array[0])
        data_dict["mu_diff"].append(array[1])
        data_dict["beta_n_diff"].append(array[2])
        data_dict["beta_phi_diff"].append(array[3])
    data_dict["Epoch"] = list(range(1, 100))
    data_dict["Train_Loss"] = train_loss_history
    data_dict["Cl_Accuracy"] = test_accuracy_history

    # Convert the dictionary to a pandas DataFrame
    model_performance = pd.DataFrame(data_dict)
    final_differences = pd.DataFrame(final_test_diffs, columns=["lambda_diff", "mu_diff", "beta_n_diff", "beta_phi_diff"])
    # Save the data to a file using pyreadr
    pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_diffpool.rds"), model_performance)
    pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_final_diffs_diffpool.rds"), final_differences)

    torch.save(model.state_dict(), os.path.join(name, task_type, f"{task_type}_model_diffpool.pt"))


if __name__ == '__main__':
    main()
