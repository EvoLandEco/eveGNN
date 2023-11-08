import sys
import os
import pandas as pd
import pyreadr
import torch
import glob
import functools
import torch_geometric.transforms as T
import torch.nn.functional as F
import optuna
from optuna.study import Study
from optuna.trial import FrozenTrial
from math import ceil
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool


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

    # Normalize carrying capacity by dividing by 1000
    for vector in params_list:
        vector[2] = vector[2] / 1000

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

        params_current_tensor = torch.tensor(params_current[0:3], dtype=torch.float)

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
        def __init__(self, trial):
            super(DiffPool, self).__init__()

            diff_ratio = trial.suggest_float("diff_ratio", 0.1, 0.5)
            drop_ratio = trial.suggest_float("drop_ratio", 0.1, 0.5)

            num_nodes = ceil(diff_ratio * max_nodes)
            self.gnn1_pool = GNN(training_dataset.num_node_features, 256, num_nodes)
            self.gnn1_embed = GNN(training_dataset.num_node_features, 256, 256)

            num_nodes = ceil(diff_ratio * num_nodes)
            self.gnn2_pool = GNN(256, 128, num_nodes)
            self.gnn2_embed = GNN(256, 128, 128, lin=False)

            self.gnn3_embed = GNN(128, 64, 64, lin=False)

            self.lin1 = torch.nn.Linear(64, 32)
            self.lin2 = torch.nn.Linear(32, 3)

            # Store drop_ratio for use in the forward pass
            self.drop_ratio = drop_ratio

        def forward(self, x, adj, mask=None):
            s = self.gnn1_pool(x, adj, mask)
            x = self.gnn1_embed(x, adj, mask)

            x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

            s = self.gnn2_pool(x, adj)
            x = self.gnn2_embed(x, adj)

            x, adj, l2, e2 = dense_diff_pool(x, adj, s)

            x = self.gnn3_embed(x, adj)

            x = x.mean(dim=1)

            x = F.dropout(x, p=self.drop_ratio, training=self.training)
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_ratio, training=self.training)
            x = self.lin2(x)
            return x, l1 + l2, e1 + e2

    def train(model, optimizer, criterion, loader):
        model.train()

        loss_all = 0  # Keep track of the loss
        for data in loader:
            data.to(device)
            optimizer.zero_grad()
            out, _, _ = model(data.x, data.adj, data.mask)
            loss = criterion(out, data.y.view(data.num_nodes.__len__(), 3))
            loss.backward()
            loss_all += loss.item() * data.num_nodes.__len__()
            optimizer.step()

        return loss_all / len(loader.dataset)

    @torch.no_grad()
    def test_diff(model, loader):
        model.eval()

        diffs_all = torch.tensor([], dtype=torch.float, device=device)

        for data in loader:
            data.to(device)
            out, _, _ = model(data.x, data.adj, data.mask)
            diffs = torch.abs(out - data.y.view(data.num_nodes.__len__(), 3))
            diffs_all = torch.cat((diffs_all, diffs), dim=0)

        print(f"diffs_all length: {len(diffs_all)}; test_loader.dataset length: {len(test_loader.dataset)}; Equal: {len(diffs_all) == len(test_loader.dataset)}")
        mean_diffs = torch.sum(diffs_all, dim=0) / len(test_loader.dataset)
        return mean_diffs.cpu().detach().numpy(), diffs_all.cpu().detach().numpy()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training using {device}")

    def shape_check(dataset, max_nodes):
        incorrect_shapes = []  # List to store indices of data elements with incorrect shapes
        for i in range(len(dataset)):
            data = dataset[i]
            # Check the shapes of data.x, data.adj, and data.mask
            if data.x.shape != torch.Size([max_nodes, 3]) or \
                    data.y.shape != torch.Size([3]) or \
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

    def objective(trial):
        tr_loader = train_loader
        vl_loader = test_loader
        model = DiffPool(trial).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        )
        criterion = torch.nn.MSELoss().to(device)

        for epoch in range(20):
            train(model, optimizer, criterion, tr_loader)
            diffs, _ = test_diff(model, vl_loader)
            lambda_diff = diffs[0]
            mu_diff = diffs[1]
            cap_diff = diffs[2]

            return lambda_diff, mu_diff, cap_diff

    study = optuna.create_study(directions=["minimize", "minimize", "minimize"])
    study.optimize(objective, timeout=86400, gc_after_trial=True)
    print("Number of finished trials: ", len(study.trials))
    trials_df = study.trials_dataframe()
    print("Saving trials dataframe to RDS...")
    pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_diffpool_trials.rds"), trials_df)


if __name__ == '__main__':
    main()
