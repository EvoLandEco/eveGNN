import sys
import os
import pandas as pd
import pyreadr
import glob
import functools
import torch_geometric.transforms as T
import random
from torch_geometric.data import InMemoryDataset, Data
from typing import Any, Dict, Optional

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention

# Global variables
epoch_number = 21
cap_norm_factor = 1000
attn_kwargs = {'dropout': 0.5}
attn_type = 'performer'


def read_table(path):
    return pd.read_csv(path, sep="\s+", header=0)  # assuming the tables are tab-delimited


def check_same_across_rows(df):
    return df.apply(lambda x: x.nunique() == 1)


def count_rds_files(path):
    # Get the list of .rds files in the specified path
    rds_files = glob.glob(os.path.join(path, '*.rds'))
    return len(rds_files)


def check_rds_files_count(tree_path, node_path, edge_path):
    # Count the number of .rds files in both paths
    tree_count = count_rds_files(tree_path)
    node_count = count_rds_files(node_path)
    edge_count = count_rds_files(edge_path)

    # Check if the counts are equal
    if tree_count == node_count == edge_count:
        return tree_count
    else:
        raise ValueError("The number of .rds files in the paths are not equal")


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


def check_file_consistency(files_tree, files_node, files_edge):
    # Check if the three lists have the same length
    if not (len(files_tree) == len(files_node) == len(files_edge)):
        raise ValueError("Mismatched lengths among file lists")

    # Define a function to extract parameters from filename
    def get_params_tuple(filename):
        return tuple(map(float, filename.split('_')[1:-1]))

    # Check each triplet of files for matching parameters
    for tree_file, node_file, edge_file in zip(files_tree, files_node, files_edge):
        tree_params = get_params_tuple(tree_file)
        node_params = get_params_tuple(node_file)
        edge_params = get_params_tuple(edge_file)

        if not (tree_params == node_params == edge_params):
            raise ValueError(f"Mismatched parameters: {tree_file}, {node_file}, {edge_file}")

    # If we get here, all checks passed
    print("File lists consistency check passed")


def check_params_consistency(params_tree_list, params_node_list, params_edge_list):
    is_consistent = all(a == b == c for a, b, c in zip(params_tree_list, params_node_list, params_edge_list))

    if is_consistent:
        print("Parameters are consistent across the tree, node, and edge datasets.")
    else:
        raise ValueError("Mismatch in parameters among the tree, node, and edge datasets.")

    return is_consistent


def check_list_count(count, data_list, node_list, edge_list, params_list):
    # Get the number of elements in each list
    data_count = len(data_list)
    node_count = len(node_list)
    edge_count = len(edge_list)
    params_count = len(params_list)

    # Check if the count matches the number of elements in each list
    if count != data_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, data_list has {data_count} elements.")

    if count != node_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, node_list has {node_count} elements.")

    if count != edge_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, edge_list has {edge_count} elements.")

    if count != params_count:
        raise ValueError(f"Count mismatch: input argument count is {count}, params_list has {params_count} elements.")

    # If all checks pass, print a success message
    print("Count check passed")


def read_rds_to_pytorch(path, count):
    # List all files in the directory
    files_tree = [f for f in os.listdir(os.path.join(path, 'GPS', 'tree'))
                  if f.startswith('tree_') and f.endswith('.rds')]
    files_node = [f for f in os.listdir(os.path.join(path, 'GPS', 'tree', 'node'))
                  if f.startswith('node_') and f.endswith('.rds')]
    files_edge = [f for f in os.listdir(os.path.join(path, 'GPS', 'tree', 'edge'))
                  if f.startswith('edge_') and f.endswith('.rds')]

    # Sort the files based on the parameters
    files_tree = sort_files(files_tree)
    files_node = sort_files(files_node)
    files_edge = sort_files(files_edge)

    # Check if the files are consistent
    check_file_consistency(files_tree, files_node, files_edge)

    # List to hold the data from each .rds file
    data_list = []
    params_tree_list = []

    # Loop through the files with the prefix 'tree_'
    for filename in files_tree:
        file_path = os.path.join(path, 'GPS', 'tree', filename)
        result = pyreadr.read_r(file_path)
        data = result[None]
        data_list.append(data)
        params_tree_list.append(get_params_string(filename))

    node_list = []
    params_node_list = []

    # Loop through the files with the prefix 'node_'
    for filename in files_node:
        node_file_path = os.path.join(path, 'GPS', 'tree', 'node', filename)
        node_result = pyreadr.read_r(node_file_path)
        node_data = node_result[None]
        node_list.append(node_data)
        params_node_list.append(get_params_string(filename))

    edge_list = []
    params_edge_list = []

    # Loop through the files with the prefix 'edge_'
    for filename in files_edge:
        edge_file_path = os.path.join(path, 'GPS', 'tree', 'edge', filename)
        edge_result = pyreadr.read_r(edge_file_path)
        edge_data = edge_result[None]
        edge_list.append(edge_data)
        params_edge_list.append(get_params_string(filename))

    check_params_consistency(params_tree_list, params_node_list, params_edge_list)

    params_list = []

    for filename in files_tree:
        params = get_params(filename)
        params_list.append(params)

    # Normalize carrying capacity by dividing by a factor
    for vector in params_list:
        vector[2] = vector[2] / cap_norm_factor

    check_list_count(count, data_list, node_list, edge_list, params_list)

    # List to hold the Data objects
    pytorch_geometric_data_list = []

    for i in range(0, count):
        # Ensure the DataFrame is of integer type and convert to a tensor
        edge_index_tensor = torch.tensor(data_list[i].values, dtype=torch.long)

        # Make sure the edge_index tensor is of size [2, num_edges]
        edge_index_tensor = edge_index_tensor.t().contiguous()

        # Determine the number of nodes
        num_nodes = edge_index_tensor.max().item() + 1

        # Create a tensor of node features. The data type is set to long because for now we only use
        # node class labels as features (root, internal, or leaf)
        # If we want to use other features, we need to change the data type
        node_feature_tensor = torch.tensor(node_list[i].values, dtype=torch.long)

        # Create a tensor of edge features. Edge lengths are stored in the first column of the DataFrame
        edge_feature_tensor = torch.tensor(edge_list[i].values, dtype=torch.float)

        params_current = params_list[i]

        params_current_tensor = torch.tensor(params_current[0:3], dtype=torch.float)

        # Create a Data object with the edge index, number of nodes, and category value
        data = Data(x=node_feature_tensor,
                    edge_attr=edge_feature_tensor,
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

    # Concatenate the base directory path with the set_i folder name
    full_dir = os.path.join(name, task_type)
    full_dir_tree = os.path.join(full_dir, 'GPS', 'tree')
    full_dir_node = os.path.join(full_dir, 'GPS', 'tree', 'node')
    full_dir_edge = os.path.join(full_dir, 'GPS', 'tree', 'edge')
    # Call read_rds_to_pytorch with the full directory path
    print(full_dir)
    # Check if the number of .rds files in the tree and el paths are equal
    rds_count = check_rds_files_count(full_dir_tree, full_dir_node, full_dir_edge)
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

    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    training_dataset = TreeData(root=None, data_list=filtered_training_data, transform=transform)
    # validation_dataset = TreeData(root=None, data_list=filtered_validation_data, transform=transform)
    testing_dataset = TreeData(root=None, data_list=filtered_testing_data, transform=transform)

    train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(validation_dataset, batch_size=64)
    test_loader = DataLoader(testing_dataset, batch_size=64)

    class GPS(torch.nn.Module):
        def __init__(self, channels: int, pe_dim: int, num_layers: int,
                     attn_type: str, attn_kwargs: Dict[str, Any]):
            super().__init__()

            self.node_emb = Embedding(28, channels - pe_dim)
            self.pe_lin = Linear(20, pe_dim)
            self.pe_norm = BatchNorm1d(20)
            self.edge_emb = Embedding(4, channels)

            self.convs = ModuleList()
            for _ in range(num_layers):
                nn = Sequential(
                    Linear(channels, channels),
                    ReLU(),
                    Linear(channels, channels),
                )
                conv = GPSConv(channels, GINEConv(nn), heads=4,
                               attn_type=attn_type, attn_kwargs=attn_kwargs)
                self.convs.append(conv)

            self.mlp = Sequential(
                Linear(channels, channels // 2),
                ReLU(),
                Linear(channels // 2, channels // 4),
                ReLU(),
                Linear(channels // 4, 3),
            )
            self.redraw_projection = RedrawProjection(
                self.convs,
                redraw_interval=1000 if attn_type == 'performer' else None)

        def forward(self, x, pe, edge_index, edge_attr, batch):
            x_pe = self.pe_norm(pe)
            x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
            edge_attr = self.edge_emb(edge_attr)

            for conv in self.convs:
                x = conv(x, edge_index, batch, edge_attr=edge_attr)
            x = global_add_pool(x, batch)
            return self.mlp(x)

    class RedrawProjection:
        def __init__(self, model: torch.nn.Module,
                     redraw_interval: Optional[int] = None):
            self.model = model
            self.redraw_interval = redraw_interval
            self.num_last_redraw = 0

        def redraw_projections(self):
            if not self.model.training or self.redraw_interval is None:
                return
            if self.num_last_redraw >= self.redraw_interval:
                fast_attentions = [
                    module for module in self.model.modules()
                    if isinstance(module, PerformerAttention)
                ]
                for fast_attention in fast_attentions:
                    fast_attention.redraw_projection_matrix()
                self.num_last_redraw = 0
                return
            self.num_last_redraw += 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = GPS(channels=64, pe_dim=8, num_layers=10, attn_type=attn_type,
                attn_kwargs=attn_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)

    def train():
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            model.redraw_projection.redraw_projections()
            out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                        data.batch)
            loss = (out.squeeze() - data.y).abs().mean()
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def test_diff(loader):
        model.eval()

        diffs_all = torch.tensor([], dtype=torch.float, device=device)

        for data in loader:
            data = data.to(device)
            out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                        data.batch)
            diffs = torch.abs(out - data.y.view(data.num_nodes.__len__(), 3))
            diffs_all = torch.cat((diffs_all, diffs), dim=0)

        print(f"diffs_all length: {len(diffs_all)}; test_loader.dataset length: {len(loader.dataset)}; Equal: {len(diffs_all) == len(loader.dataset)}")
        mean_diffs = torch.sum(diffs_all, dim=0) / len(test_loader.dataset)

        return mean_diffs.cpu().detach().numpy(), diffs_all.cpu().detach().numpy()

    for epoch in range(1, epoch_number):
        loss = train()
        test_mean_diffs, _ = test_diff(test_loader)
        # scheduler.step(test_mae)
        print(f'Epoch: {epoch:03d}, Par 1 Mean Diff: {test_mean_diffs[0]:.4f}, Par 2 Mean Diff: {test_mean_diffs[1]:.4f}, Par 3 Mean Diff: {test_mean_diffs[2]:.4f}, Train Loss: {loss:.4f}')


if __name__ == '__main__':
    main()
