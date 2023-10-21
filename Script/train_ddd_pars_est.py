import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyreadr
import torch
import glob
import functools
import umap
import imageio
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import InMemoryDataset, Data


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


def read_rds_to_pytorch(path, model, count):
    # Map metrics to category values
    metric_to_category = {'BD_TES': 0, 'DDD_TES': 1, 'EVE_TES': 2}

    # Check if the provided metric is valid
    if model not in metric_to_category:
        raise ValueError(f"Unknown model: {model}. Expected one of: {', '.join(metric_to_category.keys())}")

    # Get the category value for the provided prefix
    category_value = torch.tensor([metric_to_category[model]], dtype=torch.long)

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
    current_dataset = read_rds_to_pytorch(full_dir, task_type, rds_count)
    current_training_data = get_training_data(current_dataset)
    current_testing_data = get_testing_data(current_dataset)
    training_dataset_list.append(current_training_data)
    testing_dataset_list.append(current_testing_data)

    sum_training_data = functools.reduce(lambda x, y: x + y, training_dataset_list)
    sum_testing_data = functools.reduce(lambda x, y: x + y, testing_dataset_list)

    class TreeData(InMemoryDataset):
        def __init__(self, root, data_list, transform=None, pre_transform=None):
            super(TreeData, self).__init__(root, transform, pre_transform)
            self.data, self.slices = self.collate(data_list)

        def _download(self):
            pass  # No download required

        def _process(self):
            pass  # No processing required

    training_dataset = TreeData(root=None, data_list=sum_training_data)
    testing_dataset = TreeData(root=None, data_list=sum_testing_data)

    class GCN(torch.nn.Module):
        def __init__(self, hidden_size=32, num_params=3):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(training_dataset.num_node_features, hidden_size)
            self.conv2 = GCNConv(hidden_size, hidden_size)
            self.conv3 = GCNConv(hidden_size, hidden_size)
            self.linear = Linear(hidden_size, num_params)

        def forward(self, x, edge_index, batch, return_embeddings=False):
            # 1. Obtain node embeddings
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)

            # Readout layer
            embeddings = global_mean_pool(x, batch)

            # Apply a final classifier
            x = F.dropout(embeddings, p=0.5, training=self.training)
            out = self.linear(x)

            if return_embeddings:
                return out, embeddings
            else:
                return out

    def train():
        model.train()

        loss_all = 0  # Keep track of the loss
        all_embeddings = []  # Collect embeddings
        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()
            out, embeddings = model(data.x, data.edge_index, data.batch, return_embeddings=True)
            loss = criterion(out, data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

            all_embeddings.append(embeddings.cpu().detach().numpy())

        all_embeddings = np.vstack(all_embeddings)  # Stack the embeddings into one array
        return loss_all / len(train_loader.dataset), all_embeddings

    def test(loader):
        model.eval()

        loss_all = 0
        all_embeddings = []

        for data in loader:
            data.to(device)
            out, embeddings = model(data.x, data.edge_index, data.batch, return_embeddings=True)
            all_embeddings.append(embeddings.cpu().detach().numpy())  # Save the embeddings
            loss = criterion(out, data.y)
            loss_all += loss.item() * data.num_graphs

        all_embeddings = np.vstack(all_embeddings)  # Stack the embeddings into one array

        return loss_all / len(loader.dataset), all_embeddings

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training using {device}")

    model = GCN(hidden_size=64)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss().to(device)
    train_loader = DataLoader(training_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testing_dataset, batch_size=64, shuffle=False)

    print(model)

    test_loss_history = []
    train_loss_history = []

    train_dir = os.path.join(name, task_type, "training")
    test_dir = os.path.join(name, task_type, "testing")

    # Check and create directories if not exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for epoch in range(1, 200):
        train_loss_all, train_embeddings = train()
        test_loss_all, test_embeddings = test(test_loader)
        print(f'Epoch: {epoch:03d}, Test Acc: {test_loss_all:.4f}, Loss: {train_loss_all:.4f}')

        # Record the values
        test_loss_history.append(test_loss_all)
        train_loss_history.append(train_loss_all)

        export_to_rds(train_embeddings, epoch, name, task_type, 'train')
        export_to_rds(test_embeddings, epoch, name, task_type, 'test')

    # After the loop, create a dictionary to hold the data
    data_dict = {
        'Epoch': list(range(1, 200)),
        'Test_LOSS': test_loss_history,
        'Train_LOSS': train_loss_history
    }

    # Convert the dictionary to a pandas DataFrame
    model_performance = pd.DataFrame(data_dict)
    write_data_name = '_'.join(params_current.astype(str))
    # Save the data to a file using pyreadr
    pyreadr.write_rds(os.path.join(name, task_type, f"{task_type}_{write_data_name}.rds"), model_performance)


if __name__ == '__main__':
    main()
