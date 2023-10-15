import sys
import os
import pandas as pd
import pyreadr
import torch
import glob
import functools
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import InMemoryDataset, Data


def get_params(name, set_index):
    # Form the filename by concatenating the base directory with 'params.txt'
    file_path = os.path.join(name, 'params.txt')

    # Read the data from params.txt
    df = pd.read_csv(file_path, sep="\s+", header=0)

    # Convert set_index to integer
    set_index = int(set_index) - 1  # subtract 1 because iloc is 0-based

    # Print the specified row
    return df.iloc[set_index]


def check_params(params):
    # Check if all the list elements have the same lambda, mu, beta_n, and beta_phi
    first_row = params[0]
    for row in params[1:]:
        if not (first_row[['lambda', 'mu', 'beta_n', 'beta_phi']].equals(row[['lambda', 'mu', 'beta_n', 'beta_phi']])):
            print("Parameter mismatch found. Data is not consistent.")
            sys.exit()
    print("All rows have the same lambda, mu, beta_n, and beta_phi.")


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


def read_rds_to_pytorch(path, set_index, count):
    params_current = get_params(path, set_index)

    metric = params_current['metric']

    # Map metrics to category values
    metric_to_category = {'pd': 0, 'ed': 1, 'nnd': 2}

    # Check if the provided metric is valid
    if metric not in metric_to_category:
        raise ValueError(f"Unknown metric: {metric}. Expected one of: {', '.join(metric_to_category.keys())}")

    # Get the category value for the provided prefix
    category_value = torch.tensor([metric_to_category[metric]], dtype=torch.long)

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


def main():
    # The base directory path is passed as the first argument
    name = sys.argv[1]

    print(f'Project: {name}')

    # The set_i folder names are passed as the remaining arguments
    set_paths = sys.argv[2:]

    params_list = []

    training_dataset_list = []
    testing_dataset_list = []

    # Iterate through each set_i folder name
    for set_index in set_paths:
        set_path = f'set_{set_index}'
        # Concatenate the base directory path with the set_i folder name
        full_dir = os.path.join(name, set_path)
        full_dir_tree = os.path.join(full_dir, 'GNN', 'tree')
        full_dir_el = os.path.join(full_dir, 'GNN', 'tree', 'EL')
        # Call read_rds_to_pytorch with the full directory path
        print(full_dir)  # The set_i folder names are passed as the remaining arguments
        params_current = get_params(name, set_index)
        print(params_current)
        params_list.append(params_current)

        # Check if the number of .rds files in the tree and el paths are equal
        rds_count = check_rds_files_count(full_dir_tree, full_dir_el)
        print(f'There are: {rds_count} trees in the set_{set_index} folder.')

        print(f"Now reading set_{set_index}...")
        # Read the .rds files into a list of PyTorch Geometric Data objects
        current_dataset = read_rds_to_pytorch(full_dir, set_index, rds_count)
        current_training_data = get_training_data(current_dataset)
        current_testing_data = get_testing_data(current_dataset)
        training_dataset_list.append(current_training_data)
        testing_dataset_list.append(current_testing_data)

    # Check if all the list elements have the same lambda, mu, beta_n, and beta_phi
    check_params(params_list)

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
        def __init__(self, hidden_size=32):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(training_dataset.num_node_features, hidden_size)
            self.conv2 = GCNConv(hidden_size, hidden_size)
            self.conv3 = GCNConv(hidden_size, hidden_size)
            self.linear = Linear(hidden_size, training_dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)

        return x

    def train():
        model.train()

        lost_all = 0
        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()  # Clear gradients.
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)   # Compute the loss.
            loss.backward()  # Derive gradients.
            lost_all += loss.item() * data.num_graphs
            optimizer.step()  # Update parameters based on gradients.

        return lost_all / len(train_loader.dataset)

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with the highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training using {device}")

    model = GCN(hidden_size=64)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    train_loader = DataLoader(training_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testing_dataset, batch_size=64, shuffle=False)

    print(model)

    train_acc_history = []
    loss_history = []

    for epoch in range(1, 200):
        loss = train()
        train_acc = test(test_loader)
        # test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Loss: {loss:.4f}')

        # Record the values
        train_acc_history.append(train_acc)
        loss_history.append(loss)

    # After the loop, create a dictionary to hold the data
    data_dict = {
        'Epoch': list(range(1, 200)),
        'Train_Accuracy': train_acc_history,
        'Loss': loss_history
    }

    # Convert the dictionary to a pandas DataFrame
    model_performance = pd.DataFrame(data_dict)
    params = get_params(os.path.join(name, f'set_{set_paths[0]}'), set_paths[0])
    write_data_name = '_'.join(params.drop(['metric', 'offset']).astype(str))
    # Save the data to a file using pyreadr
    pyreadr.write_rds(os.path.join(name, f"{write_data_name}.rds"), model_performance)


if __name__ == '__main__':
    main()
