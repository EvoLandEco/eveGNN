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


def read_table(path):
    return pd.read_csv(path, sep="\s+", header=0)  # assuming the tables are tab-delimited


def check_same_across_rows(df):
    return df.apply(lambda x: x.nunique() == 1)


def compare_tables(bd_path, ddd_path, eve_path):
    bd_df = read_table(bd_path)
    ddd_df = read_table(ddd_path)
    eve_df = read_table(eve_path)

    # Check if each table has the same lambda, mu, age across rows
    bd_check = check_same_across_rows(bd_df[['lambda', 'mu', 'age']])
    ddd_check = check_same_across_rows(ddd_df[['lambda', 'mu', 'age']])
    eve_check = check_same_across_rows(eve_df[['lambda', 'mu', 'age']])

    print("BD Params Consistency Check:", bd_check.to_dict())
    print("DDD Params Consistency Check:", ddd_check.to_dict())
    print("EVE Params Consistency Check:", eve_check.to_dict())

    # Check if the three tables' first row have the same lambda, mu, and age
    bd_first_row = bd_df.iloc[0][['lambda', 'mu', 'age']]
    ddd_first_row = ddd_df.iloc[0][['lambda', 'mu', 'age']]
    eve_first_row = eve_df.iloc[0][['lambda', 'mu', 'age']]

    first_row_check = (bd_first_row == ddd_first_row).all() and (bd_first_row == eve_first_row).all()
    print("Params Consistency Check across BD, DDD and EVE:", first_row_check)

    # Combine all check results
    all_checks_pass = bd_check.all() & ddd_check.all() & eve_check.all() & first_row_check

    if all_checks_pass:
        print("All checks passed.")
    else:
        print("Some checks failed.")

    return all_checks_pass


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


def get_params(model_name, set_string):
    try:
        # Split the string on the underscore
        parts = set_string.split('_')
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(f"Invalid format: {set_string}")
        # Convert the second part to an integer
        index = int(parts[1])

        # Construct the path to the params file
        file_path = f'{model_name}/{model_name}_params.txt'

        # Read the table
        df = pd.read_table(file_path, delimiter="\t")

        # Get the specified row (Python uses 0-based indexing, so we subtract 1 from index)
        row = df.iloc[index - 1]
        return row
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def read_rds_to_pytorch(path, model, count):
    # Map metrics to category values
    metric_to_category = {'BD_TAS': 0, 'DDD_TAS': 1, 'EVE_TAS': 2}

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


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <name> <set_i> <task_type>")
        sys.exit(1)

    name = sys.argv[1]
    set_i = sys.argv[2]
    task_type = sys.argv[3]

    # Now you can use the variables name and set_i in your code
    print(f'Name: {name}, Set: {set_i}, Task Type: {task_type}')

    bd_tree_path = os.path.join(name, "BD_TAS/set_1")

    training_dataset_list = []
    testing_dataset_list = []

    os.path.join(bd_tree_path, 'GNN', 'tree')
    os.path.join(bd_tree_path, 'GNN', 'tree', 'EL')
    rds_count = check_rds_files_count(os.path.join(bd_tree_path, 'GNN', 'tree'), os.path.join(bd_tree_path, 'GNN', 'tree', 'EL'))
    bd_dataset = read_rds_to_pytorch(bd_tree_path, "BD_TAS", rds_count)
    bd_training_data = get_training_data(bd_dataset)
    bd_testing_data = get_testing_data(bd_dataset)
    training_dataset_list.append(bd_training_data)
    testing_dataset_list.append(bd_testing_data)

    # Concatenate the base directory path with the set_i folder name
    full_dir = os.path.join(task_type, set_i)
    full_dir_tree = os.path.join(full_dir, 'GNN', 'tree')
    full_dir_el = os.path.join(full_dir, 'GNN', 'tree', 'EL')
    # Call read_rds_to_pytorch with the full directory path
    print(full_dir)  # The set_i folder names are passed as the remaining arguments
    params_current = get_params(task_type, set_i)
    print(params_current)

    # Check if the number of .rds files in the tree and el paths are equal
    rds_count = check_rds_files_count(full_dir_tree, full_dir_el)
    print(f'There are: {rds_count} trees in the {set_i} folder.')
    print(f"Now reading {task_type}:{set_i}...")
    # Read the .rds files into a list of PyTorch Geometric Data objects
    current_path = os.path.join(task_type, set_i)
    current_dataset = read_rds_to_pytorch(current_path, task_type, rds_count)
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
    write_data_name = '_'.join(params_current.astype(str))
    # Save the data to a file using pyreadr
    pyreadr.write_rds(os.path.join(task_type, f"{write_data_name}.rds"), model_performance)


if __name__ == '__main__':
    main()
