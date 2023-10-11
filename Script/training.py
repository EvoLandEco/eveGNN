import pyreadr
import os
import torch
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import InMemoryDataset, Data
import networkx

# Directory where the .rds files are saved
directory = "D:/Data/GNN/tree/"


def detect_prefix_counts(path):
    # Known prefixes
    prefixes = ['pd', 'ed', 'nnd']
    # Initialize a dictionary to hold the count of files for each prefix
    prefix_counts = {prefix: 0 for prefix in prefixes}

    # Iterate through the files in the directory
    for file_name in os.listdir(path):
        # Check if the file_name starts with any of the known prefixes
        for prefix in prefixes:
            if file_name.startswith(prefix):
                prefix_counts[prefix] += 1

    return prefix_counts


def read_rds_to_pytorch(path, prefix):
    # Map prefixes to category values
    prefix_to_category = {'pd': 0, 'ed': 1, 'nnd': 2}

    # Check if the provided prefix is valid
    if prefix not in prefix_to_category:
        raise ValueError(f"Unknown prefix: {prefix}. Expected one of: {', '.join(prefix_to_category.keys())}")

    # Get the category value for the provided prefix
    category_value = torch.tensor([prefix_to_category[prefix]], dtype=torch.long)

    # Get the count of files for the specified prefix
    prefix_counts = detect_prefix_counts(path)
    num_files = prefix_counts[prefix]

    # List to hold the data from each .rds file
    data_list = []

    # Loop through the files for the specified prefix
    for i in range(1, num_files + 1):
        # Construct the file path using the specified prefix
        file_path = os.path.join(path, f"{prefix}_tas_{i}.rds")

        # Read the .rds file
        result = pyreadr.read_r(file_path)

        # The result is a dictionary where keys are the name of objects and the values python dataframes
        # Since RDS can only contain one object, it will be the first item in the dictionary
        data = result[None]

        # Append the data to data_list
        data_list.append(data)

    length_list = []

    for i in range(1, num_files + 1):
        length_file_path = os.path.join(path, "EL/", f"{prefix}_tas_EL_{i}.rds")
        length_result = pyreadr.read_r(length_file_path)
        length_data = length_result[None]
        length_list.append(length_data)

    # List to hold the Data objects
    pytorch_geometric_data_list = []

    for i in range(0, num_files):
        # Ensure the DataFrame is of integer type and convert to a tensor
        edge_index_tensor = torch.tensor(data_list[i].values, dtype=torch.long)

        # Make sure the edge_index tensor is of size [2, num_edges]
        edge_index_tensor = edge_index_tensor.t().contiguous()

        # Determine the number of nodes
        num_nodes = edge_index_tensor.max().item() + 1

        edge_length_tensor = torch.tensor(length_list[i].values, dtype=torch.float)

        num_node_features = 1

        # Create a tensor of zeros for the node features
        node_features = torch.zeros((num_nodes, num_node_features), dtype=torch.float)

        # Create a Data object with the edge index, number of nodes, and category value
        data = Data(x=edge_length_tensor,
                    edge_index=edge_index_tensor,
                    num_nodes=num_nodes,
                    edge_attr=edge_length_tensor,
                    y=category_value)

        # Append the Data object to the list
        pytorch_geometric_data_list.append(data)

    return pytorch_geometric_data_list


graph_pd = read_rds_to_pytorch(directory, 'pd')
graph_ed = read_rds_to_pytorch(directory, 'ed')
graph_nnd = read_rds_to_pytorch(directory, 'nnd')
all_graphs = graph_pd + graph_ed + graph_nnd


class TreeData(InMemoryDataset):
    def __init__(self, root, data_list, transform=None, pre_transform=None):
        super(TreeData, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass  # No download required

    def _process(self):
        pass  # No processing required


tree_dataset = TreeData(root=None, data_list=all_graphs)


class GCN(torch.nn.Module):
    def __init__(self, hidden_size=32):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(tree_dataset.num_node_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.linear = Linear(hidden_size, tree_dataset.num_classes)

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


model = GCN(hidden_size=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
train_loader = DataLoader(tree_dataset, batch_size=64, shuffle=True)
print(model)


def train():
    model.train()

    lost_all = 0
    for data in train_loader:
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
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 200):
    loss = train()
    train_acc = test(train_loader)
    # test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Loss: {loss:.4f}')

#%%
