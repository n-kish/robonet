import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

def load_graphs(pickle_file):
    graphs = []
    try:
        with open(pickle_file, "rb") as file:
            while True:
                try:
                    graph = pickle.load(file)
                    graphs.append(graph)  # Append each loaded graph to the list
                except EOFError:
                    break  # End of file reached
    except FileNotFoundError:
        print("The specified file does not exist.")
    return graphs

def create_dataset(graphs):
    data_list = []
    for graph in graphs:
        # Assuming graph has attributes: nodes, edges, node_categories, edge_categories
        node_features = torch.tensor(graph.nodes, dtype=torch.float).unsqueeze(1)        
        node_categories = torch.tensor(graph.node_categories, dtype=torch.long)  # Node categories
        
        # Debugging: Print shapes of node_features and node_categories
        print("Node features shape:", node_features.shape)
        print("Node categories shape:", node_categories.shape)

        # One-hot encode node categories
        node_categories_one_hot = torch.nn.functional.one_hot(node_categories).float()
        
        # Debugging: Print shape of one-hot encoded categories
        print("Node categories one-hot shape:", node_categories_one_hot.shape)
        
        # One-hot encode node categories
        node_categories_one_hot = torch.nn.functional.one_hot(node_categories).float()
        # Combine existing features with one-hot encoded categories
        x = torch.cat((node_features, node_categories_one_hot), dim=1)

        edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()  # Edge indices
        edge_categories = torch.tensor(graph.edge_categories, dtype=torch.long)  # Edge categories
        # One-hot encode edge categories
        edge_categories_one_hot = torch.nn.functional.one_hot(edge_categories).float()
        # You may need to adjust how you handle edge features based on your model's requirements

        # Create Data object
        data = Data(x=x, edge_index=edge_index)
        # If you have edge features, you can add them here
        # data.edge_attr = edge_categories_one_hot  # Uncomment if you want to include edge features

        data_list.append(data)
    return data_list

# Define the GraphConvNet model
class GraphConvNet(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GraphConvNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Function to fit the model and plot distances
def fit_and_plot(graphs):
    dataset = create_dataset(graphs)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = GraphConvNet(num_node_features=dataset[0].x.shape[1], num_classes=1)  # Adjust num_classes as needed
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(100):  # Adjust number of epochs as needed
        for data in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, torch.tensor([0.0]))  # Replace with actual target values
            loss.backward()
            optimizer.step()

    # Plotting distances (dummy example)
    distances = [torch.norm(model(data).detach()).item() for data in loader]
    plt.plot(distances)
    plt.title('Graph Distances')
    plt.xlabel('Graph Index')
    plt.ylabel('Distance')
    plt.show()

def main():
    pickle_file = '/home/knagiredla/robonet/logs/exp_GSCA_10_flat_base_ant_40_000_134_1731216187/xmlrobots/gen_1000_steps/valid/graph_jects.pickle'
    graphs = load_graphs(pickle_file)
    # Create dataset from loaded graphs
    dataset = create_dataset(graphs)
    # Optionally, print the dataset to verify
    print("Dataset created with", len(dataset), "graphs.")

    # Fit the model and plot distances
    fit_and_plot(graphs)

if __name__ == '__main__':
    main()