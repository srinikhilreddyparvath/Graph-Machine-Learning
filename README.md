# Graph-Machine-Learning
 This repository focuses on Subgraph-based Link Prediction in large-scale networks. It includes implementations of various graph neural networks (GNNs) and other graph-based algorithms to process large graphs efficiently and perform tasks such as link prediction and node classification.


For this, I’m leveraging subgraph-based methods that allow me to extract relevant subgraphs around potential links and use them as input features for the GNN. By processing these subgraphs, the model learns to identify patterns and structural properties that indicate whether a link should exist.

Code Implementation




pip install torch torchvision torch-geometric

python
Copy code
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges

I’m using the Cora dataset, which is a citation network commonly used for testing graph algorithms. The dataset is loaded from PyTorch Geometric.


# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
Step 4: Preprocess the Data


# Split the edges for link prediction
data = train_test_split_edges(data)

I’m using a simple Graph Convolutional Network (GCN) for this task.


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize the model
model = GCN(dataset.num_features, 16)

Set up the training loop for the model.


# Set up optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

def train():
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.train_pos_edge_index)
    loss = criterion(z, data.train_pos_edge_index, data.train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(200):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

Finally, evaluate the model’s performance on the test set.


def test():
    model.eval()
    z = model(data.x, data.test_pos_edge_index)
    pos_score = model(data.x, data.test_pos_edge_index)
    neg_score = model(data.x, data.test_neg_edge_index)
    # Calculate the ROC AUC score
    return roc_auc_score(pos_score, neg_score)

auc_score = test()
print(f'AUC Score: {auc_score:.4f}')

Explanation
In this project, I’ve focused on processing large-scale graphs using Graph Neural Networks (GNNs) to perform link prediction. Here’s how I approached it:

Dataset Selection: I chose the Cora dataset, a well-known citation network, to demonstrate link prediction. The goal was to predict if there’s a link (citation) between two papers (nodes).

Data Preprocessing: I split the edges of the graph into training and test sets, ensuring that the model would learn from some edges while being evaluated on others.

Model Development: I developed a Graph Convolutional Network (GCN) that learns node embeddings, which are then used to predict whether a link exists between two nodes. The GCN captures the structural information of the graph, which is crucial for tasks like link prediction.

Training the Model: I trained the GCN using a binary cross-entropy loss function, which is appropriate for link prediction tasks where we want to distinguish between the presence and absence of links.

Model Evaluation: Finally, I evaluated the model’s performance using the ROC AUC score, a common metric for binary classification tasks.

By working through this project, I was able to explore large-scale graph processing techniques and apply them to real-world problems like link prediction. This work has broadened my understanding of how GNNs can be used to analyze and interpret complex graph structures, which is highly relevant for applications in social networks, citation networks, and beyond.
