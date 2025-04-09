import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATConv

from torch_geometric.utils import from_networkx

class OpennessGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(OpennessGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return torch.sigmoid(self.lin(x)).squeeze()



# define graphs

pyg_graphs = []
for G in graphs:
    # Convert node features to a matrix (e.g. [x, y, s, a, o, dir])
    for nid in G.nodes:
        node = G.nodes[nid]
        G.nodes[nid]['x'] = torch.tensor([node["x"], node["y"], node["s"], node["a"], node["o"], node["dir"]], dtype=torch.float)

    # Prepare labels
    for nid in G.nodes:
        G.nodes[nid]['y'] = torch.tensor([G.nodes[nid]["is_target"]], dtype=torch.float)

    pyg_graph = from_networkx(G, group_node_attrs=['x', 'y'], group_edge_attrs=['separation'])
    pyg_graphs.append(pyg_graph)










model = OpennessGNN(in_channels=6, hidden_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

loader = DataLoader(pyg_graphs, batch_size=1, shuffle=True)

for epoch in range(20):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = loss_fn(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")



model.eval()
with torch.no_grad():
    output = model(graph.x, graph.edge_index, graph.edge_attr)
    # output[i] is the predicted openness score for player i
