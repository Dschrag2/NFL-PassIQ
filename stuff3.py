import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

# === 1. Dataset Preprocessing ===
def normalize_features(play_data):
    # Flatten all node features to fit scaler
    all_nodes = [p["receivers"] + p["defenders"] for p in play_data]
    flat_features = np.vstack(all_nodes)

    scaler = StandardScaler()
    scaler.fit(flat_features)

    for play in play_data:
        combined = play["receivers"] + play["defenders"]
        normalized = scaler.transform(combined)
        play["receivers"] = normalized[:len(play["receivers"])]
        play["defenders"] = normalized[len(play["receivers"]):]
    
    return play_data

def create_graph_data(play_data):
    data_list = []
    for play in play_data:
        x = torch.tensor(play["receivers"] + play["defenders"], dtype=torch.float)
        num_receivers = len(play["receivers"])
        num_nodes = x.size(0)

        # Convert edge index to tensor
        edge_index = torch.tensor(play["edges"], dtype=torch.long).t().contiguous()

        # Label: 1 for the targeted receiver, 0 otherwise
        y = torch.zeros(num_nodes)
        y[play["target"]] = 1

        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_receivers = num_receivers  # keep track of which nodes are receivers
        data_list.append(data)
    
    return data_list

# === 2. GNN Model ===
class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, dropout=0.1)
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=1)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        return self.out(x).squeeze(-1)  # [num_nodes]

# === 3. Training ===
def train(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        
        # Only compute loss for receiver nodes
        target_mask = torch.zeros_like(batch.y, dtype=torch.bool)
        for i, d in enumerate(batch.to_data_list()):
            target_mask[i*d.num_nodes:(i*d.num_nodes)+d.num_receivers] = True

        loss = criterion(out[target_mask], batch.y[target_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

# === 4. Evaluation ===
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            out = model(batch.x, batch.edge_index)

            for i, d in enumerate(batch.to_data_list()):
                start = i * d.num_nodes
                end = start + d.num_nodes
                scores = out[start:end]
                preds = scores[:d.num_receivers]
                pred_index = torch.argmax(preds)
                if pred_index.item() == d.y[:d.num_receivers].argmax().item():
                    correct += 1
                total += 1
    return correct / total

# === 5. Main ===
def main(play_data, input_dim):
    # Normalize and prepare data
    play_data = normalize_features(play_data)
    graph_data = create_graph_data(play_data)
    
    # Shuffle and split
    random.shuffle(graph_data)
    split = int(0.8 * len(graph_data))
    train_data = graph_data[:split]
    val_data = graph_data[split:]

    train_loader = DataLoader(train_data, batch_size=8)
    val_loader = DataLoader(val_data, batch_size=8)

    model = GNNModel(in_channels=input_dim, hidden_channels=32)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, 31):
        train_loss = train(model, train_loader, optimizer, criterion)
        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch} | Loss: {train_loss:.4f} | Val Acc: {acc:.4f}")


    return model


if __name__ == "__main__":
    # Create a tiny synthetic dataset to test it
    def generate_mock_play(num_receivers=5, num_defenders=7, feature_dim=6):
        receivers = np.random.rand(num_receivers, feature_dim).tolist()
        defenders = np.random.rand(num_defenders, feature_dim).tolist()
        edges = []

        for r in range(num_receivers):
            for d in range(num_receivers, num_receivers + num_defenders):
                edges.append([r, d])
                edges.append([d, r])  # undirected

        target = np.random.randint(0, num_receivers)
        return {
            "receivers": receivers,
            "defenders": defenders,
            "edges": edges,
            "target": target,
        }

    # Generate fake dataset of 100 plays
    mock_dataset = [generate_mock_play() for _ in range(100)]
    model = main(mock_dataset, input_dim=6)