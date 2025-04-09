import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
from torchsummary import summary


# Reading in Data
play_data = pd.read_csv("data/plays.csv")
player_data = pd.read_csv("data/players.csv")
player_play_data = pd.read_csv("data/player_play.csv")

passing_plays = play_data[play_data["passResult"].isin(["C", "I", "IN"])]

# Helper Functions

# Calculate distance between nodes
def get_edge_weights(edges, play, tracking_data):
    weights = []
    for i in range(len(edges[0])):
        P1 = tracking_data[(tracking_data["gameId"] == play.gameId) & (tracking_data["playId"] == play.playId) & (tracking_data["nflId"] == edges[0][i])]
        P2 = tracking_data[(tracking_data["gameId"] == play.gameId) & (tracking_data["playId"] == play.playId) & (tracking_data["nflId"] == edges[1][i])]
        point1 = np.array([P1["x"], P1["y"]])
        point2 = np.array([P2["x"], P2["y"]])
        dist = np.linalg.norm(point2 - point1)
        weights.append(dist)
    return weights

# Finding targeted node
def get_targeted_receiver(Recs, nflIDs):
    for WR in Recs.itertuples(index=True):
        if (WR.wasTargettedReceiver):
            return np.where(nflIDs == WR.nflId)[0][0]

def prepare_data(graph):
    edge_index = torch.tensor(graph['edges'], dtype=torch.long)
    edge_weights = torch.tensor(graph['edge_weights'], dtype=torch.float)
    receivers = torch.tensor(graph['receivers'], dtype=torch.float)
    defenders = torch.tensor(graph['defenders'], dtype=torch.float)
    x = torch.cat((receivers, defenders), dim=0)  # Combine receivers and defenders as node features
    
    # Assuming we are predicting openness for receivers only, extract the relevant 'y' values
    target_receiver_index = graph['y']
    y = torch.tensor([graph['y']], dtype=torch.float).view(-1)  # Assuming 'y' is the target label for receivers
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=y)
    return data

# Model Setup (GNN using GCNConv)
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1)  # Assuming regression task for openness prediction

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.fc(x)
        return x

# Getting tracking data at moment of pass
week_data = pd.read_csv("data/tracking_week_1.csv")
week_data = week_data[week_data["event"].isin(["pass_forward", "pass_shovel"])]

# Getting list of completions by gameId and playId
completions = passing_plays[passing_plays["passResult"] == "C"]
playIds = completions[["gameId", "playId"]].drop_duplicates()

# Filtering tracking data to only completion plays
week_data_completions = week_data.merge(playIds, on=["gameId", "playId"])

# Removing playIds not from this week
playIds = playIds[playIds["gameId"].isin(set(week_data_completions["gameId"]))]

graph_data = []

for i, row in tqdm(enumerate(playIds.itertuples(index=True)), total=len(playIds), desc="Processing Plays", unit="play"):
    play_data = {}
    Recs = player_play_data[(player_play_data["gameId"] == row.gameId) & (player_play_data["playId"] == row.playId) & (player_play_data["wasRunningRoute"] == True)]
    Defs = player_play_data[(player_play_data["gameId"] == row.gameId) & (player_play_data["playId"] == row.playId) & ~pd.isna(player_play_data["pff_defensiveCoverageAssignment"])]
    
    R_nodes = np.arange(len(Recs))
    D_nodes = np.arange(len(Recs), len(Defs) + len(Recs))
    play_data["edges"] = np.vstack([np.repeat(R_nodes, len(Defs)), np.tile(D_nodes, len(Recs))])
    
    R_nodes2 = Recs["nflId"].values
    D_nodes2 = Defs["nflId"].values
    nflIDs = np.concatenate([Recs["nflId"].values, Defs["nflId"].values])
    edgeIDs = np.vstack([np.repeat(R_nodes2, len(Defs)), np.tile(D_nodes2, len(Recs))])

    play_data["edge_weights"] = get_edge_weights(edgeIDs, row, week_data_completions)

    Receivers = []
    for WR in Recs.itertuples(index=True):
        WR_row = week_data_completions[(week_data_completions["gameId"] == row.gameId) & (week_data_completions["playId"] == row.playId) & (week_data_completions["nflId"] == WR.nflId)]
        Receivers.append([WR_row["x"].iloc[0], WR_row["y"].iloc[0], WR_row["s"].iloc[0], WR_row["a"].iloc[0], WR_row["o"].iloc[0], WR_row["dir"].iloc[0]])
    Defenders = []
    for Def in Defs.itertuples(index=True):
        Def_row = week_data_completions[(week_data_completions["gameId"] == row.gameId) & (week_data_completions["playId"] == row.playId) & (week_data_completions["nflId"] == Def.nflId)]
        Defenders.append([Def_row["x"].iloc[0], Def_row["y"].iloc[0], Def_row["s"].iloc[0], Def_row["a"].iloc[0], Def_row["o"].iloc[0], Def_row["dir"].iloc[0]])

    play_data["receivers"] = Receivers
    play_data["defenders"] = Defenders

    play_data["y"] = get_targeted_receiver(Recs, nflIDs)

    graph_data.append(play_data)

# Process each play and prepare the data
data_list = []
for play in graph_data:
    data = prepare_data(play)
    data_list.append(data)

# Create DataLoader for batching
train_loader = DataLoader(data_list, batch_size=8, shuffle=True)

# Model, loss function, optimizer
input_dim = 6  # Number of features for each node (x, y, speed, acceleration, orientation, direction)
hidden_dim = 64
output_dim = 32
model = GNNModel(input_dim, hidden_dim, output_dim)

criterion = nn.MSELoss()  # Assuming regression (openness prediction)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 125
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        
        # Get the predictions from the model
        out = model(data.x, data.edge_index, data.edge_attr)
        
        # Squeeze output tensor to match the shape of target 'y'
        # If you're only predicting for receivers, make sure you only consider the relevant subset
        out = out[:len(data.y)]  # Make sure to slice 'out' to match 'y' size

        loss = criterion(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
summary(model, input_size=(input_dim, 6))


# After training, you can evaluate your model on the validation/test set or use it for inference.
