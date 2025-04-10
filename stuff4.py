import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
from tqdm import tqdm

# ----------------------
# Data Loading
# ----------------------
games = pd.read_csv("data/games.csv")
players = pd.read_csv("data/players.csv")
plays = pd.read_csv("data/plays.csv")
# For simplicity, here we load one tracking week.
tracking_data = pd.read_csv("data/tracking_week_1.csv")
player_plays = pd.read_csv("data/player_play.csv")

# ----------------------
# Helper Functions
# ----------------------
def adjust_position(row, game, play):
    row = row.copy()  # create an explicit copy to avoid SettingWithCopyWarning
    try:
        pos_team = play["possessionTeam"].iloc[0]
        yardline_side = play["yardlineSide"].iloc[0]
        yardline_number = play["yardlineNumber"].iloc[0]
        home_team = game["homeTeamAbbr"].iloc[0]
        away_team = game["visitorTeamAbbr"].iloc[0]
    except IndexError:
        return row  # skip if malformed

    if pos_team == home_team:
        absYard = yardline_number + 10 if yardline_side == home_team else 110 - yardline_number
        row.loc["x"] -= absYard
        row.loc["o"] = (row["o"] + 90) % 360
        row.loc["dir"] = (row["dir"] + 90) % 360
    elif pos_team == away_team:
        absYard = 110 - yardline_number if yardline_side == away_team else yardline_number + 10
        row.loc["x"] = absYard - row["x"]
        row.loc["y"] = 53.3 - row["y"]
        row.loc["o"] = (row["o"] + 270) % 360
        row.loc["dir"] = (row["dir"] + 270) % 360
    return row


def get_edge_weights(edge_index, play, tracking_subset):
    """
    Calculate distances between pairs of players using tracking data.
    """
    # Set the index to nflId for quick look-up; we use the throw frame.
    tracking_play = tracking_subset.set_index("nflId")
    weights = []
    # Loop using edge_index's second dimension (number of edges)
    for i in range(edge_index.size(1)):
        try:
            # Convert tensor scalars to Python integers
            idx1 = edge_index[0, i].item()
            idx2 = edge_index[1, i].item()
            # Look up corresponding positions. If the player isn't found, KeyError will be raised.
            point1 = tracking_play.loc[idx1][["x", "y"]].values
            point2 = tracking_play.loc[idx2][["x", "y"]].values
            dist = np.linalg.norm(point2 - point1)
        except KeyError:
            dist = 100.0  # Assign a high distance if one of the nodes is missing
        weights.append(dist)
    return weights

def get_targeted_receiver_and_label(Recs):
    """
    Find the targeted receiver (assumed flagged in player_plays) and return the label.
    Here, we assume that the row for the targeted receiver contains a column "ballCaught" that is 1 if caught.
    """
    # print(Recs)

    targeted = Recs[Recs["wasTargettedReceiver"] == True]
    if not targeted.empty:
        # Return both the receiver's nflId and the catch outcome label.
        rec = targeted.iloc[0]
        label = int(rec["hadPassReception"])  # Assumed binary: 1=caught, 0=not caught
        return rec["nflId"], label
    else:
        return None, None

# ----------------------
# Build Graph from Throw Frame
# ----------------------
def build_graph_from_throw(row):
    """
    Constructs a graph for a given play using the throw frame (pass_forward or pass_shovel).
    Nodes: Players on the field at the throw moment.
    Features: x, y, s, a, (optionally include orientation/direction).
    Label: Binary label indicating whether the targeted receiver caught the ball.
    """
    # Get play and game information for metadata adjustments
    curr_play = plays[(plays["gameId"] == row.gameId) & (plays["playId"] == row.playId)]
    curr_game = games[games["gameId"] == row.gameId]
    
    # Filter tracking data to only the throw frame based on event.
    # Use proper parentheses to group events.
    throw_events = ["pass_forward", "pass_shovel"]
    curr_tracking = tracking_data[
        (tracking_data["gameId"] == row.gameId) &
        (tracking_data["playId"] == row.playId) &
        (tracking_data["event"].isin(throw_events))
    ]
    # Remove rows for the football
    curr_tracking = curr_tracking[curr_tracking["club"] != "football"]
    
    # Get receiver candidates from player_plays.
    # Here we filter to players running routes as a proxy for eligible pass targets.
    Recs = player_plays[
        (player_plays["gameId"] == row.gameId) &
        (player_plays["playId"] == row.playId) &
        (player_plays["wasRunningRoute"] != "NA")
    ]
    
    if Recs.empty or curr_tracking.empty:
        return None
    
    # Identify the targeted receiver and the outcome label.
    target_nflid, label = get_targeted_receiver_and_label(Recs)
    if target_nflid is None:
        return None  # No targeted receiver, skip this play

    # Create a list of nflIds for nodes.
    # Start with all players running routes (from Recs) and then include others from the throw frame that are not in Recs.
    rec_nflids = list(Recs.nflId)
    other_nflids = list(
        curr_tracking[~curr_tracking["nflId"].isin(Recs.nflId)]["nflId"]
    )
    nflIDs = np.array(rec_nflids + other_nflids)
    
    # If the graph has too few nodes, skip.
    if len(nflIDs) < 2:
        return None
    
    # Build edges: use all pairs of nodes.
    edge_index = torch.combinations(torch.arange(len(nflIDs)), r=2).t().long()
    # Calculate edge attributes (e.g. distances) using the throw frame tracking info.
    edge_attr = torch.tensor(
        get_edge_weights(edge_index, row, curr_tracking),
        dtype=torch.float
    )
    
    # Construct node features.
    # Here we use a subset of features. You can easily add more (e.g. orientation, direction) as needed.
    features = []
    for pid in nflIDs:
        # Get the player’s row from tracking data.
        player_row = curr_tracking[curr_tracking["nflId"] == pid]
        if not player_row.empty:
            player_row = adjust_position(player_row.iloc[0], curr_game, curr_play)
            # Select features; adjust to include desired features.
            feat = player_row[["x", "y", "s", "a"]].values.astype(float)
            features.append(feat)
        else:
            # If missing tracking data for this player, append zeros.
            features.append(np.zeros(4))
    x = torch.tensor(np.vstack(features), dtype=torch.float)

    # Optionally, store the index of the targeted receiver in the node list
    # so that later you can directly address its embedding.
    try:
        target_index = np.where(nflIDs == target_nflid)[0][0]
    except IndexError:
        # If not found, skip this play.
        return None

    # Create the graph data object.
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.long))
    # Storing additional metadata (optional):
    graph.target_index = target_index  # targeted node index
    graph.nflIDs = nflIDs  # list of player IDs in the graph

    return graph

# ----------------------
# Build the Dataset
# ----------------------
graph_data = []
print("Building dataset from throw frames...")
for i, row in tqdm(player_plays.drop_duplicates(subset=["gameId", "playId"]).iterrows()):
    try:
        g = build_graph_from_throw(row)
        if g is not None:
            graph_data.append(g)
    except Exception as e:
        print(f"Error in play {row.playId}: {e}")
    # break

print(f"Created {len(graph_data)} graph samples.")

# ----------------------
# GNN Model for Binary Classification
# ----------------------
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)  # output a single logit for binary classification

    def forward(self, data):
        """
        data: a Batch object from torch_geometric.data.DataLoader containing:
              - x: node features,
              - edge_index, edge_attr, and
              - batch: mapping of nodes to graph instances.
        """
        x, edge_index = data.x, data.edge_index
        # First graph convolution layer
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        # Second graph convolution layer
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        # Global pooling (mean) over all nodes in each graph.
        x = global_mean_pool(x, data.batch)
        # Final classification layer
        out = self.fc(x)
        return out

# ----------------------
# Model Ready for Training
# ----------------------
# Example training setup: 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNModel(input_dim=4, hidden_dim=16).to(device)

# Create a DataLoader from the list of graph objects.
loader = DataLoader(graph_data, batch_size=32, shuffle=True)

# Define a loss function and optimizer.
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Simple training loop
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch).view(-1)  # shape: (batch_size,)
        # Assume labels are in batch.y (need to convert to float for BCE loss)
        labels = batch.y.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.num_graphs
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/len(graph_data):.4f}")

print("Training complete.")
