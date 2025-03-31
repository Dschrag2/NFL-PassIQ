import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("./data/plays.csv", na_values=["NA"])

# ------ HANDLE MISSING VALUES ------ #
categorical_cols = ["offenseFormation", "dropbackType", "passLocationType", 
                    "rushLocationType", "pff_passCoverage", "pff_manZone",
                    "possessionTeam", "defensiveTeam", "yardlineSide"]
numeric_cols = [
    "yardsToGo", "yardlineNumber", "preSnapHomeScore", "preSnapVisitorScore",
    "absoluteYardlineNumber", "preSnapHomeTeamWinProbability",
    "preSnapVisitorTeamWinProbability", "expectedPoints", "playClockAtSnap",
    "passLength", "targetX", "targetY", "dropbackDistance",
    "timeToThrow", "timeInTackleBox", "timeToSack", "penaltyYards",
    "prePenaltyYardsGained", "yardsGained", "homeTeamWinProbabilityAdded",
    "visitorTeamWinProbabilityAdded", "expectedPointsAdded"
]
bool_cols = [
    "playAction", "passTippedAtLine", "unblockedPressure", "qbSpike",
    "qbKneel", "qbSneak", "isDropback"
]

# Fill missing values separately
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# ------ CATEGORICAL ENCODING ------ #
one_hot_cols = ["offenseFormation", "dropbackType", "passLocationType", "rushLocationType", 
                "pff_passCoverage", "pff_manZone"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_cats = encoder.fit_transform(df[one_hot_cols])
df_encoded_cats = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(one_hot_cols))

df = df.drop(columns=one_hot_cols).reset_index(drop=True)
df = pd.concat([df, df_encoded_cats], axis=1)

# Label Encode ordinal categorical columns
label_encoders = {}
ordinal_cols = ["quarter", "down"]
for col in ordinal_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Convert team names to numeric IDs
team_encoders = {}
team_cols = ["possessionTeam", "defensiveTeam", "yardlineSide"]
for col in team_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    team_encoders[col] = le

# ------ BOOLEAN FEATURES TO BINARY ------ #
# Handle NaN in boolean columns before conversion
df[bool_cols] = df[bool_cols].fillna(0)  # Fill NaN with 0 (or False)
df[bool_cols] = df[bool_cols].astype(int)

# ------ TIME-BASED FEATURE ENGINEERING ------ #
df["gameClock"] = df["gameClock"].str.split(":").apply(lambda x: int(x[0]) * 60 + int(x[1]))
df["gameClock"] = MinMaxScaler().fit_transform(df[["gameClock"]])

# ------ NUMERIC FEATURE SCALING ------ #
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ------ SPLIT INTO TRAIN, VALIDATION, TEST SETS ------ #
# Define target variable (modify if needed)
target_col = "yardsGained"  # Change this based on your prediction goal
X = df.drop(columns=[target_col])  # Features
y = df[target_col]  # Target

# First, split into train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=None)

# Then, split temp into validation (15%) and test (15%)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=None)

# Print dataset sizes
print(f"Training Set: {X_train.shape}, Validation Set: {X_valid.shape}, Test Set: {X_test.shape}")

# ------ MODEL TRAINING ------ #
# Initialize and train the model (e.g., Random Forest Regressor)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ------ EVALUATION ON VALIDATION SET ------ #
y_val_pred = model.predict(X_valid)
mse_val = mean_squared_error(y_valid, y_val_pred)
r2_val = r2_score(y_valid, y_val_pred)
print(f"Validation MSE: {mse_val}")
print(f"Validation R2: {r2_val}")

# ------ FINAL EVALUATION ON TEST SET ------ #
y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"Test MSE: {mse_test}")
print(f"Test R2: {r2_test}")
