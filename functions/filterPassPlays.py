import pandas as pd

# Load the dataset
df = pd.read_csv("plays.csv")

# Filter only passing plays (excluding sacks and scrambles)
df_pass_plays = df[df["passResult"].isin(["C", "I", "IN"])]

# Save the filtered data to a new CSV file
df_pass_plays.to_csv("filtered_pass_plays.csv", index=False)

print("Filtered pass plays saved to 'filtered_pass_plays.csv'.")