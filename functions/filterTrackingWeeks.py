import pandas as pd

# Load play-by-play data
pbp_data = pd.read_csv("filtered_pass_plays.csv")

# Loop through tracking week files
for week in range(1, 10):
    filename = f"tracking_week_{week}.csv"
    output_filename = f"filtered_tracking_week_{week}.csv"

    # Load tracking data
    tracking_data = pd.read_csv(filename)

    # Filter for 'pass_forward' events
    tracking_data = tracking_data[tracking_data['event'].str.contains("pass_forward", na=False)]

    # Merge with play-by-play data on gameId and playId
    filtered_data = tracking_data.merge(pbp_data[['gameId', 'playId']], on=['gameId', 'playId'], how='inner')

    # Save to a new CSV file
    filtered_data.to_csv(output_filename, index=False)

    print(f"Saved {output_filename}")
