from pybaseball import statcast

# This pulls a tiny amount of data (one day) just to test the connection
print("Attempting to download data...")
data = statcast(start_dt='2024-06-01', end_dt='2024-06-01')

print("Success! Here is the data:")
print(data.head())