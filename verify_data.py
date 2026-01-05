import torch
import pickle
import random

# 1. Load the Artifacts
print("Loading data...")
data = torch.load('processed_data.pt')
with open('mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

X_cat = data['X_cat']   # The IDs
X_num = data['X_num']   # The Fatigue
y = data['y']           # The Results

# 2. Basic Shape Checks (Do we actually have data?)
print(f"Total Samples: {len(y)}")
print(f"Categorical Input Shape: {X_cat.shape} (Should be Rows x 11)")
print(f"Numerical Input Shape: {X_num.shape} (Should be Rows x 1)")

# 3. The "Human Readable" Test
# We need to invert the dictionaries (ID -> Name) to read the data
# The maps are currently {Name: ID}, we need {ID: Name}
inv_batter = {v: k for k, v in mappings['batter_map'].items()}
inv_pitcher = {v: k for k, v in mappings['pitcher_map'].items()}
inv_team = {v: k for k, v in mappings['team_map'].items()}
inv_event = {v: k for k, v in mappings['event_map'].items()}

# 4. Pick a Random Sample to Inspect
idx = random.randint(0, len(y) - 1)
print(f"\n--- Inspecting Sample Row #{idx} ---")

# Extract Raw IDs from Tensor
row_cat = X_cat[idx].tolist()
row_num = X_num[idx].item()
result_code = y[idx].item()

# Decode Inputs
# Note: X_cat columns are [Batter, Pitcher, Team, C, 1B, 2B, 3B, SS, LF, CF, RF]
batter_name = inv_batter.get(row_cat[0], "UNKNOWN/PAD")
pitcher_name = inv_pitcher.get(row_cat[1], "UNKNOWN/PAD")
team_name = inv_team.get(row_cat[2], "UNKNOWN/PAD")

# Decode Fatigue (Reverse the normalization)
# We divided by 120 in preprocessing, so multiply by 120 to get the real count
pitch_count = int(row_num * 120)

# Decode Result
outcome = inv_event.get(result_code, "ERROR")

print(f"Stadium: {team_name}")
print(f"Batter:  {batter_name}")
print(f"Pitcher: {pitcher_name} (Pitch #{pitch_count} of game)")
print(f"RESULT:  {outcome.upper()}")

# Check Defense (Just checking Catcher and CF as examples)
# Columns: 3=Catcher, 9=CenterField (Indices 3 and 9 in the list)
inv_fielder = {v: k for k, v in mappings['fielder_map'].items()}
catcher_name = inv_fielder.get(row_cat[3], "No Input/Unknown")
cf_name = inv_fielder.get(row_cat[9], "No Input/Unknown")

print(f"Defense Sample -> Catcher: {catcher_name}, CF: {cf_name}")