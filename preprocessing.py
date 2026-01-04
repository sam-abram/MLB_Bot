import pandas as pd
import torch
import pickle
import numpy as np

# --- SETUP: LOAD DATA ---
# If continuing directly from the previous script, assume 'final_df' exists.
# If loading from a file, uncomment the line below:
# final_df = pd.read_csv('mlb_pa_data_defense.csv')

print(f"Original Row Count: {len(final_df)}")

# --- STEP 1: CLEAN & ENCODE TARGET (MANUAL DICTIONARY) ---
# We manually define the mapping to ensure absolute control over the output classes.
# This prevents "random" assignment of numbers to results.

manual_event_map = {
    'strikeout': 0,
    'field_out': 1,
    'single': 2,
    'walk': 3,
    'double': 4,
    'home_run': 5,
    'force_out': 1,  # Grouping 'force_out' with 'field_out'
    'grounded_into_double_play': 1, # Grouping GIDP with 'field_out'
    'sac_fly': 1,    # Grouping Sac Fly with 'field_out'
    'hit_by_pitch': 3, # Grouping HBP with 'walk'
    'triple': 6
}

# Filter data to only include these known events
final_df = final_df[final_df['events'].isin(manual_event_map.keys())].copy()

# Apply the mapping to create the Target Column
final_df['event_code'] = final_df['events'].map(manual_event_map)

print(f"Filtered Row Count: {len(final_df)}")
print("Target Encoding Complete.")


# --- STEP 2: CREATE ID MAPPINGS (TOKENIZATION) ---
# We build dictionaries to convert Names -> Integers.
# We explicitly reserve Index 0 for 'Unknown/None'.

def create_id_mapping(unique_values):
    # Filter out 0 if it exists (since we reserve 0 for unknown)
    clean_values = [x for x in unique_values if x != 0]
    # Create map starting at 1
    return {val: i+1 for i, val in enumerate(clean_values)}

# 2a. Gather unique IDs
unique_batters = final_df['batter'].unique()
unique_pitchers = final_df['pitcher'].unique()
unique_teams = final_df['home_team'].unique()

# 2b. Gather unique Fielders (across all 8 positions)
defense_cols = ['fielder_2', 'fielder_3', 'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9']
# Flatten all fielder columns into one array of unique IDs
unique_fielders = pd.unique(final_df[defense_cols].values.ravel('K'))

# 2c. Build the Dictionaries
batter_map = create_id_mapping(unique_batters)
pitcher_map = create_id_mapping(unique_pitchers)
team_map = create_id_mapping(unique_teams)
fielder_map = create_id_mapping(unique_fielders)

print("ID Mappings Created.")


# --- STEP 3: APPLY MAPPINGS TO DATA ---
# Convert the dataframe columns from Real IDs to Model Indices (1, 2, 3...)
# Any ID not found in the map (or 0) becomes 0.

# Map Batter, Pitcher, Team
final_df['batter_idx'] = final_df['batter'].map(batter_map).fillna(0).astype(int)
final_df['pitcher_idx'] = final_df['pitcher'].map(pitcher_map).fillna(0).astype(int)
final_df['team_idx'] = final_df['home_team'].map(team_map).fillna(0).astype(int)

# Map All Fielders
# This loop handles the "Optional Defense" logic. 
# If a fielder was 0 (Unknown) in the raw data, they map to 0 here.
for col in defense_cols:
    final_df[col + '_idx'] = final_df[col].map(fielder_map).fillna(0).astype(int)

print("Data Tokenization Complete.")


# --- STEP 4: SAVE ARTIFACTS (THE "BRAIN") ---
# We save the dictionaries so the App can use them later.

artifacts = {
    'batter_map': batter_map,
    'pitcher_map': pitcher_map,
    'team_map': team_map,
    'fielder_map': fielder_map,
    'event_map': manual_event_map
}

with open('mappings.pkl', 'wb') as f:
    pickle.dump(artifacts, f)
print("Saved mappings.pkl")


# --- STEP 5: CREATE TENSORS (FOR PYTORCH) ---
# Format the data into the mathematical structure the model needs.

# A. Categorical Inputs (The IDs) - LongTensor
# Order: Batter, Pitcher, Team, Fielders(2-9)
cat_cols = ['batter_idx', 'pitcher_idx', 'team_idx'] + [c + '_idx' for c in defense_cols]
X_categorical = torch.tensor(final_df[cat_cols].values, dtype=torch.long)

# B. Numerical Inputs (Pitch Count) - FloatTensor
# We Normalize it (0.0 to 1.0) for better model stability.
# We'll use a fixed max (e.g., 120 pitches) so the scaling is consistent in the App.
MAX_PITCHES = 120
X_numerical = torch.tensor(
    (final_df[['pitcher_game_pitch_count']] / MAX_PITCHES).fillna(0).values, 
    dtype=torch.float32
)

# C. Targets (The Results) - LongTensor
y = torch.tensor(final_df['event_code'].values, dtype=torch.long)

# Save the training data
torch.save({
    'X_cat': X_categorical, 
    'X_num': X_numerical, 
    'y': y,
    'num_classes': len(manual_event_map)
}, 'processed_data.pt')

print("Saved processed_data.pt")
print("Preprocessing Complete. Ready for Training.")