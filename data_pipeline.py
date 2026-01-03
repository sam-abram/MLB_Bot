import pandas as pd
from pybaseball import statcast

# 1. Fetch Data (Example: May 2024)
print("Fetching Statcast data...")
data = statcast(start_dt='2024-05-01', end_dt='2024-06-01')

# 2. Sort for Pitch Counts
data = data.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number'])

# 3. Calculate Fatigue (Pitch Count)
data['pitcher_game_pitch_count'] = data.groupby(['game_pk', 'pitcher']).cumcount() + 1

# 4. Isolate Terminal Events (End of At-Bat)
pa_data = data.dropna(subset=['events']).copy()

# 5. Define Column List
# Note: 'fielder_2' is Catcher. 'fielder_1' would be pitcher, but we already have 'pitcher'.
feature_cols = [
    'events',                   # TARGET
    'batter',                   # INPUT: ID
    'pitcher',                  # INPUT: ID
    'stand',                    # INPUT: L/R
    'p_throws',                 # INPUT: L/R
    'home_team',                # INPUT: Stadium
    'pitcher_game_pitch_count', # INPUT: Fatigue
    
    # NEW: Specific Defense IDs
    'fielder_2', # Catcher
    'fielder_3', # 1B
    'fielder_4', # 2B
    'fielder_5', # 3B
    'fielder_6', # SS
    'fielder_7', # LF
    'fielder_8', # CF
    'fielder_9'  # RF
]

# 6. Extraction & Cleanup
final_df = pa_data[feature_cols].copy()

# Fill Missing Fielders (Important!)
# Sometimes Statcast misses a fielder ID. We fill with 0 (our "Unknown" token).
defense_cols = ['fielder_2', 'fielder_3', 'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9']
final_df[defense_cols] = final_df[defense_cols].fillna(0).astype(int)

print(final_df.head())
# final_df.to_csv('mlb_pa_data_defense.csv', index=False)