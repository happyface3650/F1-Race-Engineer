# Enhanced model with driver metadata and embeddings
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os

# First, run the comprehensive merge script to create the enhanced dataset
exec(open('data manipulation/merge_all_metadata.py').read())

# Load the comprehensive enhanced dataset
df = pd.read_csv('data\\All_Laps_with_All_Metadata.csv')

print(f"Enhanced dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Define features including driver metadata
base_features = ['LapNumber', 'TyreLife', 'Position', 'AirTemp', 'Humidity', 'Pressure', 
                'Rainfall', 'TrackTemp', 'WindSpeed', 'WindDirection', 'TimeSinceLastWeatherMeasurement', 
                'CircuitLength', 'Number of Laps', 'Slick', 'Wet', 'Inter', 'GapToLeader', 
                'GapToAhead', 'GapToBehind']

# Driver metadata features
driver_features = ['Age', 'Driver_Wins', 'Driver_Podiums', 'Championships', 'Races', 'Driver_Points', 'DNFs',
                  'Driver_WinRate', 'Driver_PodiumRate', 'Driver_DNFRate', 'Driver_PointsPerRace']

# Team metadata features
team_features = ['ConstructorsChampionships', 'Team_Wins', 'Team_Podiums', 'TotalRaces', 'Team_Points',
                'Team_WinRate', 'Team_PodiumRate', 'Team_PointsPerRace', 'Team_Competitiveness']

# Derived synergy features
synergy_features = ['Championship_Driver', 'Driver_Team_Synergy']

# Categorical features for embeddings
categorical_features = ['Driver', 'Team', 'Name']  # Track name

# All features combined
features = base_features + driver_features + team_features + synergy_features
target = 'LapTime_sec'

# Filter for available columns
available_features = [f for f in features if f in df.columns]
print(f"Available features: {len(available_features)} out of {len(features)}")

X = df[available_features].copy()
y = df[target]

# Handle categorical variables with label encoding for LightGBM
label_encoders = {}
for cat_feature in categorical_features:
    if cat_feature in df.columns:
        le = LabelEncoder()
        X[f'{cat_feature}_encoded'] = le.fit_transform(df[cat_feature].astype(str))
        label_encoders[cat_feature] = le
        features.append(f'{cat_feature}_encoded')

# Convert TimeSinceLastWeatherMeasurement to numeric (seconds)
if 'TimeSinceLastWeatherMeasurement' in X.columns:
    X['TimeSinceLastWeatherMeasurement'] = pd.to_timedelta(X['TimeSinceLastWeatherMeasurement']).dt.total_seconds()

# Handle categorical columns for experience, age groups, and team tier
if 'Driver_ExperienceLevel' in df.columns:
    X['Driver_ExperienceLevel_encoded'] = pd.Categorical(df['Driver_ExperienceLevel']).codes
    features.append('Driver_ExperienceLevel_encoded')

if 'Driver_AgeGroup' in df.columns:
    X['Driver_AgeGroup_encoded'] = pd.Categorical(df['Driver_AgeGroup']).codes
    features.append('Driver_AgeGroup_encoded')

if 'Team_Tier' in df.columns:
    X['Team_Tier_encoded'] = pd.Categorical(df['Team_Tier']).codes
    features.append('Team_Tier_encoded')

# Check for any remaining non-numeric columns
print("\nData types:")
print(X.dtypes)
non_numeric = X.select_dtypes(exclude=['int64', 'float64', 'bool']).columns.tolist()
if non_numeric:
    print(f"Non-numeric columns found: {non_numeric}")
    for col in non_numeric:
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Handle NaN values
print(f"\nNaN values per column:")
nan_counts = X.isnull().sum()
print(nan_counts[nan_counts > 0])

# Fill NaN values appropriately
X = X.fillna(X.median())

# Update features list to only include available encoded features
final_features = [f for f in features if f in X.columns]
X_final = X[final_features]

print(f"\nFinal feature set ({len(final_features)} features):")
for i, feature in enumerate(final_features):
    print(f"{i+1:2d}. {feature}")

print(f"\nFinal data shape: {X_final.shape}")
print(f"Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Create and train the enhanced model
model = lgb.LGBMRegressor(
    n_estimators=1000, 
    learning_rate=0.01,
    max_depth=8,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

# Evaluate
test_predictions = model.predict(X_test)
enhanced_mae = mean_absolute_error(y_test, test_predictions)
print(f"\nComprehensive Model MAE: {enhanced_mae:.3f} seconds")
print("(Includes driver metadata, team metadata, and synergy features)")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': final_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
print(feature_importance.head(15))

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the comprehensive model and encoders
model.booster_.save_model('models/comprehensive_base_pace_model.txt')
pd.to_pickle(label_encoders, 'models/comprehensive_label_encoders.pkl')
pd.to_pickle(final_features, 'models/comprehensive_feature_list.pkl')

print(f"\nComprehensive model saved successfully!")
print(f"Includes: Driver metadata + Team metadata + Synergy features")
print(f"Label encoders saved for future predictions")
