# Pseudocode using LightGBM
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import os

df = pd.read_csv('data\\BasePaceAllLaps.csv')
# Your filtered dataset
features = ['LapNumber', 'TyreLife', 'Position', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindSpeed', 'WindDirection', 'TimeSinceLastWeatherMeasurement', 'CircuitLength', 'Number of Laps', 'Slick', 'Wet', 'Inter', 'GapToLeader', 'GapToAhead', 'GapToBehind', ]
target = 'LapTime_sec'

X = df[features].copy()
y = df[target]

# Convert TimeSinceLastWeatherMeasurement to numeric (seconds)
if 'TimeSinceLastWeatherMeasurement' in X.columns:
    X['TimeSinceLastWeatherMeasurement'] = pd.to_timedelta(X['TimeSinceLastWeatherMeasurement']).dt.total_seconds()

# Check for any remaining non-numeric columns
print("Data types:")
print(X.dtypes)
print("\nAny non-numeric columns:")
non_numeric = X.select_dtypes(exclude=['int64', 'float64', 'bool']).columns.tolist()
if non_numeric:
    print(f"Non-numeric columns found: {non_numeric}")
    # Convert object columns to numeric where possible
    for col in non_numeric:
        X[col] = pd.to_numeric(X[col], errors='coerce')
else:
    print("All columns are numeric!")

# Handle any NaN values created during conversion
print(f"\nNaN values per column:")
print(X.isnull().sum())

# Fill NaN values with median for numeric columns
X = X.fillna(X.median())

print(f"\nFinal data shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

# Evaluate
test_predictions = model.predict(X_test)
base_mae = mean_absolute_error(y_test, test_predictions)
print(f"Base Model MAE: {base_mae:.3f} seconds")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model
model.booster_.save_model('models/base_pace_model.txt')
print("Model saved successfully!")