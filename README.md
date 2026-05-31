# F1 Race Engineer 🏎️

An AI-powered Formula 1 lap time prediction system using Transformer architecture. Predicts race lap times with 0.90s RMSE accuracy by analyzing tyre strategy, weather conditions, track characteristics, and race positioning.

## 🎯 Overview

This project simulates the role of an F1 race engineer by predicting lap times in real-time based on:
- **Tyre Strategy**: Compound type, tyre life, stint progression
- **Weather Conditions**: Temperature, humidity, rainfall, wind
- **Track Characteristics**: Circuit length, number of turns, corner complexity
- **Race Dynamics**: Gap to leader, gap to car ahead/behind
- **Driver & Team**: Historical performance patterns via embeddings

## 🏆 Performance

- **RMSE**: 0.90 seconds
- **MAE**: 0.64 seconds
- **Median Error**: 0.48 seconds
- **92.6%** of predictions within 1 second
- **96.7%** of predictions within 2 seconds

## 🏗️ Architecture

### Model: Stint Transformer
- **Input**: Sequential lap data organized by racing stints
- **Embeddings**: Driver, Team, Tyre Compound, Race Mode
- **Encoder**: Multi-head self-attention with positional encoding
- **Output**: Lap time prediction (seconds)

### Key Features
- Handles variable-length stints via padding and masking
- Captures temporal dependencies across laps
- Learns driver-specific and team-specific patterns
- Properly lags gap features to prevent data leakage

## 📊 Data Pipeline

### Data Sources
- **FastF1**: Official F1 timing data (lap times, weather, telemetry)
- **Circuit Data**: Track characteristics and corner analysis
- **Tyre Compounds**: Race-specific tyre hardness mappings

### Feature Engineering
1. **Temporal Features**: Lap number (normalized), tyre life
2. **Weather Features**: Air/track temp, humidity, pressure, wind, rainfall
3. **Position Features**: Gap to leader/ahead/behind (lagged by 1 lap)
4. **Circuit Features**: Length, number of laps/turns, average corner angle
5. **Status Features**: Track status one-hot encoding (green flag, yellow flag, etc.)
6. **Categorical Embeddings**: Driver, Team, Tyre Compound, Mode

### Preprocessing
- **Normalization Strategy**:
  - Gaussian features: StandardScaler (mean=0, std=1)
  - Skewed features (gaps): Log1p + StandardScaler
  - Sequential features (lap number, tyre life): MinMaxScaler [0,1]
- **Data Leakage Prevention**: 
  - Gaps lagged by 1 lap (use previous lap's position)
  - DNF flag excluded from features
  - Target variable (LapTime_sec) never in feature set

## 🚀 Installation

### Requirements
```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `torch` - PyTorch deep learning framework
- `fastf1` - F1 data API
- `pandas` - Data manipulation
- `scikit-learn` - Preprocessing and metrics
- `joblib` - Model serialization

## 💻 Usage

### 1. Data Collection & Processing
```python
from data_cleaning.dataframe import MakeDataSet
import datetime

# Initialize dataset generator
dataset = MakeDataSet(
    tyres_for_each_race=tyre_mappings,
    circuit_data=circuit_df,
    current_datetime=datetime.datetime(2024, 12, 31),
    csv_path='labeled_ALL.csv',
    pit=False,  # Filter out pit laps
    base_pace=False,  # Include all laps
    test=False
)

# Create dataset from past races
df = dataset.create_dataset()
```

### 2. Training
```python
from dataloader import assign_stints, StintDataset, collate_fn
from model import StintTransformer
import torch

# Load and prepare data
df = pd.read_csv('labeled_ALL.csv')
df = assign_stints(df)

# Define features
cont_features = [
    'LapNumber', 'TyreLife', 'AirTemp', 'Humidity', 'Pressure', 
    'TrackTemp', 'WindDirection', 'WindSpeed', 'GapToLeader', 
    'GapToAhead', 'GapToBehind', 'Circuit_CircuitLength', 
    'Circuit_Number_of_Laps', 'Circuit_NumberOfTurns', 
    'Circuit_AverageAngleAbs', 'Circuit_AverageAngle',
    'status_1', 'status_12', 'status_124', 'status_21', 
    'status_24', 'status_4', 'status_41', 
    'TimeSinceLastWeatherMeasurement'
]

# Create dataset and dataloader
dataset = StintDataset(df, cont_features)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, 
                       collate_fn=collate_fn)

# Initialize and train model
model = StintTransformer(
    n_drivers=df['Driver_idx'].nunique(),
    n_teams=df['Team_idx'].nunique(),
    n_tyres=df['Compound'].max(),
    n_modes=df['mode'].nunique(),
    n_cont_features=len(cont_features)
)

# Training loop (see training.py for full implementation)
```

### 3. Inference
```python
# Load trained model
model.load_state_dict(torch.load('stint_transformer_model.pth'))
model.eval()

# Load scalers for denormalization
scalers = joblib.load('labeled_ALL_scalers.joblib')

# Make predictions
with torch.no_grad():
    predictions = model(cont_feats, driver_idx, team_idx, 
                       tyre_idx, mode_idx, mask)
```

### 4. Error Analysis
```python
from calculate_errors import print_error_report

# Print comprehensive error statistics
errors, distribution = print_error_report('PREDICTED.csv')
```

## 📁 Project Structure

```
F1-Race-Engineer/
├── data_cleaning/
│   └── dataframe.py           # MakeDataSet class for data processing
├── data_manipulation/
│   ├── base_pace.py          # Base pace calculation
│   └── embeddings.py         # Embedding utilities
├── data/
│   ├── All_Laps.csv          # Processed lap data
│   ├── F1DriversDataset.csv # Driver metadata
│   └── team_metadata.csv    # Team metadata
├── IFCNN/                    # CNN-based experimental models
├── scripts/
│   └── test_scheduler_run.py
├── model.py                  # StintTransformer architecture
├── dataloader.py            # PyTorch Dataset and collate functions
├── training.py              # Training loop and model training
├── calculate_errors.py      # Error analysis utilities
├── labeled_ALL.csv          # Full training dataset
├── PREDICTED.csv            # Model predictions with residuals
├── stint_transformer_model.pth  # Trained model weights
├── labeled_ALL_scalers.joblib   # Feature scalers
└── requirements.txt
```

## 🔬 Model Architecture Details

### StintTransformer Components

**Embeddings**:
- Driver Embedding: 64 dimensions
- Team Embedding: 32 dimensions
- Tyre Compound Embedding: 16 dimensions
- Mode Embedding: 8 dimensions

**Transformer Encoder**:
- Hidden dimension: 128
- Number of attention heads: 4
- Number of layers: 2
- Dropout: 0.1
- Feedforward dimension: 512

**Output**:
- Dense layer: 128 → 64 → 1
- Activation: ReLU
- Output: Single lap time prediction (seconds)

## 📈 Key Insights

### What Makes It Accurate?
1. **Stint-based modeling**: Groups laps by tyre stints, capturing degradation patterns
2. **Temporal attention**: Self-attention learns lap-to-lap dependencies
3. **Proper lagging**: Gap features use previous lap data to avoid leakage
4. **Rich embeddings**: Driver and team embeddings capture performance characteristics
5. **Weather integration**: Real-time weather conditions impact predictions

### Feature Importance
- Tyre life and compound: Primary degradation indicators
- Gap to leader/ahead/behind: Racing intensity and traffic
- Weather conditions: Track grip and car performance
- Circuit characteristics: Baseline lap time expectations
- Driver/Team: Consistent performance patterns

## 🛠️ Future Improvements

- [ ] trying multiseason model, with extended features to compare performance
- [ ] trying different architectures such as RNN, LSTM, Gradient Boosting, Random Forest etc.
- [ ] ablations and failure analysis
- [ ] risk model
- [ ] strategy

## 📊 Data Leakage Prevention

This project carefully addresses data leakage:

✅ **Properly Handled**:
- Gap features lagged by 1 lap
- Target variable excluded from features
- Train/val/test split by race (no same-race leakage)
- Normalized after feature engineering

⚠️ **Known Limitations**:
- DNF flag removed (caused leakage, minimal impact: 0.26% of laps)

## 📝 Notes

- Model trained on 2025 F1 season data
- Predictions assume normal racing conditions (no safety cars, red flags)
- Best performance on dry weather races
- Circuit-specific characteristics encoded via embeddings and explicit features

## 🤝 Contributing

This is a research project exploring ML applications in motorsport analytics. Feedback and suggestions welcome!

## 📄 License

Educational/Research use only. F1 data provided by FastF1 library.

---

**Built with**: PyTorch • FastF1 • Transformers • Formula 1 Data
