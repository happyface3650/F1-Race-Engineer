import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import os
import numpy as np
from embeddings import create_track_embeddings_dict, create_entity_embeddings, add_embeddings_to_dataframe

class BasePaceModel:
    def __init__(self, df, mae=0.317):
        self.df = df
        self.features = []
        self.target = 'LapTime_sec'
        self.mae = mae
        self.X = None
        self.y = None
        self.model = None 
        self.mae = None
        
        print("Preparing data and generating embeddings...")
        self._generate_embeddings()
        self._define_features()
        self._clean_data()

    def _generate_embeddings(self):
        # Create embedding dictionaries
        # Note: Set use_pca=False for better performance (MAE ~0.8) or use_pca=True with n_components for smaller model
        self.driver_embeddings_dict = create_entity_embeddings(self.df, 'Driver_idx', embedding_dim=16)
        self.team_embeddings_dict = create_entity_embeddings(self.df, 'Team_idx', embedding_dim=16)
        
        # Add embeddings to dataframe
        self.df = add_embeddings_to_dataframe(
            self.df, 
            self.driver_embeddings_dict, 
            self.team_embeddings_dict
        )
    
    def _define_features(self):
        base_features = [
            'LapNumber', 'Compound', 'TyreLife', 'AirTemp', 'Humidity', 
            'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed', 'Circuit_CircuitLength', 'Circuit_Number_of_Laps',
            'Circuit_NumberOfTurns', 'Circuit_AverageAngleAbs',
            'Circuit_AverageAngle',
            'GapToLeader', 'GapToAhead', 'GapToBehind',
            'status_1','status_12','status_124','status_21','status_24','status_4','status_41',
            'TimeSinceLastWeatherMeasurement',
            # Driver and Team indices
            # Note: These are not used as features directly, but are needed for embedding lookup
        ]
        
        self.driver_embed_features = [col for col in self.df.columns if col.startswith('driver_embed_')]
        self.team_embed_features = [col for col in self.df.columns if col.startswith('team_embed_')]

        self.features = base_features + self.driver_embed_features + self.team_embed_features
        self.X = self.df[self.features].copy()
        self.y = self.df[self.target].copy()
        
        print(f"Defined {len(self.features)} features")

    def _clean_data(self):
        print(f"\nCleaning data...")
        print(f"Initial shape: X={self.X.shape}, y={self.y.shape}")
        
        if 'TimeSinceLastWeatherMeasurement' in self.X.columns:
            self.X['TimeSinceLastWeatherMeasurement'] = pd.to_timedelta(
                self.X['TimeSinceLastWeatherMeasurement']
            ).dt.total_seconds()

        mask = ~self.y.isna()
        rows_before = len(self.y)
        self.X = self.X[mask]
        self.y = self.y[mask]
        self.df = self.df[mask]
        rows_dropped = rows_before - len(self.y)
        print(f"Dropped {rows_dropped} rows with NaN target ({rows_dropped/rows_before*100:.1f}%)")
        
        nan_counts = self.X.isna().sum()
        if nan_counts.sum() > 0:
            print(f"\nNaN values in features before filling:")
            print(nan_counts[nan_counts > 0])
        
        embedding_cols = [col for col in self.X.columns if 'embed' in col]
        other_cols = [col for col in self.X.columns if 'embed' not in col]
        
        if embedding_cols and self.X[embedding_cols].isna().sum().sum() > 0:
            self.X[embedding_cols] = self.X[embedding_cols].fillna(0)
            print(f"Filled {len(embedding_cols)} embedding features with 0")
        
        if other_cols and self.X[other_cols].isna().sum().sum() > 0:
            self.X[other_cols] = self.X[other_cols].fillna(self.X[other_cols].median())
            print(f"Filled {len(other_cols)} non-embedding features with median")
        
        print(f"\nData cleaning complete.")
        print(f"Final shape: X={self.X.shape}, y={self.y.shape}")
        print(f"Remaining NaN - X: {self.X.isna().sum().sum()}, y: {self.y.isna().sum()}")

    def prepare_dataframe_for_prediction(self, df):
        """
        Prepare any dataframe for prediction using the SAME embeddings as training.
        This ensures feature consistency with the trained model.
        """
        df_prepared = df.copy()
        
        # Add embeddings using the SAME embedding dictionaries from training
        df_prepared = add_embeddings_to_dataframe(
            df_prepared, 
            self.driver_embeddings_dict,
            self.team_embeddings_dict
        )
        
        # Extract the SAME features as training
        X_new = df_prepared[self.features].copy()
        
        # Apply same preprocessing
        if 'TimeSinceLastWeatherMeasurement' in X_new.columns:
            X_new['TimeSinceLastWeatherMeasurement'] = pd.to_timedelta(
                X_new['TimeSinceLastWeatherMeasurement']
            ).dt.total_seconds()
        
        # Fill NaN values
        embedding_cols = [col for col in X_new.columns if 'embed' in col]
        other_cols = [col for col in X_new.columns if 'embed' not in col]
        
        if embedding_cols:
            X_new[embedding_cols] = X_new[embedding_cols].fillna(0)
        if other_cols:
            X_new[other_cols] = X_new[other_cols].fillna(X_new[other_cols].median())
        
        print(f"Prepared {len(df_prepared)} rows for prediction with {X_new.shape[1]} features")
        
        return df_prepared, X_new

    def train(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        self.model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        print("Training model...")
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(50)]
        )
        
    def test(self):
        test_predictions = self.model.predict(self.X_test)
        self.mae = mean_absolute_error(self.y_test, test_predictions)
        print(f"\nModel training complete. Validation MAE: {self.mae:.3f} seconds")
        
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        driver_importance = feature_importance[feature_importance['feature'].str.contains('driver_embed')]['importance'].sum()
        team_importance = feature_importance[feature_importance['feature'].str.contains('team_embed')]['importance'].sum()
        
        print(f"\nEmbedding Importance:")
        print(f"  Driver embeddings: {driver_importance:.2f}")
        print(f"  Team embeddings: {team_importance:.2f}")
        
    def save(self, model_path='models/base_pace_model.txt'):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .train() before saving.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.booster_.save_model(model_path)
        print(f"Model saved successfully to {model_path}")

    def load(self, model_path='models/base_pace_model.txt'):
        self.model = lgb.Booster(model_file=model_path)
        print(f"Model loaded successfully from {model_path}")


def label_modes_with_confidence(df, model, X, confidence_threshold=1.5, mae=0.317, target='LapTime_sec'):
    if model is None:
        raise ValueError("Model not found. Train a new model with .train() or load one with .load().")
    
    predicted_pace = model.predict(X)
    delta = df[target].values - predicted_pace
    
    threshold = confidence_threshold * mae
    conditions = [
        delta < -threshold,  # Push
        delta > threshold    # Conserve
    ]
    choices = [2, 0]
    
    labeled_df = df.copy()
    labeled_df['mode'] = np.select(conditions, choices, default=1)
    labeled_df['predicted_base_pace'] = predicted_pace
    labeled_df['delta'] = delta
    
    print("\n=== Labeling Statistics ===")
    print(f"Total laps: {len(labeled_df)}")
    print(f"Mode distribution:")
    mode_counts = labeled_df['mode'].value_counts().sort_index()
    mode_names = {0: 'conserve', 1: 'base', 2: 'push'}
    for mode_num, count in mode_counts.items():
        print(f"  {mode_names[mode_num]}: {count}")
    
    return labeled_df


if __name__ == "__main__":


    df1 = pd.read_csv('ALLLAPS.csv')
    df = pd.read_csv('BASEPACELAPS.csv')
    obj = BasePaceModel(df)
    obj.train()
    obj.test()
    obj.save('models/base_pace_model.txt')
    obj1 = BasePaceModel(df1)

    labeled_df = label_modes_with_confidence(obj1.df, obj.model, obj1.X, confidence_threshold=1.5, mae=obj.mae)
    labeled_df = labeled_df.drop(obj1.driver_embed_features + obj1.team_embed_features + ['predicted_base_pace', 'delta'], axis=1)
    labeled_df.to_csv('labeled_ALL.csv', index=False)
    print("Labeled laps saved to labeled_ALL.csv")
