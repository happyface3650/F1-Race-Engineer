import fastf1
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Mapping from FastF1 location names to circuit_data names
location_mapping = {
    'Miami Gardens': 'Miami',
    'Montréal': 'Montréal',  # Handle potential encoding issues
    'Montreal': 'Montréal',
    'São Paulo': 'São Paulo',
    'Sao Paulo': 'São Paulo',
    # Add more mappings as needed
}

tyre_for_each_race = {
        '345': ['Melbourne', 'Jeddah', 'Miami', 'Spielberg', 'Monza', 'Baku'],
        '234': ['Shanghai', 'Silverstone', 'Budapest', 'Zandvoort'],
        '123': ['Suzuka', 'Sakhir', 'Barcelona', 'Marina Bay', 'Austin', 'Mexico City', 'São Paulo', 'Las Vegas', 'Lusail', 'Yas Island'],
        '456': ['Imola', 'Monaco', 'Montréal'],
        '134': ['Spa-Francorchamps']
}

circuit_data = pd.DataFrame({
    'Name'              : ['Sakhir', 'Jeddah', 'Melbourne', 'Suzuka', 'Shanghai', 
                           'Miami', 'Imola', 'Monaco', 'Montréal', 'Barcelona', 
                           'Spielberg', 'Silverstone', 'Budapest', 'Spa-Francorchamps', 'Zandvoort', 
                           'Monza', 'Baku', 'Marina Bay', 'Austin', 'Mexico City', 
                           'São Paulo', 'Las Vegas', 'Lusail', 'Yas Island'],

    'CircuitLength'    : [5.412, 6.174, 5.303, 5.807, 5.451,
                          5.412, 4.909, 3.337, 4.361, 4.655,
                          4.326, 5.891, 4.381, 7.004, 4.259, 
                          5.793, 6.003, 4.94, 5.513, 4.304,
                          4.309, 6.201, 5.419, 5.281
                          ],

    'Number of Laps'    : [57, 50, 58, 53, 56, 
                           57, 63, 78, 70, 66, 
                           71, 52, 70, 44, 72, 
                           53, 51, 62, 56, 71,
                           71, 50, 57, 58
                           ],
    'NumberOfTurns'     : [15, 27, 14, 18, 16, 
                           19, 19, 19, 14, 14, 
                           10, 18, 16, 19, 14,
                           11, 20, 19, 20, 17,
                           15, 17, 16, 16],

    'AverageAngleAbs'      : [216.48826392555523, 104.29061912054225, 89.05551872226904, 176.5787662838012, 85.05921720383002, 97.73264647184526, 96.86016529526317, 92.17177042065626, 80.46619041837172, 206.491918642282, 69.25595030609429, 83.35334581360956, 82.49930645037058, 100.01189241086918, 79.15591232751017, 90.24350869943476, 88.96085343584257, 107.3090988326499, 105.37917997627748, 97.04221228936298, 84.39904822399014, 107.92673960682204, 98.32126702370911, 74.68854478807737], #using absolute value of angles
    'AverageAngle': [-216.48826392555523, -58.73678219611126, -0.38805452678854174, -176.5787662838012, -6.794049069006466, -73.75106253395187, -32.77440998822381, 13.800079131391477, -16.00914472766793, -206.491918642282, 21.23424492222545, -4.988783677148164, 12.842773471410073, 16.089674092163328, 0.8365054834529914, 51.36152104590719, -23.597804580524446, -5.6658573384283315, -58.16669648878112, -58.946680967818814, 17.059245892889503, 9.372151834849562, -7.006878086293883, -12.815079854190804],
})
current_datetime = datetime(2025, 5, 1)

class MakeDataSet():
    def __init__(self, tyres_for_each_race, circuit_data, current_datetime, csv_path, pit, base_pace):
        self.tyres_for_each_race = tyres_for_each_race
        self.circuit_data = circuit_data
        self.current_datetime = current_datetime
        self.csv_path = csv_path
        self.schedule = fastf1.get_event_schedule(current_datetime.year) #pd dataframe
        self.past_races = self.schedule[self.schedule['Session5DateUtc'] < current_datetime]
        self.All_Laps = pd.DataFrame()
        self.pit = pit
        self.base_pace = base_pace

    def hardness_mapping(self, a, b, c):
        hardness_map = {
                    'HARD': a,
                    'MEDIUM': b,
                    'SOFT': c,
                    'INTERMEDIATE':  7,
                    'WET':  8,
                }
        return hardness_map

    def encode_tyre_compound(self, compound, hardness_map):
        self.All_Laps[compound] = self.All_Laps['Compound'].map(hardness_map)
        self.All_Laps[compound] = self.All_Laps[compound].ffill().bfill()

    def merge_weather(self, weather):
            weather['Minute'] = pd.to_timedelta(weather['Time']).dt.components['minutes']
            hourly_weather = weather[((weather['Minute'].isin([0])))].drop('Minute', axis=1)
            hourly_weather['TimeStamp'] = pd.to_timedelta(hourly_weather['Time'])

            self.All_Laps['TimeStamp'] = pd.to_timedelta(self.All_Laps['Time']) 
            
            self.All_Laps['TimeBin'] = self.All_Laps['TimeStamp'].dt.floor('h')  # Rounds down to the hour

            self.All_Laps = pd.merge_asof(
                self.All_Laps.sort_values('TimeStamp'),
                hourly_weather.sort_values('TimeStamp'),
                left_on='TimeStamp',
                right_on='Time',
                direction='backward',  # Assigns weather from the most recent hour
                tolerance=pd.Timedelta('60min')  # Only match within the same hour
            )

    def merge_circuit(self, location):
        # Map FastF1 location names to circuit_data names
        mapped_location = location_mapping.get(location, location)
        
        matching_circuits = circuit_data[circuit_data['Name'] == mapped_location]
        if matching_circuits.empty:
            raise ValueError(f"Location '{mapped_location}' (original: '{location}') not found in circuit_data.")
        circuit_data_index = matching_circuits.index[0]
        circuit_info = circuit_data.iloc[circuit_data_index]

        circuit_info = circuit_info.to_frame().T
        # Prefix circuit columns to avoid name collisions and ensure they persist
        circuit_info.columns = [f"Circuit_{c.replace(' ', '_')}" for c in circuit_info.columns]

        duplicated_rows = pd.concat([circuit_info]*len(self.All_Laps), ignore_index=True)

        self.All_Laps = pd.concat([self.All_Laps, duplicated_rows], axis=1)
        # Debug: report which circuit columns were added
        added = [c for c in self.All_Laps.columns if c.startswith('Circuit_')]
        print(f"Added circuit columns for {mapped_location}: {added}")

    def fbfill_nas(self, col):
        self.All_Laps[col] = self.All_Laps[col].ffill().bfill()
        print(f"{self.All_Laps[col].isna().sum()} missing values in {col} after forward/backward fill")

    def total(self):
        self.All_Laps['TotalTime'] = self.All_Laps.groupby('Driver')['LapTime_sec'].cumsum() #cumulative sum of laptimes
    # Rank drivers by cumulative time per lap (lower = better)
        self.All_Laps['Position'] = self.All_Laps.groupby('LapNumber')['TotalTime'].rank(method='first')
    # Get leader's time per lap

    def gap_to_lead(self):
        leader_times = self.All_Laps[self.All_Laps['Position'] == 1].set_index('LapNumber')['TotalTime']
        self.All_Laps['LeaderTime'] = self.All_Laps['LapNumber'].map(leader_times)
        self.All_Laps['GapToLeader'] = self.All_Laps['TotalTime'] - self.All_Laps['LeaderTime'] #gap to leader

    def gap_to_ahead(self):
        """
        For each lap, sort drivers by TotalTime (ascending) and take the previous
        row as the 'driver ahead' -> robust to unsorted index or missing rows.
        """
        df = self.All_Laps
        # Rank drivers by time to ensure deterministic ordering
        df['RankByTime'] = df.groupby('LapNumber')['TotalTime'].rank(method='first', ascending=True)
        sorted_df = df.sort_values(['LapNumber', 'RankByTime'])
        sorted_df['NextDriverTime'] = sorted_df.groupby('LapNumber')['TotalTime'].shift(1)
        # Assign back using original index alignment
        self.All_Laps['NextDriverTime'] = sorted_df['NextDriverTime']
        self.All_Laps['GapToAhead'] = self.All_Laps['TotalTime'] - self.All_Laps['NextDriverTime']
        # Leader has no ahead
        self.All_Laps.loc[self.All_Laps.groupby('LapNumber')['RankByTime'].transform('min') == self.All_Laps['RankByTime'], 'GapToAhead'] = 0
        self.All_Laps.drop(columns=['RankByTime'], inplace=True, errors='ignore')

    def gap_to_behind(self):
        """
        Symmetric of gap_to_ahead: sort by TotalTime then look at the next row for the 'driver behind'.
        """
        df = self.All_Laps
        df['RankByTime'] = df.groupby('LapNumber')['TotalTime'].rank(method='first', ascending=True)
        sorted_df = df.sort_values(['LapNumber', 'RankByTime'])
        sorted_df['PrevDriverTime'] = sorted_df.groupby('LapNumber')['TotalTime'].shift(-1)
        self.All_Laps['PrevDriverTime'] = sorted_df['PrevDriverTime']
        self.All_Laps['GapToBehind'] = self.All_Laps['PrevDriverTime'] - self.All_Laps['TotalTime']
        # Last-placed driver has no behind
        max_rank = self.All_Laps.groupby('LapNumber')['RankByTime'].transform('max')
        self.All_Laps.loc[self.All_Laps['RankByTime'] == max_rank, 'GapToBehind'] = 0
        self.All_Laps.drop(columns=['RankByTime'], inplace=True, errors='ignore')



    def handle_last_place(self):
        self.All_Laps.loc[self.All_Laps['Position'] == self.All_Laps.groupby('LapNumber')['Position'].transform('max'), 'GapToBehind'] = 0
        self.All_Laps.loc[self.All_Laps['Position'] == 1, 'NextDriverTime'] = 0
        self.All_Laps.loc[self.All_Laps['Position'] == 1, 'GapToAhead'] = 0

    def tyre_life(self):
        self.All_Laps = self.All_Laps.sort_values(['DriverNumber', 'LapNumber'])
        self.All_Laps['TyreLife'] = np.where(
            self.All_Laps['FreshTyre'] == True,
            1,
            self.All_Laps['TyreLife']
        )

        self.All_Laps['TyreLife'] = self.All_Laps.groupby(['Driver', 'Compound'])['LapNumber'].transform(
            lambda x: x - x.min() + 1  # TyreLife = laps since first use of this compound
        )
    def one_hot_track_status(self):
        one_hot = pd.get_dummies(
        self.All_Laps['TrackStatus'], 
            prefix='status'
        )
        # Remove any existing status columns to avoid duplicates
        status_cols = [col for col in self.All_Laps.columns if col.startswith('status_')]
        self.All_Laps = self.All_Laps.drop(columns=status_cols)
        # Add the new one-hot encoded status columns
        self.All_Laps = pd.concat([self.All_Laps, one_hot.astype(int)], axis=1)

    def time_since_last_weather(self):
        self.All_Laps["TimeSinceLastWeatherMeasurement"] = pd.to_timedelta(self.All_Laps["Time_x"]) - pd.to_timedelta(self.All_Laps["TimeStamp_y"])

    def csv(self):
        self.All_Laps.to_csv(self.csv_path, index=False)

# Get unique race names and shuffle
    def train_val_test(self):
        All_Laps = pd.read_csv(self.csv_path)
        all_races = All_Laps['Name'].unique()
        np.random.shuffle(all_races)  # Randomize to avoid season bias

        # Split ratios (adjust as needed)

        train_races, val_races, test_races = np.split(
            all_races, 
            [int(0.7 * len(all_races)), int(0.85 * len(all_races))]
        )

        # Create splits

        train = All_Laps[All_Laps['Name'].isin(train_races)]
        train.to_csv('All_Laps_Train.csv', index=False)
        unique_values = train['Name'].unique()
        print(unique_values)

        val = All_Laps[All_Laps['Name'].isin(val_races)]
        unique_values = val['Name'].unique()
        print(unique_values)
        val.to_csv('All_Laps_val.csv', index=False)

        test = All_Laps[All_Laps['Name'].isin(test_races)]
        test.to_csv('All_Laps_test.csv', index=False)
        unique_values = test['Name'].unique()
        print(unique_values)
    def driver_and_teams_to_int(self):
        # Create mappings from categories to integer indices
        driver_to_idx = {driver: idx for idx, driver in enumerate(self.All_Laps['Driver'].unique())}
        team_to_idx = {team: idx for idx, team in enumerate(self.All_Laps['Team'].unique())}

        # Convert to indices
        self.All_Laps['Driver_idx'] = self.All_Laps['Driver'].map(driver_to_idx)
        self.All_Laps['Team_idx'] = self.All_Laps['Team'].map(team_to_idx)

    def clean_data_for_base_pace(self):
        self.All_Laps = self.All_Laps[self.All_Laps['GapToAhead'] >= 2]
        self.All_Laps = self.All_Laps[self.All_Laps['GapToBehind'] >= 2]
        self.All_Laps = self.All_Laps[self.All_Laps['dnf'] == 0]
        self.All_Laps = self.All_Laps[self.All_Laps['PitDuration'] == 0.0]
        # Only filter by status_4 if the column exists
        if 'status_1' in self.All_Laps.columns:
            self.All_Laps = self.All_Laps[self.All_Laps['status_1'] == 1]
        self.All_Laps = self.All_Laps.drop(['PitDuration', 'SpeedFL', 'SpeedST'], axis=1)

    def no_pits(self):
        # Filter for laps where there were no pit stops (NaT values)
        self.All_Laps = self.All_Laps[self.All_Laps['PitInTime'].isna()]
    
    def have_pits(self):
        # Filter for laps where there were pit stops (not NaT values)
        self.All_Laps = self.All_Laps[self.All_Laps['PitInTime'].notna()]

    def DNF(self):
        """
        Mark dnf only on the final lap row for drivers whose last lap is less
        than the race's maximum lap. (Previously this flagged all laps for that driver.)
        """
        race_max_lap = self.All_Laps['LapNumber'].max()
        driver_last_lap = self.All_Laps.groupby('Driver')['LapNumber'].transform('max')
        dnf_last_lap_mask = (self.All_Laps['LapNumber'] == driver_last_lap) & (driver_last_lap < race_max_lap)
        self.All_Laps['dnf'] = dnf_last_lap_mask.astype(int)

    def fill_laptimes_with_pit_duration(self):
        """
        Fill missing lap times by adding pit duration to normal lap time
        """
        pit_lap_mask = self.All_Laps['LapTime_sec'].isna() & self.All_Laps['PitDuration'].notna() & (self.All_Laps['PitDuration'] > 0)
        self.All_Laps.loc[pit_lap_mask, 'LapTime_sec'] = self.All_Laps.loc[pit_lap_mask, 'LapTime_sec'].ffill().bfill() 

    def drop_cols(self):
        # Columns to drop after all processing
        cols_to_drop = [
            # Time/timestamp columns (converted to numeric or not needed)
            'Time_x', 'LapStartTime', 'TimeStamp_x', 'TimeStamp_y', 'TimeBin', 'Time_y',
            
            # Categorical identifiers (use embeddings instead)
            'Driver', 'Team',
            
            # Pit stop columns (converted to numeric or intermediate calculations)
            'PitInTime', 'PitOutTime', 'NextPitOutTime', 'FreshTyre',
            
            # Track status (one-hot encoded)
            'TrackStatus',
            
            # Sector times (not needed for lap time prediction)
            'Sector1Time', 'Sector2Time', 'Sector3Time', 
            'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
            
            # FastF1 metadata columns
            'Deleted', 'DeletedReason', 'FastF1Generated', 'IsAccurate',
            
            # Intermediate calculation columns
            'TotalTime', 'LeaderTime', 'NextDriverTime', 'PrevDriverTime',
            
            # Speed columns 
            'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
            
            # Index columns
            'Unnamed: 0', 'Unnamed: 0.1',
            
            # Stint info (redundant with TyreLife)
            'Stint', 'DriverNumber',
            
            # Lap metadata
            'IsPersonalBest', 'LapStartDate',
            
            # Original lap time (keeping LapTime_sec)
            'LapTime', 'Position', 

            'status_14', 'status_126', 'status_16', 'status_167', 'status_67', 
            'status_671', 'status_71', 'status_26', 'status_6', 'status_2', 
            'status_6712', 'status_712'
        ]

        if not self.pit:
            cols_to_drop.append('PitDuration')
        
        # Only drop columns that actually exist in the dataframe
        existing_cols_to_drop = [col for col in cols_to_drop if col in self.All_Laps.columns]
        self.All_Laps = self.All_Laps.drop(existing_cols_to_drop, axis=1)
    def pit_duration(self):
        # Calculate pit durations without filtering the data
        self.All_Laps['NextPitOutTime'] = self.All_Laps.groupby('Driver')['PitOutTime'].shift(-1)
        # Compute duration only where PitInTime is not null
        self.All_Laps['PitDuration'] = (
            (self.All_Laps['NextPitOutTime'] - self.All_Laps['PitInTime'])
        ).fillna(0)
    def convert_to_numbers(self):
        self.All_Laps.loc[:, 'TimeSinceLastWeatherMeasurement'] = pd.to_timedelta(self.All_Laps['TimeSinceLastWeatherMeasurement']).dt.total_seconds()
        self.All_Laps['Rainfall'] = (
            self.All_Laps['Rainfall']
            .fillna(False)
            .astype(int)
        )
        self.All_Laps.loc[:, 'dnf'] = self.All_Laps['dnf'].astype(int)

    def normalize(self):
        """
        Normalizes columns using a mixed strategy optimized for Transformers.
        - Gaussian-like: StandardScaler
        - Skewed: Log1p + StandardScaler
        - Sequential: Min-Max
        - Binary/Flags: Filled with 0
        - Categorical: Excluded (to be handled via Embeddings)
        """
        
        # 1. Define Column Groups
        # Gaussian-like continuous features
        gaussian_cols = [
            'AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindDirection', 
            'WindSpeed', 'TimeSinceLastWeatherMeasurement', 'CircuitLength', 
            'Number of Laps', 'LapTime_sec', 'Circuit_CircuitLength', 'Circuit_Number_of_Laps',
            'Circuit_NumberOfTurns', 'Circuit_AverageAngleAbs', 'Circuit_AverageAngle'
        ]
        
        # Skewed features (Gaps often follow a power-law distribution)
        skewed_cols = ['GapToLeader', 'GapToAhead', 'GapToBehind']
        
        # Sequential/Counter features
        sequential_cols = ['LapNumber', 'TyreLife']
        
        # Dynamic list of status flags
        status_cols = [col for col in self.All_Laps.columns if col.startswith('status_')]
        
        # 2. Pre-processing: Fill NaNs
        # Transformers cannot handle NaNs. We fill gaps and status flags with 0.
        self.All_Laps = self.All_Laps.fillna(0)

        # 3. Apply Transformations
        
        # A. Skewed Gaps: Log Transform then Standardize
        # np.log1p handles 0 values safely as log(1 + x)
        for col in skewed_cols:
            if col in self.All_Laps.columns:
                self.All_Laps[col] = np.log1p(self.All_Laps[col])
                ss_skew = StandardScaler()
                self.All_Laps[[col]] = ss_skew.fit_transform(self.All_Laps[[col]])

        # B. Gaussian features: Standardization (Mean=0, Std=1)
        gaussian_to_process = [c for c in gaussian_cols if c in self.All_Laps.columns]
        if gaussian_to_process:
            ss_gauss = StandardScaler()
            self.All_Laps[gaussian_to_process] = ss_gauss.fit_transform(self.All_Laps[gaussian_to_process])

        # C. Sequential features: Min-Max Scaling [0, 1]
        sequential_to_process = [c for c in sequential_cols if c in self.All_Laps.columns]
        if sequential_to_process:
            mm_scaler = MinMaxScaler()
            self.All_Laps[sequential_to_process] = mm_scaler.fit_transform(self.All_Laps[sequential_to_process])
        
        print("Normalization complete: Gaussian (StandardScaler), Skewed (Log+SS), Sequential (MinMax).")

    def subtract_pit_duration_from_laptime(self):
        self.All_Laps['LapTime_sec'] = self.All_Laps['LapTime_sec'] - self.All_Laps['PitDuration']
    def create_dataset(self):
        all_races_data = []  # List to accumulate data from all races
        
        for location in self.past_races['Location']:
            # Map FastF1 location names to circuit_data names for tyre lookup
            mapped_location = location_mapping.get(location, location)
            
            hardness_map = None
            for key in self.tyres_for_each_race:
                if mapped_location in self.tyres_for_each_race[key]:
                    # Extract the three numbers from the key string
                    a, b, c = int(key[0]), int(key[1]), int(key[2])
                    hardness_map = self.hardness_mapping(a, b, c)
                    break
            
            if hardness_map is None:
                print(f"Warning: No tyre mapping found for location '{location}' (mapped: '{mapped_location}'), skipping this race.")
                continue  # Skip this race if no tyre mapping is found
            
            race = fastf1.get_session(self.current_datetime.year, location,'R' )
            race.load(weather=True, messages=False)
            Laps = race.laps
            weather = race.weather_data #need to drop  things that weather forecast APIs dont have (by the minute)
                
            self.All_Laps = Laps  # Start with the laps data
            
            self.merge_weather(weather)
            self.merge_circuit(location)
                    
            self.All_Laps = self.All_Laps.sort_values(['Driver', 'LapNumber'])

            self.encode_tyre_compound('Compound', hardness_map)
            
            self.All_Laps['LapTime_sec'] = self.All_Laps['LapTime'].dt.total_seconds()
            self.pit_duration()
            self.All_Laps['PitDuration'] = pd.to_timedelta(self.All_Laps['PitDuration']).dt.total_seconds()
            self.fill_laptimes_with_pit_duration()

            self.fbfill_nas('LapTime_sec')
            
            self.All_Laps = self.All_Laps.sort_values(['Driver', 'LapNumber'])
            self.total()
            self.gap_to_lead()
            self.gap_to_ahead()
            self.gap_to_behind()
            self.handle_last_place()
            gap_cols = ['GapToLeader', 'GapToAhead', 'GapToBehind']

            self.All_Laps[gap_cols] = (
                self.All_Laps
                .sort_values(['Driver', 'LapNumber'])
                .groupby('Driver')[gap_cols]
                .shift(1)
            )
            self.All_Laps[gap_cols] = self.All_Laps[gap_cols].fillna(0.0)
            
            self.tyre_life()
            self.one_hot_track_status()
            self.time_since_last_weather()
            
            self.driver_and_teams_to_int()
            
            self.DNF()
            
            if self.pit:
                self.have_pits()
                self.subtract_pit_duration_from_laptime()
            else:   
                self.no_pits()
            if self.base_pace:
                self.clean_data_for_base_pace()
            
            self.convert_to_numbers()
            self.drop_cols()
            
            
            # Add processed race data to the list
            all_races_data.append(self.All_Laps.copy())
            print(f"Processed {location} - {len(self.All_Laps)} laps")
        
        # Combine all race data
        if all_races_data:
            self.All_Laps = pd.concat(all_races_data, ignore_index=True)
            self.normalize()
            print(f"Total dataset created with {len(self.All_Laps)} laps from {len(all_races_data)} races")
        else:
            print("No race data was processed successfully")
            self.All_Laps = pd.DataFrame()
        
        return self.All_Laps
data_processor = MakeDataSet(tyre_for_each_race, circuit_data, current_datetime, 'ALLLAPS.csv', False, False)
All_Laps = data_processor.create_dataset()
All_Laps.to_csv('ALLLAPS.csv', index=False)

#add function to use base pace model 
#label
#remove base_pace and delta
#subtract pit duration from laptime
#spit train val test

'''


def circuit_info(race):
        avg=[]
        std=[]
        circuit = race.get_circuit_info()
        total_angle = 0
        count = 0
        for angle in circuit.corners['Angle']:
            total_angle = total_angle + angle
            count = count + 1
        
        average = total_angle/count
        avg.append(average)

        std_sum=0
        for angle in circuit.corners['Angle']:
            std_sum = std_sum + (angle - average)*(angle - average)
        
        std_deviation = math.sqrt(std_sum/(count-1))
        std.append(std_deviation)

- embedded encoding driver and team
- one hot encoding for tyre compound
'''