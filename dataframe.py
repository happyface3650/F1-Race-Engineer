import fastf1
from datetime import datetime
import math
import pandas as pd
from PIL import Image
import os
import numpy as np

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
   
    })
'''
    image data already has this but we can use thsi to test for better performance 
    'NumberOfTurns'     : [15, 27, 14, 18, 16, 
                           19, 19, 19, 14, 14, 
                           10, 18, 16, 19, 14,
                           11, 20, 19, 20, 17,
                           15, 17, 16, 16],

    'AverageAngleAbs'      : [216.48826392555523, 104.29061912054225, 89.05551872226904, 176.5787662838012, 85.05921720383002, 97.73264647184526, 96.86016529526317, 92.17177042065626, 80.46619041837172, 206.491918642282, 69.25595030609429, 83.35334581360956, 82.49930645037058, 100.01189241086918, 79.15591232751017, 90.24350869943476, 88.96085343584257, 107.3090988326499, 105.37917997627748, 97.04221228936298, 84.39904822399014, 107.92673960682204, 98.32126702370911, 74.68854478807737], #using absolute value of angles
    'StandardDeviationAbs' : [458.8321683105066, 201.21858500671442, 141.92319960989195, 378.75640168691336, 134.24976856539408, 199.24054987097637, 172.03360900844714, 136.68541610904137, 144.42911520242296, 442.8473776098799, 97.7800636367041, 135.11085264375768, 126.41269070362975, 147.02821846563003, 127.56801371284563, 104.98507600629529, 157.39691493147953, 174.90917801992327, 200.19981762095892, 191.06166466444748, 118.73844523747309, 161.6280596206068, 156.92287708079837, 129.32895027770468], #using absolute value of angles
    'AverageAngle': [-216.48826392555523, -58.73678219611126, -0.38805452678854174, -176.5787662838012, -6.794049069006466, -73.75106253395187, -32.77440998822381, 13.800079131391477, -16.00914472766793, -206.491918642282, 21.23424492222545, -4.988783677148164, 12.842773471410073, 16.089674092163328, 0.8365054834529914, 51.36152104590719, -23.597804580524446, -5.6658573384283315, -58.16669648878112, -58.946680967818814, 17.059245892889503, 9.372151834849562, -7.006878086293883, -12.815079854190804],
    'StandardDeviation': [98.32428722681023, 113.52866227268522, 107.36221599361905, 106.76930335283502, 94.99215581583938, 93.03994190032616, 108.8890332818007, 110.45158206929294, 104.09763798932771, 111.52795445066761, 83.65778553357201, 99.95765261078193, 103.9454998793724, 119.09275344827954, 98.32511329440977, 96.74128551848803, 106.94636084950332, 130.84649536460861, 109.20147925529666, 103.20485191367843, 96.1263018168946, 125.71221087323826, 113.09805475271118, 92.51288612600074],
'''


'''
        3 diff tyre compounds are picked each race  (soft med hard) are refering to diff coumpounds for each race

        slicks: 1, 2, 3, 4, 5, 6

        melbourne, saudi arabia, miami, austrian: 3, 4, 5
        shanghai, silverstone: 2, 3, 4
        suzuka, bahrain, spain: 1, 2, 3
        imola, monaco, montreal: 4, 5, 6
        belgian: 1, 3, 4 ???
        
'''
tyre_for_each_race = {
        'c345': ['Melbourne', 'Jeddan', 'Miami', 'Spielberg'],
        'c234': ['Shanghai', 'Silverstone'],
        'c123': ['Suzuka', 'Sakhir', 'Barcelona'],
        'c456': ['Imola', 'Monaco', 'Montreal'],
        'c134': ['Spa-Francorchamps']
}

current_datetime = datetime.utcnow()
schedule = fastf1.get_event_schedule(current_datetime.year) #pd dataframe
past_races = schedule[schedule['Session5DateUtc'] < current_datetime]
wet_map = dict()
wet_map['WET'] = 1
inter_map = dict()
inter_map['INTERMEDIATE'] = 1

All_Laps = pd.DataFrame()

def hardness_mapping( a, b, c):
     hardness_map = {
                    'HARD': a,
                    'MEDIUM': b,
                    'SOFT': c,
                }
     return hardness_map

def encode_tyre_compound(Laps_WC, compound,hardness_map):
    Laps_WC[compound] = Laps_WC['Compound'].map(hardness_map)
    Laps_WC[compound] = Laps_WC[compound].fillna(0)
     
def merge_weather(weather, Laps):
            weather['Minute'] = pd.to_timedelta(weather['Time']).dt.components['minutes']
            hourly_weather = weather[((weather['Minute'].isin([0])))].drop('Minute', axis=1)
            hourly_weather['TimeStamp'] = pd.to_timedelta(hourly_weather['Time'])

            Laps['TimeStamp'] = pd.to_timedelta(Laps['Time']) 
            
            Laps['TimeBin'] = Laps['TimeStamp'].dt.floor('h')  # Rounds down to the hour

            Laps_with_weather = pd.merge_asof(
                Laps.sort_values('TimeStamp'),
                hourly_weather.sort_values('TimeStamp'),
                left_on='TimeStamp',
                right_on='Time',
                direction='backward',  # Assigns weather from the most recent hour
                tolerance=pd.Timedelta('60min')  # Only match within the same hour
            )
            return Laps_with_weather

def merge_circuit(Laps_with_weather, location):
    circuit_data_index = circuit_data[circuit_data['Name'] == location].index[0]
    circuit_info = circuit_data.iloc[circuit_data_index]

    circuit_info = circuit_info.to_frame().T

    duplicated_rows = pd.concat([circuit_info]*len(Laps_with_weather), ignore_index=True)

    Laps_WC = pd.concat([Laps_with_weather, duplicated_rows], axis=1)

    return Laps_WC

def fbfill_nas(col, Laps_WC):
    Laps_WC['SpeedFL'].ffill(inplace=True)
    Laps_WC['SpeedFL'].bfill(inplace=True)# type: ignore # Critical step to match original index

def total(Laps_WC):
    Laps_WC['TotalTime'] = Laps_WC.groupby('Driver')['LapTime_sec'].cumsum() #cumulative sum of laptimes
    # Rank drivers by cumulative time per lap (lower = better)
    Laps_WC['Position'] = Laps_WC.groupby('LapNumber')['TotalTime'].rank(method='first')
    # Get leader's time per lap

def gap_to_lead(Laps_WC):
    leader_times = Laps_WC[Laps_WC['Position'] == 1].set_index('LapNumber')['TotalTime']
    Laps_WC['LeaderTime'] = Laps_WC['LapNumber'].map(leader_times)
    Laps_WC['GapToLeader'] = Laps_WC['TotalTime'] - Laps_WC['LeaderTime'] #gap to leader

def gap_to_ahead(Laps_WC):
    Laps_WC['NextDriverTime'] = Laps_WC.groupby('LapNumber')['TotalTime'].shift(1)  # Time of driver ahead
    Laps_WC['GapToAhead'] = Laps_WC['TotalTime'] - Laps_WC['NextDriverTime']

def gap_to_behind(Laps_WC):
    # Shift data to compare with the previous driver
    Laps_WC['PrevDriverTime'] = Laps_WC.groupby('LapNumber')['TotalTime'].shift(-1)  # Time of driver behind
    Laps_WC['GapToBehind'] = Laps_WC['PrevDriverTime'] - Laps_WC['TotalTime']

    # Handle last/first-place driver 
    Laps_WC.loc[Laps_WC['Position'] == Laps_WC.groupby('LapNumber')['Position'].transform('max'), 'GapToBehind'] = 0
    Laps_WC.loc[Laps_WC['Position'] == 1, 'NextDriverTime'] = 0
    Laps_WC.loc[Laps_WC['Position'] == 1, 'GapToAhead'] = 0
def tyre_life(Laps_WC):
    Laps_WC = Laps_WC.sort_values(['DriverNumber', 'LapNumber'])
    Laps_WC['TyreLife'] = np.where(
        Laps_WC['FreshTyre'] == True,
        1,
        Laps_WC['TyreLife']
    )

    Laps_WC['TyreLife'] = Laps_WC.groupby(['Driver', 'Compound'])['LapNumber'].transform(
        lambda x: x - x.min() + 1  # TyreLife = laps since first use of this compound
    )

def time_since_last_weather(df):
    df["TimeSinceLastWeatherMeasurement"] = pd.to_timedelta(df["Time_x"]) - pd.to_timedelta(df["TimeStamp_y"])

    return df

def make(All_Laps):
    for location in past_races['Location']:
            if location in tyre_for_each_race['c345']:
                hardness_map = hardness_mapping( 3, 4, 5)
            if location in tyre_for_each_race['c234']:
                hardness_map = hardness_mapping( 2, 3, 4)
            if location in tyre_for_each_race['c123']:
                hardness_map = hardness_mapping( 1, 2, 3)
            if location in tyre_for_each_race['c456']:
                hardness_map = hardness_mapping( 4, 5, 6)
        
            race = fastf1.get_session(current_datetime.year, location,'R' )
            race.load(weather=True, messages=False)
            Laps = race.laps
            weather = race.weather_data #need to drop  things that weather forecast APIs dont have (by the minute)
            
            Laps_with_weather = merge_weather(weather, Laps)
            
            Laps_WC = merge_circuit(Laps_with_weather, location)
                
            Laps_WC = Laps_WC.sort_values(['DriverNumber', 'LapNumber'])

            encode_tyre_compound(Laps_WC, 'Slick',hardness_map)
            encode_tyre_compound(Laps_WC, 'Wet',wet_map)
            encode_tyre_compound(Laps_WC, 'Inter',inter_map)
            
            Laps_WC ['LapTime_sec'] = Laps_WC ['LapTime'].dt.total_seconds()

            fbfill_nas('SpeedFL', Laps_WC)
            fbfill_nas('SpeedST', Laps_WC)
            fbfill_nas('LapTime_sec', Laps_WC)
            
            Laps_WC = Laps_WC.sort_values(['LapNumber', 'Position'])
            total(Laps_WC)
            gap_to_lead(Laps_WC)
            gap_to_ahead(Laps_WC)
            gap_to_behind(Laps_WC)
            tyre_life(Laps_WC)
            #pd.set_option('display.max_rows', None)
            Laps_WC['PitInTime'] = Laps_WC['PitInTime'].fillna('NO_PIT')
            Laps_WC['PitOutTime'] = Laps_WC['PitOutTime'].fillna('NO_PIT')

            Laps_WC = Laps_WC.drop(['Stint', 'DriverNumber', 'IsPersonalBest', 'LapStartDate', 
                                    'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime', 
                                    'Sector3SessionTime', 'Deleted', 'DeletedReason', 'FastF1Generated', 
                                    'IsAccurate', 'TimeStamp_x', 'TimeBin', 'Time_y', 'LapTime',
                                    'TotalTime', 'LeaderTime', 'NextDriverTime', 'PrevDriverTime', 
                                    'SpeedI1', 'SpeedI2', 'Compound'], axis=1)
            All_Laps = pd.concat([All_Laps, Laps_WC], axis=0)

def csv(All_Laps):
    All_Laps.to_csv('All_Laps.csv', index=False)

def one_hot_track_status(df):
    one_hot = pd.get_dummies(
    df['TrackStatus'], 
        prefix='status'
    )
    df = pd.concat([df, one_hot.astype(int)], axis=1)

    return df
     
# Get unique race names and shuffle
def train_val_test():
    All_Laps = pd.read_csv('All_Laps1.csv')
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

def driver_and_teams_to_int(df):
    # Create mappings from categories to integer indices
    driver_to_idx = {driver: idx for idx, driver in enumerate(df['Driver'].unique())}
    team_to_idx = {team: idx for idx, team in enumerate(df['Team'].unique())}

    # Convert to indices
    df['Driver_idx'] = df['Driver'].map(driver_to_idx)
    df['Team_idx'] = df['Team'].map(team_to_idx)

    df = df.drop(['Driver', 'Team'], axis=1)

    return df



'''
- embedded encoding driver and team
- one hot encoding for tyre compound
- output: laptime, position
'''




        





        
        






    

    







    





   



   
  








    





