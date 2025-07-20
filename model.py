import fastf1
from datetime import datetime
import numpy as np
import pandas as pd

big_ahh_3d_array = np.zeros((0,0,0))
current_datetime = datetime.utcnow()


schedule = fastf1.get_event_schedule(current_datetime.year) #pd dataframe
past_races = schedule[schedule['Session5DateUtc'] < current_datetime]

upcoming_races = schedule[schedule['Session5DateUtc'] > current_datetime]

all_laps = pd.DataFrame()
weather_by_lap = pd.DataFrame()


for location in past_races['Location']:
    race = fastf1.get_session(current_datetime.year, location,'R' )
    race.load(weather=True, messages=False)
    Laps = race.laps
    weather = race.weather_data #need to drop  things that weather forecast APIs dont have (by the minute)
    #print(race.date) #datetime in UTC
    #print(race.session_start_time)
    #print(weather)
    
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


    #print(Laps_with_weather)

    all_laps = pd.concat([all_laps, Laps_with_weather], axis=0)
    circuit = race.get_circuit_info()
    corners = circuit.corners
    print(circuit.corners)


    





   



   
  








    





