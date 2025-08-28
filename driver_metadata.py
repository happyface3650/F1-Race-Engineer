import pandas as pd
import numpy as np
from datetime import datetime

driver_metadata = pd.DataFrame({
    'Name': ['VER', 'NOR', 'LEC', 'HAM'],
    'Age': [27, 25, 27, 40],
    'Wins': [65, 9, 8, 105],
    'Championships': [4, 0, 0, 7],
    'Races': [223, 142, 161, 370]
})

team_metadata = pd.DataFrame({
    'Team' : ['McLaren', 'Red Bull Racing', 'Mercedes', 'Ferrari', 'Racing Bulls',
 'Williams', 'Alpine', 'Aston Martin', 'Kick Sauber', 'Haas F1 Team'],
    'ConstructorsChampionships': [],
    'Wins': [],
    'Podiums': [],
    'DNFs': [],
    'TotalRaces': [],
    'F1Experience': [],
})

# driver meta data, team meta data, 