import pandas as pd

driver_metadata = pd.DataFrame({
    'Name': ['VER', 'NOR', 'LEC', 'HAM', 'ALB', 
             'PIA', 'SAI', 'HUL', 'GAS', 'ANT',
             'BEA', 'ALO', 'HAD', 'COL', 'OCO',
             'LAW', 'TSU', 'RUS', 'BOR', 'STR'],
    'Age': [27, 25, 27, 40, 29, 
            24, 30, 38, 29, 19,
            20, 44, 20, 22, 28,
            23, 25, 27, 20, 26],
    'Wins': [65, 9, 8, 105, 0, 
             8, 4, 0, 1, 0,
             0, 32, 0, 0, 1,
             0, 0, 4, 0, 0],
    'Podiums': [117, 38, 48, 202, 2, 
                22, 27, 1, 5, 1,
                0, 106, 0, 0, 4,
                0, 0, 21, 0, 3],
    'Championships': [4, 0, 0, 7, 0, 
                      0, 0, 0, 0, 0,
                      0, 2, 0, 0, 0,
                      0, 0, 0, 0, 0],
    'Races': [223, 142, 161, 370, 118, 
              60, 220, 241, 167, 14,
              17, 417, 13, 17, 170,
              25, 101, 142, 14, 180],
    'Points': [3210.5, 1282, 1581, 4971.5, 294,
               673, 1288.5, 608, 456, 64,
               15, 2363, 22, 5, 472,
               26, 101, 886, 14, 318],
    'DNFs': [33, 12, 21, 32, 21,
             3, 40, 42, 26, 4, 
             2, 81, 1, 3, 25,
             5, 15, 19, 3, 29]
})

team_metadata = pd.DataFrame({
    'Team' : ['McLaren', 'Red Bull Racing', 'Mercedes', 'Ferrari', 'Racing Bulls',
 'Williams', 'Alpine', 'Aston Martin', 'Kick Sauber', 'Haas F1 Team'],
    'ConstructorsChampionships': [9, 6, 8, 16, 0,
                                  0, 0, 0, 0, 0,],
    'Wins': [200, 124, 130, 248, 0,
             114, 1, 0, 1, 0],
    'Podiums': [548, 287, 305, 834, 0,
                313, 6, 9, 28, 0],
    'TotalRaces': [984, 407, 331, 1112, 14,
                   865, 104, 109, 500, 204],
    'Points': [7516.5, 8031, 7926.5, 10584, 45,
               3707, 533, 558, 920, 342]
})

driver_metadata.to_csv('driver_metadata.csv', index=False)
team_metadata.to_csv('team_metadata.csv', index=False)
