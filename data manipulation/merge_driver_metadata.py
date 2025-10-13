import pandas as pd
import numpy as np


'''
in the case that we want to just merge driver metadata with the dataset
'''

def merge_driver_metadata():
    # Load the datasets
    lap_data = pd.read_csv('data\\All_Laps.csv')
    driver_metadata = pd.read_csv('data\\driver_metadata.csv')
    
    print(f"Original lap data shape: {lap_data.shape}")
    print(f"Driver metadata shape: {driver_metadata.shape}")
    
    # Check unique drivers in both datasets
    print(f"\nDrivers in lap data: {sorted(lap_data['Driver'].unique())}")
    print(f"Drivers in metadata: {sorted(driver_metadata['Name'].unique())}")
    
    # Merge driver metadata with lap data
    merged_data = lap_data.merge(
        driver_metadata, 
        left_on='Driver', 
        right_on='Name', 
        how='left'
    )
    
    print(f"\nMerged data shape: {merged_data.shape}")
    
    # Check for drivers without metadata
    missing_metadata = merged_data[merged_data['Age'].isna()]['Driver'].unique()
    if len(missing_metadata) > 0:
        print(f"Drivers missing metadata: {missing_metadata}")
    
    # Drop the redundant 'Name' column from metadata
    merged_data = merged_data.drop('Name', axis=1)
    
    # Create derived features from driver metadata
    merged_data['WinRate'] = merged_data['Wins'] / merged_data['Races']
    merged_data['PodiumRate'] = merged_data['Podiums'] / merged_data['Races']
    merged_data['DNFRate'] = merged_data['DNFs'] / merged_data['Races']
    merged_data['PointsPerRace'] = merged_data['Points'] / merged_data['Races']
    
    # Experience categories
    merged_data['ExperienceLevel'] = pd.cut(
        merged_data['Races'], 
        bins=[0, 50, 150, 300, 500], 
        labels=['Rookie', 'Experienced', 'Veteran', 'Legend']
    )
    
    # Age categories
    merged_data['AgeGroup'] = pd.cut(
        merged_data['Age'], 
        bins=[0, 25, 30, 35, 50], 
        labels=['Young', 'Prime', 'Experienced', 'Veteran']
    )
    
    print(f"\nFinal merged data shape: {merged_data.shape}")
    print(f"New columns added: {['Age', 'Wins', 'Podiums', 'Championships', 'Races', 'Points', 'DNFs', 'WinRate', 'PodiumRate', 'DNFRate', 'PointsPerRace', 'ExperienceLevel', 'AgeGroup']}")
    
    # Save the merged dataset
    merged_data.to_csv('data\\All_Laps_with_Driver_Metadata.csv', index=False)
    print("\nMerged dataset saved as 'All_Laps_with_Driver_Metadata.csv'")
    
    return merged_data

if __name__ == "__main__":
    merged_data = merge_driver_metadata()
    
    # Display some statistics
    print("\n=== Driver Performance Statistics ===")
    driver_stats = merged_data.groupby('Driver').agg({
        'WinRate': 'first',
        'PodiumRate': 'first', 
        'PointsPerRace': 'first',
        'Age': 'first',
        'Races': 'first'
    }).round(3)
    
    print(driver_stats.sort_values('PointsPerRace', ascending=False))
