import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from dataset import ImageTabularDataset
from sklearn.decomposition import PCA
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def add_embeddings_to_dataframe(df, track_embeddings, driver_embeddings, team_embeddings):
    """
    Add track, driver, and team embeddings to the DataFrame.
    
    Args:
        df: DataFrame with race data
        track_embeddings: Dict of track name to embedding
        driver_embeddings: Dict of driver name to embedding
        team_embeddings: Dict of team name to embedding
    """ 
    # Add track embeddings as features
    
    if 'Name' in df.columns:
        track_embedding_features = []
        for idx, row in df.iterrows():
            track_name = row['Name']
            if track_name in track_embeddings:
                embedding = track_embeddings[track_name]
                track_embedding_features.append(embedding)
            else:
                # Use zero embedding for unknown tracks
                track_embedding_features.append(np.zeros(2048))
        
        # Add track embedding features to df
        track_embedding_df = pd.DataFrame(track_embedding_features,
        columns=[f'track_embed_{i}' for i in range(len(track_embedding_features[0]))],
        index=df.index)
        df = pd.concat([df, track_embedding_df], axis=1)
        print(f"Added {len(track_embedding_df.columns)} track embedding features")
    
    # Add driver embeddings as features
    if 'Driver_idx' in df.columns:
        driver_embedding_features = []
        for idx, row in df.iterrows():
            driver_name = row['Driver_idx']
            if driver_name in driver_embeddings:
                embedding = driver_embeddings[driver_name]
                driver_embedding_features.append(embedding)
            else:
                # Use zero embedding for unknown drivers
                driver_embedding_features.append(np.zeros(16))
        
        # Add driver embedding features to df
        driver_embedding_df = pd.DataFrame(driver_embedding_features)
        driver_embedding_df.columns = [f'driver_embed_{i}' for i in range(driver_embedding_df.shape[1])]
        df = pd.concat([df, driver_embedding_df], axis=1)
        print(f"Added {len(driver_embedding_df.columns)} driver embedding features")
    
    # Add team embeddings as features
    if 'Team_idx' in df.columns:
        team_embedding_features = []
        for idx, row in df.iterrows():
            team_name = row['Team_idx']
            if team_name in team_embeddings:
                embedding = team_embeddings[team_name]
                team_embedding_features.append(embedding)
            else:
                # Use zero embedding for unknown teams
                team_embedding_features.append(np.zeros(16))
        
        # Add team embedding features to df
        team_embedding_df = pd.DataFrame(team_embedding_features)
        team_embedding_df.columns = [f'team_embed_{i}' for i in range(team_embedding_df.shape[1])]
        df = pd.concat([df, team_embedding_df], axis=1)
        print(f"Added {len(team_embedding_df.columns)} team embedding features") 
    return df            



def create_track_embeddings_dict(tracks_dir, n_components=50, use_pca=True):
    """
    Create track embeddings with optional PCA reduction.
    
    Args:
        tracks_dir: Directory containing track images
        n_components: Number of PCA components to reduce to (default 50)
        use_pca: Whether to apply PCA reduction (default True)
    """
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # First pass: collect all embeddings
    track_names = []
    all_embeddings = []

    print("Extracting embeddings from images...")
    for img_file in os.listdir(tracks_dir):
        if img_file.endswith('.png'):
            track_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(tracks_dir, img_file)
            
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = model(image_tensor)
                embedding = features.cpu().numpy().flatten()
            
            track_names.append(track_name)
            all_embeddings.append(embedding)
    
    all_embeddings = np.array(all_embeddings)
    print(f"Original embedding shape: {all_embeddings.shape}")
    
    # Apply PCA if requested
    if use_pca and len(all_embeddings) > n_components:
        print(f"Applying PCA to reduce from {all_embeddings.shape[1]} to {n_components} dimensions...")
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(all_embeddings)
        
        # Print variance explained
        variance_explained = pca.explained_variance_ratio_.sum()
        print(f"PCA variance explained: {variance_explained:.2%}")
        print(f"Reduced embedding shape: {reduced_embeddings.shape}")
        
        all_embeddings = reduced_embeddings
    elif use_pca:
        print(f"Warning: Not enough samples ({len(all_embeddings)}) for PCA with {n_components} components")
    
    # Create dictionary
    track_embeddings = {
        name: embedding 
        for name, embedding in zip(track_names, all_embeddings)
    }
    
    return track_embeddings

def create_entity_embeddings(df, column_name, embedding_dim=16):
    """
    Create embeddings for drivers or teams using performance statistics.
    
    Args:
        df: DataFrame with race data
        column_name: 'Driver' or 'Team'
        embedding_dim: Dimension of embeddings to create
    """
    embeddings = {}
    
    for entity in df[column_name].unique():
        entity_data = df[df[column_name] == entity]
        
        # Calculate performance statistics
        stats = [
            entity_data['LapTime_sec'].mean(),
            entity_data['LapTime_sec'].std(),
            entity_data['LapTime_sec'].median(),
            entity_data['LapTime_sec'].quantile(0.25),
            entity_data['LapTime_sec'].quantile(0.75),
            entity_data['Position'].mean(),
            entity_data['Position'].std(),
            entity_data['TyreLife'].mean(),
            len(entity_data),  # Number of laps
        ]
        
        # Pad or truncate to embedding_dim
        if len(stats) < embedding_dim:
            stats.extend([0] * (embedding_dim - len(stats)))
        else:
            stats = stats[:embedding_dim]
        
        embeddings[entity] = np.array(stats)
    
    # Normalize embeddings
    all_embeddings = np.array(list(embeddings.values()))
    mean = all_embeddings.mean(axis=0)
    std = all_embeddings.std(axis=0) + 1e-8  # Avoid division by zero
    
    for entity in embeddings:
        embeddings[entity] = (embeddings[entity] - mean) / std
    
    print(f"Created {embedding_dim}-dim embeddings for {len(embeddings)} {column_name}s")
    return embeddings