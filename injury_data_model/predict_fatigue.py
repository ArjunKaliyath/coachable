"""
Simple fatigue prediction script
Load saved models and predict fatigue for any player

USAGE:
------
from predict_fatigue import predict_player_fatigue

# Get fatigue score for a player
result = predict_player_fatigue('Ousmane Dembélé')
print(result['fatigue_score'])  # e.g., 45.0

# Available players:
# - Ousmane Dembélé
# - Pedri
# - Cole Palmer
# - Martin Ødegaard
# - Désiré Doué

EXAMPLE OUTPUT:
--------------
{
    'player': 'Ousmane Dembélé',
    'fatigue_score': 45.0,
    'n_matches': 3,
    'date_range': '2026-02-08 to 2026-02-17',
    'total_minutes': 173,
    'avg_minutes': 57.7,
    'high_intensity_games': 0,
    'total_fouled': 1,
    'yellow_cards': 0
}
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path


def load_player_data(player_name):
    """Load FBref data for a player"""
    fbref_files = {
        'Ousmane Dembélé': 'injury_data_model/fbref_data/dembele_fbref.csv',
        'Pedri': 'injury_data_model/fbref_data/pedri_fbref.csv',
        'Cole Palmer': 'injury_data_model/fbref_data/palmer_fbref.csv',
        'Martin Ødegaard': 'injury_data_model/fbref_data/odegaard_fbref.csv',
        'Désiré Doué': 'injury_data_model/fbref_data/doue_fbref.csv'
    }
    
    if player_name not in fbref_files:
        raise ValueError(f"Player '{player_name}' not found. Available players: {list(fbref_files.keys())}")
    
    df = pd.read_csv(fbref_files[player_name])
    
    # Fix column alignment (Player header is shifted)
    if 'Player' in df.columns and df.columns[0] == 'Player':
        new_columns = df.columns[1:].tolist()
        df = df.iloc[:, :-1]
        df.columns = new_columns
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Ensure numeric columns
    numeric_cols = ['Min', 'Fls', 'Fld', 'Sh', 'SoT', 'CrdY', 'CrdR', 'Int', 'TklW', 'Gls', 'Ast']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def create_aggregate_features(matches, n_matches):
    """Create aggregate features from N matches - matches the training model exactly"""
    features = {
        'avg_minutes': matches['Min'].mean(),
        'total_minutes': matches['Min'].sum(),
        'avg_fouls_committed': matches['Fls'].mean(),
        'avg_fouled_on': matches['Fld'].mean(),
        'total_fouled_on': matches['Fld'].sum(),
        'avg_shots': matches['Sh'].mean(),
        'total_shots': matches['Sh'].sum(),
        'avg_yellow_cards': matches['CrdY'].mean(),
        'total_yellow_cards': matches['CrdY'].sum(),
        'avg_red_cards': matches['CrdR'].mean(),
        'avg_tackles': matches['TklW'].mean() if 'TklW' in matches.columns else 0,
        'avg_interceptions': matches['Int'].mean(),
        'high_minute_games': (matches['Min'] >= 75).sum(),
        'very_high_minute_games': (matches['Min'] >= 85).sum(),
        'games_with_yellow': (matches['CrdY'] > 0).sum(),
        'max_minutes': matches['Min'].max(),
        'min_minutes': matches['Min'].min(),
        'std_minutes': matches['Min'].std() if len(matches) > 1 else 0,
    }
    return features


def predict_player_fatigue(player_name):
    """
    Predict fatigue score for a player
    
    Args:
        player_name (str): Name of the player
        
    Returns:
        float: Fatigue score between 0-100
    """
    # Load the saved model
    model_files = {
        'Ousmane Dembélé': 'injury_data_model/models/Ousmane_Dembele_fatigue_model.pkl',
        'Pedri': 'injury_data_model/models/Pedri_fatigue_model.pkl',
        'Cole Palmer': 'injury_data_model/models/Cole_Palmer_fatigue_model.pkl',
        'Martin Ødegaard': 'injury_data_model/models/Martin_Ødegaard_fatigue_model.pkl',
        'Désiré Doué': 'injury_data_model/models/Desire_Doue_fatigue_model.pkl'
    }
    
    if player_name not in model_files:
        raise ValueError(f"No model found for '{player_name}'. Available players: {list(model_files.keys())}")
    
    # Load model
    model_path = model_files[player_name]
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    n_matches = model_data['n_matches']
    feature_names = model_data['feature_names']
    
    # Load player data
    df = load_player_data(player_name)
    
    # Get last N matches
    if len(df) < n_matches:
        raise ValueError(f"Not enough match data. Need at least {n_matches} matches.")
    
    recent_matches = df.tail(n_matches)
    
    # Create features
    features = create_aggregate_features(recent_matches, n_matches)
    X = pd.DataFrame([features])
    X = X[feature_names]  # Ensure correct order
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    
    # Get probability of injury (fatigue)
    fatigue_probability = model.predict_proba(X_scaled)[0][1]
    fatigue_score = fatigue_probability * 100
    
    # Get date range
    start_date = recent_matches['Date'].iloc[0].strftime('%Y-%m-%d')
    end_date = recent_matches['Date'].iloc[-1].strftime('%Y-%m-%d')
    
    return {
        'player': player_name,
        'fatigue_score': round(float(fatigue_score), 1),
        'n_matches': n_matches,
        'date_range': f"{start_date} to {end_date}",
        'total_minutes': int(features['total_minutes']),
        'avg_minutes': round(float(features['avg_minutes']), 1),
        'high_intensity_games': int(features['high_minute_games']),
        'total_fouled': int(features['total_fouled_on']),
        'yellow_cards': int(features['total_yellow_cards'])
    }


if __name__ == "__main__":
    # Example usage
    players = [
        'Ousmane Dembélé',
        'Pedri',
        'Cole Palmer',
        'Martin Ødegaard',
        'Désiré Doué'
    ]
    
    print("=" * 70)
    print("PLAYER FATIGUE PREDICTIONS")
    print("=" * 70)
    
    for player in players:
        try:
            result = predict_player_fatigue(player)
            print(f"\n{result['player']}")
            print(f"  Fatigue Score: {result['fatigue_score']}/100")
            print(f"  Based on last {result['n_matches']} matches ({result['date_range']})")
            print(f"  Total minutes: {result['total_minutes']} | Avg: {result['avg_minutes']} min/game")
            print(f"  High-intensity games (75+ min): {result['high_intensity_games']}")
            print(f"  Times fouled: {result['total_fouled']} | Yellow cards: {result['yellow_cards']}")
        except Exception as e:
            print(f"\n{player}: Error - {e}")
    
    print("\n" + "=" * 70)
    
    # Example: Get single player
    print("\n# Example: Get fatigue for a specific player")
    print("result = predict_player_fatigue('Ousmane Dembélé')")
    print(f"result = {predict_player_fatigue('Ousmane Dembélé')}")
