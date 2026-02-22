import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load injury data
injury_df = pd.read_csv('soccer_injury_dataset.csv')

# Player files mapping
player_files = {
    'Ousmane Dembélé': 'fbref_data/dembele_fbref.csv',
    'Pedri': 'fbref_data/pedri_fbref.csv',
    'Cole Palmer': 'fbref_data/palmer_fbref.csv',
    'Martin Ødegaard': 'fbref_data/odegaard_fbref.csv',
    'Désiré Doué': 'fbref_data/doue_fbref.csv'
}

def load_and_prepare_data(player_name, filepath):
    """Load player data and convert dates"""
    # The CSV has misaligned columns - "Player" header but data starts with Date
    # Read raw and fix column alignment
    df = pd.read_csv(filepath)
    
    # The headers are shifted right by 1 compared to data
    # Data columns: Date, Day, Comp, Round, Venue, Result, Squad, Opponent, Start, Pos, Min, ...
    # Header columns: Player, Date, Day, Comp, Round, Venue, Result, Squad, Opponent, Start, Pos, Min, ...
    # Solution: Drop last column and shift header names left by removing 'Player'
    
    if 'Player' in df.columns and df.columns[0] == 'Player':
        # Get all headers except 'Player'
        new_columns = df.columns[1:].tolist()
        # Drop the last data column (which corresponds to the extra header shift)
        df = df.iloc[:, :-1]
        # Assign the corrected column names
        df.columns = new_columns
    
    # Now Date column should have actual dates
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Select relevant features
    feature_cols = ['Min', 'Fls', 'Fld', 'Sh', 'SoT', 'CrdY', 'CrdR', 'Int', 'TklW', 'Gls', 'Ast']
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def get_injury_dates(player_name):
    """Get all injury start dates for a player"""
    player_injuries = injury_df[injury_df['player'] == player_name]
    injury_dates = []
    for _, row in player_injuries.iterrows():
        injury_date = pd.to_datetime(row['injury_start_date'])
        injury_dates.append(injury_date)
    return injury_dates

def create_aggregate_features(matches_df, n_matches):
    """Create aggregated features from N matches"""
    if len(matches_df) < n_matches:
        return None
    
    recent_matches = matches_df.tail(n_matches)
    
    features = {
        'avg_minutes': recent_matches['Min'].mean(),
        'total_minutes': recent_matches['Min'].sum(),
        'avg_fouls_committed': recent_matches['Fls'].mean(),
        'avg_fouled_on': recent_matches['Fld'].mean(),
        'total_fouled_on': recent_matches['Fld'].sum(),
        'avg_shots': recent_matches['Sh'].mean(),
        'total_shots': recent_matches['Sh'].sum(),
        'avg_yellow_cards': recent_matches['CrdY'].mean(),
        'total_yellow_cards': recent_matches['CrdY'].sum(),
        'avg_red_cards': recent_matches['CrdR'].mean(),
        'avg_tackles': recent_matches['TklW'].mean(),
        'avg_interceptions': recent_matches['Int'].mean(),
        'high_minute_games': (recent_matches['Min'] >= 75).sum(),
        'very_high_minute_games': (recent_matches['Min'] >= 85).sum(),
        'games_with_yellow': (recent_matches['CrdY'] > 0).sum(),
        'max_minutes': recent_matches['Min'].max(),
        'min_minutes': recent_matches['Min'].min(),
        'std_minutes': recent_matches['Min'].std() if len(recent_matches) > 1 else 0,
    }
    
    return features

def create_dataset_for_player(player_name, df, n_matches=5, lookback_days=7):
    """Create training dataset with positive (pre-injury) and negative samples"""
    injury_dates = get_injury_dates(player_name)
    
    X_data = []
    y_data = []
    
    # Positive samples: N matches before each injury
    for injury_date in injury_dates:
        # Get ALL matches before injury, then take the N most recent ones
        pre_injury_matches = df[df['Date'] < injury_date].sort_values('Date', ascending=False)
        
        if len(pre_injury_matches) >= n_matches:
            # Take the N most recent matches before injury
            recent_n = pre_injury_matches.head(n_matches).sort_values('Date')
            features = create_aggregate_features(recent_n, n_matches)
            if features:
                X_data.append(features)
                y_data.append(1)  # Fatigued/at risk
    
    # Negative samples: Random periods without injuries
    # We'll sample periods that are at least 30 days away from any injury
    safe_periods = []
    for i in range(len(df) - n_matches):
        window = df.iloc[i:i+n_matches]
        window_end = window['Date'].iloc[-1]
        
        # Check if this window is far from any injury
        is_safe = True
        for injury_date in injury_dates:
            days_to_injury = (injury_date - window_end).days
            if 0 < days_to_injury < 30:  # Within 30 days before injury
                is_safe = False
                break
            if -7 < days_to_injury < 0:  # During/after injury
                is_safe = False
                break
        
        if is_safe:
            safe_periods.append(i)
    
    # Sample negative examples (same number as positive or more)
    num_negative_samples = max(len(injury_dates) * 2, len(injury_dates))
    if len(safe_periods) > num_negative_samples:
        sampled_indices = np.random.choice(safe_periods, num_negative_samples, replace=False)
    else:
        sampled_indices = safe_periods
    
    for idx in sampled_indices:
        window = df.iloc[idx:idx+n_matches]
        features = create_aggregate_features(window, n_matches)
        if features:
            X_data.append(features)
            y_data.append(0)  # Not fatigued
    
    if len(X_data) == 0:
        return None, None
    
    X = pd.DataFrame(X_data)
    y = np.array(y_data)
    
    return X, y

def train_model_for_player(player_name, df, n_matches_list=[3, 5, 7]):
    """Train model with grid search for a player"""
    print(f"\n{'='*60}")
    print(f"Training model for: {player_name}")
    print(f"{'='*60}")
    
    best_score = -1
    best_model = None
    best_n = None
    best_X_train = None
    best_y_train = None
    best_scaler = None
    best_model_name = None
    
    # Try different values of N
    for n_matches in n_matches_list:
        print(f"\nTrying N={n_matches} matches...")
        
        X, y = create_dataset_for_player(player_name, df, n_matches=n_matches)
        
        if X is None or len(X) < 5:
            print(f"  Not enough data for N={n_matches}")
            continue
        
        # Check if we have both classes
        if y.sum() ==0 or y.sum() == len(y):
            print(f"  Dataset has only one class, skipping...")
            continue
            
        print(f"  Dataset size: {len(X)} samples ({y.sum()} injured, {(1-y).sum()} healthy)")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use XGBoost (best performing model)
        model = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        model_name = 'XGBoost'
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        print(f"\n  Training XGBoost...")
        
        # For very small datasets, use leave-one-out or simpler CV
        if len(X) < 10:
            from sklearn.model_selection import LeaveOneOut, cross_val_score
            from sklearn.metrics import make_scorer, roc_auc_score
            from itertools import product
            
            # Check if we have at least 2 of each class
            min_class_count = min(y.sum(), len(y) - y.sum())
            
            if min_class_count < 2:
                print(f"    Warning: Only {min_class_count} sample(s) of minority class, cannot validate properly")
                # Train on full dataset with default params
                model.fit(X_scaled, y)
                # Use training score (overfitted but best we can do)
                y_pred_proba = model.predict_proba(X_scaled)[:, 1]
                score = roc_auc_score(y, y_pred_proba)
                print(f"    Training score: {score:.3f}")
                print(f"    Params: default")
                grid_best_estimator = model
            else:
                # Use Leave-One-Out CV for very small datasets
                print(f"    Using Leave-One-Out CV ({len(X)} folds)")
                
                best_model_score = -1
                best_params = None
                best_model_instance = None
                
                # Generate all parameter combinations
                param_names = list(param_grid.keys())
                param_values = [param_grid[name] for name in param_names]
                
                # Try each parameter combination
                for param_combo in product(*param_values):
                    params = dict(zip(param_names, param_combo))
                    
                    # Create XGBoost model with these params
                    m = XGBClassifier(**params, random_state=42, eval_metric='logloss', use_label_encoder=False)
                    
                    # Use LOO CV
                    try:
                        loo = LeaveOneOut()
                        scorer = make_scorer(roc_auc_score, needs_proba=True)
                        scores = cross_val_score(m, X_scaled, y, cv=loo, scoring=scorer)
                        score = scores.mean()
                        
                        if score > best_model_score:
                            best_model_score = score
                            best_params = params
                            # Refit on full data with best params
                            m.fit(X_scaled, y)
                            best_model_instance = m
                    except Exception as e:
                        # Skip this parameter combination if it fails
                        continue
                
                if best_model_instance is None:
                    print(f"    All parameter combinations failed, using default")
                    model.fit(X_scaled, y)
                    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
                    best_model_score = roc_auc_score(y, y_pred_proba)
                    best_params = {}
                    best_model_instance = model
                
                print(f"    Best LOO CV score: {best_model_score:.3f}")
                print(f"    Best params: {best_params}")
                score = best_model_score
                grid_best_estimator = best_model_instance
        
        else:
            # Use GridSearchCV with stratified folds
            from sklearn.model_selection import StratifiedKFold
            cv_folds = min(3, y.sum(), (len(y) - y.sum()))  # At most 3 folds, limited by minority class
            
            grid_search = GridSearchCV(
                model, 
                param_grid,
                cv=StratifiedKFold(n_splits=max(2, cv_folds), shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            try:
                grid_search.fit(X_scaled, y)
                score = grid_search.best_score_
                grid_best_estimator = grid_search.best_estimator_
                
                print(f"    Best CV AUC: {score:.3f}")
                print(f"    Best params: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        # Only update best score if current score is valid (not NaN)
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_model = grid_best_estimator
            best_model_name = 'XGBoost'
            best_n = n_matches
            best_X_train = X
            best_y_train = y
            best_scaler = scaler
    
    if best_model is None:
        print(f"\n⚠️  Could not train model for {player_name}")
        return None, None, None, None
    
    print(f"\n✓ Best configuration:")
    print(f"  Model: XGBoost")
    print(f"  N matches: {best_n}")
    print(f"  Score: {best_score:.3f}")
    
    return best_model, best_scaler, best_n, best_X_train.columns.tolist()

def predict_current_fatigue(player_name, df, model, scaler, n_matches, feature_names):
    """Predict fatigue for the last 5 matches"""
    print(f"\n{'='*60}")
    print(f"Predicting current fatigue for: {player_name}")
    print(f"{'='*60}")
    
    # Get last n_matches for prediction
    if len(df) < n_matches:
        print(f"Not enough recent matches for prediction")
        return None
    
    recent_matches = df.tail(n_matches)
    print(f"Using matches from {recent_matches['Date'].iloc[0].date()} to {recent_matches['Date'].iloc[-1].date()}")
    
    # Create features
    features = create_aggregate_features(recent_matches, n_matches)
    if features is None:
        return None
    
    X_pred = pd.DataFrame([features])
    X_pred = X_pred[feature_names]  # Ensure same column order
    X_pred_scaled = scaler.transform(X_pred)
    
    # Get probability of being fatigued
    fatigue_prob = model.predict_proba(X_pred_scaled)[0][1]
    fatigue_score = fatigue_prob * 100  # Convert to 0-100 scale
    
    print(f"\n📊 Fatigue Score: {fatigue_score:.1f}/100")
    
    # Show key stats
    print(f"\nKey Statistics (last {n_matches} matches):")
    print(f"  Total minutes: {features['total_minutes']:.0f}")
    print(f"  Avg minutes per game: {features['avg_minutes']:.1f}")
    print(f"  Times fouled: {features['total_fouled_on']:.0f}")
    print(f"  High-intensity games (75+ min): {features['high_minute_games']}")
    print(f"  Yellow cards: {features['total_yellow_cards']:.0f}")
    
    return fatigue_score

def load_saved_model(player_name):
    """
    Load a previously saved model for a player.
    
    Args:
        player_name: Name of the player
        
    Returns:
        Dictionary with model, scaler, n_matches, feature_names, and data
    """
    safe_name = player_name.replace(' ', '_').replace('é', 'e').replace('ø', 'o')
    model_filename = f'models/{safe_name}_fatigue_model.pkl'
    
    try:
        player_info = joblib.load(model_filename)
        print(f"✓ Loaded model for {player_name} from {model_filename}")
        return player_info
    except Exception as e:
        print(f"✗ Error loading model for {player_name}: {e}")
        return None

# Main execution
if __name__ == "__main__":
    print("="*80)
    print(" FATIGUE PREDICTION MODEL - TRAINING & EVALUATION")
    print("="*80)
    
    # Store models and results
    player_models = {}
    fatigue_predictions = {}
    
    # Train models for each player
    for player_name, filepath in player_files.items():
        try:
            df = load_and_prepare_data(player_name, filepath)
            model, scaler, n_matches, feature_names = train_model_for_player(player_name, df)
            
            if model is not None:
                player_models[player_name] = {
                    'model': model,
                    'scaler': scaler,
                    'n_matches': n_matches,
                    'feature_names': feature_names,
                    'data': df
                }
        except Exception as e:
            print(f"\n❌ Error processing {player_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save trained models
    print("\n" + "="*80)
    print(" SAVING MODELS")
    print("="*80)
    
    for player_name, player_info in player_models.items():
        # Create a safe filename
        safe_name = player_name.replace(' ', '_').replace('é', 'e').replace('ø', 'o')
        model_filename = f'models/{safe_name}_fatigue_model.pkl'
        
        # Save the entire player_info dictionary
        try:
            import os
            os.makedirs('models', exist_ok=True)
            joblib.dump(player_info, model_filename)
            print(f"✓ Saved model for {player_name} to {model_filename}")
        except Exception as e:
            print(f"✗ Error saving model for {player_name}: {e}")
    
    print("\n" + "="*80)
    print(" CURRENT FATIGUE PREDICTIONS")
    print("="*80)
    
    # Make predictions for all players
    for player_name, player_info in player_models.items():
        try:
            fatigue_score = predict_current_fatigue(
                player_name,
                player_info['data'],
                player_info['model'],
                player_info['scaler'],
                player_info['n_matches'],
                player_info['feature_names']
            )
            fatigue_predictions[player_name] = fatigue_score
        except Exception as e:
            print(f"\n❌ Error predicting for {player_name}: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print(" FINAL FATIGUE SCORES SUMMARY")
    print("="*80)
    
    # Sort by fatigue score
    sorted_players = sorted(fatigue_predictions.items(), key=lambda x: x[1] if x[1] else 0, reverse=True)
    
    print(f"\n{'Player':<25} {'Fatigue Score':<15} {'Status'}")
    print("-"*60)
    for player_name, score in sorted_players:
        if score is not None:
            if score >= 70:
                status = "🔴 HIGH RISK"
            elif score >= 50:
                status = "🟡 MODERATE"
            else:
                status = "🟢 LOW RISK"
            print(f"{player_name:<25} {score:>6.1f}/100      {status}")
    
    # Check if currently injured players have higher scores
    print("\n" + "="*80)
    print(" VALIDATION: Currently Injured Players")
    print("="*80)
    current_date = datetime(2026, 2, 21)
    currently_injured = []
    
    for _, row in injury_df.iterrows():
        injury_start = pd.to_datetime(row['injury_start_date'])
        injury_end = pd.to_datetime(row['injury_end_date']) if pd.notna(row['injury_end_date']) else current_date + timedelta(days=1)
        
        if injury_start <= current_date <= injury_end:
            currently_injured.append(row['player'])
    
    print(f"\nCurrently injured players: {', '.join(set(currently_injured))}")
    print("\nTheir fatigue scores:")
    for player in set(currently_injured):
        if player in fatigue_predictions:
            score = fatigue_predictions[player]
            print(f"  {player}: {score:.1f}/100")
