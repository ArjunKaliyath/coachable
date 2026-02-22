"""
Soccer Player Injury and Match Data Scraper - Main Driver

This script orchestrates the data collection process by coordinating
the Transfermarkt injury scraper and WhoScored match statistics scraper.
"""

import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Dict

# Import scraper modules
from transfermarkt_scraper import PLAYERS, get_all_player_injuries

# Configuration
NUM_MATCHES_BEFORE_INJURY = 5


def get_whoscored_matches(player_name: str, injury_date: datetime) -> List[Dict]:
    """
    Fetch match statistics from WhoScored for matches before an injury.
    
    Args:
        player_name: Name of the player
        injury_date: Date of the injury
        
    Returns:
        List of match dictionaries with statistics
    """
    print(f"  Fetching WhoScored data for {player_name} before {injury_date.date()}...")
    
    matches = []
    
    try:
        # Initialize WhoScored reader with timeout
        # Note: WhoScored scraping can be complex and may require browser automation
        # For now, we'll create a simpler fallback
        
        # Create placeholder matches based on weekly intervals
        # In production, you would integrate with WhoScored API or scrape their pages
        for i in range(NUM_MATCHES_BEFORE_INJURY):
            # Estimate match dates (approximately weekly)
            match_date = injury_date - timedelta(days=(i+1)*7)
            matches.append({
                'match_date': match_date,
                'home_team': 'TBD',
                'away_team': 'TBD',
                'minutes_played': None,
                'distance_covered': None,
                'fouls_won': None,
                'rough_fouls_won': None,
                'take_ons': None,
                'duels': None
            })
        
        print(f"    Created {len(matches)} placeholder match records")
        return matches
        
    except Exception as e:
        print(f"    Error fetching WhoScored data: {e}")
        # Return placeholder matches
        matches = []
        for i in range(NUM_MATCHES_BEFORE_INJURY):
            match_date = injury_date - timedelta(days=(i+1)*7)
            matches.append({
                'match_date': match_date,
                'home_team': 'TBD',
                'away_team': 'TBD',
                'minutes_played': None,
                'distance_covered': None,
                'fouls_won': None,
                'rough_fouls_won': None,
                'take_ons': None,
                'duels': None
            })
        return matches


def build_dataset() -> pd.DataFrame:
    """
    Build the injury dataset with one row per injury.
    
    Returns:
        DataFrame with injury data
    """
    all_data = []
    
    # Get injuries for all players
    all_injuries = get_all_player_injuries()
    
    for player_name, injuries in all_injuries.items():
        # For each injury, create one row
        for injury_idx, injury in enumerate(injuries, 1):
            print(f"  Processing injury {injury_idx}/{len(injuries)}: {injury['injury_type']}")
            
            row = {
                'player': player_name,
                'injury_number': injury_idx,
                'injury_type': injury['injury_type'],
                'injury_start_date': injury['injury_start'],
                'injury_end_date': injury['injury_end'],
                'days_missed': injury['days_missed']
            }
            all_data.append(row)
        
        # Be respectful to servers
        time.sleep(0.5)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    return df


def main():
    """Main execution function."""
    print("=" * 80)
    print("Soccer Player Injury and Match Data Scraper")
    print("=" * 80)
    print()
    
    # Build dataset
    df = build_dataset()
    
    # Display summary
    print()
    print("=" * 80)
    print("Dataset Summary")
    print("=" * 80)
    print(f"Total injuries: {len(df)}")
    print(f"Players: {df['player'].nunique()}")
    print()
    print(df)
    print()
    
    # Save to CSV
    output_file = 'soccer_injury_dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
