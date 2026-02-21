"""
Update Dataset with Real WhoScored Data

This script:
1. Loads the existing soccer_injury_dataset.csv
2. For each player with placeholders, opens WhoScored
3. Pauses for you to solve bot challenges
4. Extracts real match data
5. Updates and saves the CSV

You'll solve bot challenges once per player, then it extracts all their injury data.
"""

from playwright.sync_api import sync_playwright
from datetime import datetime, timedelta
import pandas as pd
import time
import re


# Player WhoScored IDs
PLAYER_IDS = {
    "Mason Mount": "321810",
    "Jack Grealish": "297949",
    "Matthijs de Ligt": "361155",
    "Ansu Fati": "398027",
    "Pedri": "443424",
    "Reece James": "347804",
    "Ousmane Dembélé": "296316"
}


def extract_matches_from_whoscored(page, injury_dates, num_matches_per_injury=5):
    """
    Extract match history from WhoScored for multiple injury dates.
    Returns a dict mapping injury_date -> list of matches
    """
    print(f"\n  Analyzing page content...")
   
    all_matches = []
    
    try:
        # Give page time to load
        time.sleep(3)
        
        # Look for all tables
        tables = page.query_selector_all('table')
        print(f"  Found {len(tables)} tables")
        
        # Extract all matches from all tables
        for table in tables:
            rows = table.query_selector_all('tr')
            
            for row in rows:
                cells = row.query_selector_all('td')
                if len(cells) < 3:
                    continue
                
                try:
                    # Try to find date in first few cells
                    match_date = None
                    date_text = None
                    
                    for cell in cells[:4]:
                        text = cell.inner_text().strip()
                        
                        # Check if this looks like a date
                        date_patterns = [
                            (r'\d{1,2}/\d{1,2}/\d{4}', '%d/%m/%Y'),
                            (r'\d{1,2}/\d{1,2}/\d{2}', '%d/%m/%y'),
                            (r'[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}', '%b %d, %Y'),
                        ]
                        
                        for pattern, fmt in date_patterns:
                            if re.search(pattern, text):
                                try:
                                    match_date = datetime.strptime(text, fmt)
                                    date_text = text
                                    break
                                except:
                                    pass
                        
                        if match_date:
                            break
                    
                    if not match_date:
                        continue
                    
                    # Extract other match details
                    match_info = {
                        'date': match_date,
                        'teams': [],
                        'minutes': None,
                        'rating': None
                    }
                    
                    for cell in cells:
                        text = cell.inner_text().strip()
                        
                        # Team names (short text)
                        if 3 <= len(text) <= 25 and text.replace(' ', '').isalpha():
                            match_info['teams'].append(text)
                        
                        # Minutes played
                        try:
                            mins = int(text)
                            if 0 <= mins <= 120:
                                match_info['minutes'] = mins
                        except:
                            pass
                        
                        # Rating (decimal number)
                        try:
                            rating = float(text)
                            if 0 <= rating <= 10:
                                match_info['rating'] = rating
                        except:
                            pass
                    
                    all_matches.append(match_info)
                    
                except Exception as e:
                    continue
        
        print(f"  Extracted {len(all_matches)} total matches from page")
        
    except Exception as e:
        print(f"  Error extracting: {e}")
    
    # Map matches to injuries
    injury_matches = {}
    
    for inj_date in injury_dates:
        # Find matches before this injury
        pre_injury = [m for m in all_matches if m['date'] < inj_date]
        # Sort by date descending
        pre_injury.sort(key=lambda x: x['date'], reverse=True)
        # Take first num_matches_per_injury
        selected = pre_injury[:num_matches_per_injury]
        
        # Convert to standard format
        matches = []
        for m in selected:
            matches.append({
                'match_date': m['date'],
                'home_team': m['teams'][0] if len(m['teams']) > 0 else 'Unknown',
                'away_team': m['teams'][1] if len(m['teams']) > 1 else 'Unknown',
                'minutes_played': m['minutes'],
                'distance_covered': None,
                'fouls_won': None,
                'rough_fouls_won': None,
                'take_ons': None,
                'duels': None
            })
        
        injury_matches[inj_date] = matches
    
    return injury_matches


def update_player_data(player_name, df):
    """
    Update all data for one player by scraping WhoScored once.
    """
    if player_name not in PLAYER_IDS:
        print(f"  ⚠️  No WhoScored ID for {player_name}, skipping...")
        return df
    
    player_id = PLAYER_IDS[player_name]
    
    print(f"\n{'='*80}")
    print(f"📊 Player: {player_name}")
    print(f"   WhoScored ID: {player_id}")
    print(f"{'='*80}")
    
    # Get all injuries for this player
    player_df = df[df['player'] == player_name]
    injuries = player_df.groupby('injury_number').first()[['injury_start_date']].reset_index()
    
    print(f"\n   Found {len(injuries)} injuries to update")
    for _, inj in injuries.iterrows():
        print(f"     Injury {int(inj['injury_number'])}: {inj['injury_start_date']}")
    
    # Convert injury dates to datetime
    injury_dates = []
    for _, inj in injuries.iterrows():
        try:
            inj_date = pd.to_datetime(inj['injury_start_date'])
            injury_dates.append(inj_date)
        except:
            pass
    
    url = f"https://www.whoscored.com/Players/{player_id}/Fixtures"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=300)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = context.new_page()
        
        try:
            print(f"\n🌐 Opening: {url}")
            page.goto(url, timeout=60000)
            
            print("\n" + "="*80)
            print("🤖 SOLVE BOT CHALLENGE IF NEEDED")
            print("="*80)
            print("\nThen press Enter when the match history is loaded...")
            input(">>> ")
            
            # Extract all matches for all injuries
            injury_matches = extract_matches_from_whoscored(page, injury_dates, num_matches_per_injury=5)
            
            # Update the dataframe
            for inj_date, matches in injury_matches.items():
                print(f"\n  ✓ Updating injury on {inj_date.date()}: {len(matches)} matches")
                
                # Find the injury_number for this date
                inj_row = df[(df['player'] == player_name) & 
                            (pd.to_datetime(df['injury_start_date']) == inj_date)]
                
                if not inj_row.empty:
                    injury_num = inj_row.iloc[0]['injury_number']
                    
                    # Update each match
                    for match_idx, match in enumerate(matches, 1):
                        mask = ((df['player'] == player_name) & 
                               (df['injury_number'] == injury_num) &
                               (df['match_number_before_injury'] == match_idx))
                        
                        if mask.any():
                            df.loc[mask, 'match_date'] = match['match_date']
                            df.loc[mask, 'home_team'] = match['home_team']
                            df.loc[mask, 'away_team'] = match['away_team']
                            df.loc[mask, 'minutes_played'] = match['minutes_played']
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("\n   Closing browser...")
            time.sleep(2)
            browser.close()
    
    return df


def main():
    print("="*80)
    print("🔄 UPDATE DATASET WITH WHOSCORED DATA")
    print("="*80)
    
    # Load existing dataset
    csv_file = 'soccer_injury_dataset.csv'
    print(f"\n📁 Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"   Loaded {len(df)} rows")
    
    # Get unique players
    players = df['player'].unique()
    print(f"\n   Players: {', '.join(players)}")
    
    # Check which players need updates
    tbd_counts = df.groupby('player')['home_team'].apply(lambda x: (x == 'TBD').sum())
    print(f"\n   Matches with 'TBD' teams:")
    for player, count in tbd_counts.items():
        print(f"     {player}: {count} matches")
    
    print("\n" + "="*80)
    print("We'll scrape each player one at a time.")
    print("You'll solve bot challenges, then we extract all their data.")
    print("="*80)
    
    # Process each player
    for player_name in players:
        tbd_count = tbd_counts.get(player_name, 0)
        
        if tbd_count == 0:
            print(f"\n✓ {player_name}: Already complete, skipping")
            continue
        
        print(f"\n\n{'#'*80}")
        input(f"Press Enter to scrape {player_name} ({tbd_count} matches to update)...\n>>> ")
        
        df = update_player_data(player_name, df)
        
        # Save progress after each player
        backup_file = f'soccer_injury_dataset_backup.csv'
        df.to_csv(backup_file, index=False)
        print(f"\n   💾 Progress saved to {backup_file}")
    
    # Final save
    df.to_csv(csv_file, index=False)
    print(f"\n{'='*80}")
    print(f"✅ COMPLETE! Updated dataset saved to {csv_file}")
    print(f"{'='*80}")
    
    # Show summary
    updated_tbd = df.groupby('player')['home_team'].apply(lambda x: (x == 'TBD').sum())
    print(f"\nRemaining 'TBD' matches:")
    for player, count in updated_tbd.items():
        status = "✓" if count == 0 else "⚠️"
        print(f"  {status} {player}: {count}")


if __name__ == "__main__":
    main()
