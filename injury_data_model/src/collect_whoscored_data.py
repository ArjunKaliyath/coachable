"""
Complete Data Collection Pipeline
Integrates Transfermarkt injury data with WhoScored match statistics
"""

from whoscored_scraper import get_player_matches_whoscored
from playwright.sync_api import sync_playwright
import pandas as pd
from datetime import datetime
import time


def update_dataset_with_whoscored_data():
    """
    Interactive pipeline to update the dataset with WhoScored match statistics
    """
    print("="*80)
    print("📊 SOCCER INJURY DATASET - WHOSCORED INTEGRATION")
    print("="*80)
    print("""
This script will:
1. Load your existing injury dataset (soccer_injury_dataset.csv)
2. Open WhoScored in a browser (you can solve bot challenges)
3. For each injury, scrape the 5 matches before it
4. Extract match statistics (minutes, distance, fouls, take-ons, duels)
5. Save the enhanced dataset

IMPORTANT:
- This will open a browser window (headless=False)
- You may need to solve CAPTCHAs or bot challenges
- The process is interactive - follow the prompts
- You can stop at any time (Ctrl+C)
""")
    print("="*80)
    
    # Load existing dataset
    print("\n📂 Loading existing dataset...")
    df = pd.read_csv('soccer_injury_dataset.csv')
    print(f"   Loaded {len(df)} records")
    print(f"   Columns: {list(df.columns)}")
    
    # Get unique injuries to process
    injuries = df.groupby(['player', 'injury_number']).first().reset_index()
    print(f"\n   Total injuries to process: {len(injuries)}")
    print(f"   Players: {', '.join(sorted(df['player'].unique()))}")
    
    # Ask user what to do
    print("\n" + "="*80)
    print("OPTIONS:")
    print("="*80)
    print("1. Process first player only (test mode)")
    print("2. Process specific player")
    print("3. Process all players (will take time!)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '4':
        print("👋 Exiting...")
        return
    
    # Determine which injuries to process
    injuries_to_process = []
    
    if choice == '1':
        # Test mode - first player only
        first_player = injuries.iloc[0]['player']
        injuries_to_process = injuries[injuries['player'] == first_player].to_dict('records')
        print(f"\n✅ Test mode: Processing {first_player} ({len(injuries_to_process)} injuries)")
        
    elif choice == '2':
        # Specific player
        print("\nAvailable players:")
        for idx, player in enumerate(sorted(df['player'].unique()), 1):
            num_inj = len(injuries[injuries['player'] == player])
            print(f"  {idx}. {player} ({num_inj} injuries)")
        
        player_idx = int(input("\nEnter player number: ").strip()) - 1
        selected_player = sorted(df['player'].unique())[player_idx]
        injuries_to_process = injuries[injuries['player'] == selected_player].to_dict('records')
        print(f"\n✅ Selected: {selected_player} ({len(injuries_to_process)} injuries)")
        
    elif choice == '3':
        # All players
        injuries_to_process = injuries.to_dict('records')
        print(f"\n✅ Processing ALL players: {len(injuries_to_process)} total injuries")
        print("   ⚠️  This may take a while!")
    
    if not injuries_to_process:
        print("❌ No injuries to process")
        return
    
    input("\n Press Enter to launch browser and start...")
    
    # Launch browser
    with sync_playwright() as p:
        print("\n🚀 Launching browser...")
        browser = p.chromium.launch(
            headless=False,
            slow_mo=500
        )
        
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        
        page = context.new_page()
        print("✅ Browser ready!\n")
        
        # Process each injury
        all_match_data = []
        
        for idx, injury in enumerate(injuries_to_process, 1):
            print("\n" + "="*80)
            print(f"INJURY {idx}/{len(injuries_to_process)}")
            print("="*80)
            print(f"Player: {injury['player']}")
            print(f"Injury #{injury['injury_number']}: {injury['injury_type']}")
            print(f"Date: {injury['injury_start_date']}")
            print(f"Days missed: {injury['days_missed']}")
            
            # Get matches for this injury
            injury_date = pd.to_datetime(injury['injury_start_date'])
            matches = get_player_matches_whoscored(
                page, 
                injury['player'], 
                injury_date, 
                num_matches=5
            )
            
            # Add injury context to each match
            for match_num, match in enumerate(matches, 1):
                match_data = {
                    'player': injury['player'],
                    'injury_number': injury['injury_number'],
                    'injury_type': injury['injury_type'],
                    'injury_start_date': injury['injury_start_date'],
                    'injury_end_date': injury.get('injury_end_date'),
                    'days_missed': injury['days_missed'],
                    'match_number_before_injury': match_num,
                    **match  # Add all match data
                }
                all_match_data.append(match_data)
            
            print(f"\n✅ Processed {len(matches)} matches for this injury")
            
            # Small delay between injuries
            if idx < len(injuries_to_process):
                print("\n⏳ Waiting 3 seconds before next injury...")
                time.sleep(3)
        
        print("\n" + "="*80)
        print("🎉 SCRAPING COMPLETE!")
        print("="*80)
        
        input("Press Enter to close browser...")
        browser.close()
    
    # Create new DataFrame
    print("\n📊 Creating updated dataset...")
    new_df = pd.DataFrame(all_match_data)
    
    # Display summary
    print(f"\n   Total rows: {len(new_df)}")
    print(f"   Columns: {list(new_df.columns)}")
    print(f"\n   Sample data:")
    print(new_df[['player', 'injury_number', 'match_date', 'home_team', 'away_team', 'minutes_played']].head(10))
    
    # Save
    output_file = 'soccer_injury_dataset_enhanced.csv'
    new_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved to: {output_file}")
    
    # Stats on data completeness
    print("\n" + "="*80)
    print("DATA COMPLETENESS:")
    print("="*80)
    for col in ['minutes_played', 'distance_covered', 'fouls_won', 'take_ons', 'duels']:
        non_null = new_df[col].notna().sum()
        pct = (non_null / len(new_df)) * 100
        print(f"   {col}: {non_null}/{len(new_df)} ({pct:.1f}%)")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    update_dataset_with_whoscored_data()
