"""
WhoScored Match Stats Scraper using Playwright
Searches for players and extracts offensive match statistics
"""

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import re
from urllib.parse import quote


# Player name mappings for search
PLAYER_SEARCH_NAMES = {
    "Ousmane Dembélé": "Ousmane Dembele",
    "Pedri": "Pedri",
    "Cole Palmer": "Cole Palmer",
    "Martin Ødegaard": "Martin Odegaard",
    "Désiré Doué": "Desire Doue"
}


def search_player_whoscored(page, player_name):
    """
    Search for a player on WhoScored and return their profile link.
    
    Args:
        page: Playwright page object
        player_name: Name of the player to search
        
    Returns:
        Player profile URL or None if not found
    """
    search_name = PLAYER_SEARCH_NAMES.get(player_name, player_name)
    search_url = f"https://www.whoscored.com/search/?t={quote(search_name)}"
    
    print(f"🔍 Searching for: {search_name}")
    print(f"   URL: {search_url}")
    
    try:
        page.goto(search_url, timeout=60000)
        time.sleep(3)
        
        # Look for the search-result div with Players table
        print("   Looking for search results...")
        
        # Wait for the search-result div to load
        page.wait_for_selector('div.search-result', timeout=10000)
        
        # Find the first player link in the table
        player_link = page.locator('div.search-result table tbody tr td a[href*="/players/"]').first
        
        if player_link:
            href = player_link.get_attribute('href')
            full_url = f"https://www.whoscored.com{href}" if href.startswith('/') else href
            print(f"   ✅ Found player: {full_url}")
            return full_url
        else:
            print("   ❌ No player found in search results")
            return None
            
    except Exception as e:
        print(f"   ❌ Search error: {e}")
        return None


def get_player_offensive_stats(page, player_url, before_date, num_matches=5):
    """
    Navigate to player page and extract offensive match statistics.
    
    Args:
        page: Playwright page object
        player_url: URL of player's WhoScored page
        before_date: datetime object - get matches before this date
        num_matches: Number of matches to retrieve
        
    Returns:
        List of match dictionaries with offensive stats
    """
    print(f"\n📊 Extracting offensive stats...")
    print(f"   Player URL: {player_url}")
    print(f"   Matches before: {before_date.date()}")
    
    matches = []
    
    try:
        # Convert player URL to match statistics URL
        # From: /players/402197/show/pedri or https://www.whoscored.com/players/402197/show/pedri
        # To:   https://www.whoscored.com/players/402197/matchstatistics/pedri
        match_stats_url = player_url.replace('/show/', '/matchstatistics/')
        
        print(f"\n1️⃣  Loading match statistics page directly...")
        print(f"   URL: {match_stats_url}")
        page.goto(match_stats_url, wait_until='domcontentloaded', timeout=60000)
        time.sleep(5)  # Give page time to fully load
        
        # Verify we're on the right page
        current_url = page.url
        print(f"   Current URL: {current_url}")
        
        if 'matchstatistics' not in current_url.lower():
            print(f"   ⚠️  Page redirected away from matchstatistics!")
            print(f"   Trying again...")
            page.goto(match_stats_url, wait_until='load', timeout=60000)
            time.sleep(5)
        
        # Now click on Offensive tab
        print("\n2️⃣  Clicking on Offensive tab...")
        try:
            # Try to find and click on Offensive tab/option
            offensive_selectors = [
                'a[href="#player-matches-stats-offensive"]',  # Specific selector from the page
                'a:has-text("Offensive")',
                'button:has-text("Offensive")',
                'li:has-text("Offensive") a',
                '[data-tab="offensive"]',
            ]
            
            clicked = False
            for selector in offensive_selectors:
                try:
                    # Wait for element to be visible
                    page.wait_for_selector(selector, state='visible', timeout=3000)
                    offensive_tab = page.locator(selector).first
                    offensive_tab.click()
                    
                    print(f"   ✅ Clicked Offensive tab")
                    time.sleep(3)
                    clicked = True
                    break
                except Exception as e:
                    continue
            
            if not clicked:
                print("   ⚠️  Could not find Offensive tab - may already be on correct page or different layout")
        except Exception as e:
            print(f"   ⚠️  Navigation error: {e}")
        
        # Get all available tournaments
        print("\n3️⃣  Getting available tournaments...")
        all_matches = []
        try:
            # Find the tournament dropdown
            tournament_select = page.locator('select[data-backbone-model-attribute-dd="tournamentOptions"]').first
            tournament_options = tournament_select.locator('option').all()
            
            print(f"   Found {len(tournament_options)} tournaments")
            
            # Iterate through each tournament
            for idx, option in enumerate(tournament_options):
                tournament_name = option.text_content().strip()
                tournament_value = option.get_attribute('data-value')
                
                print(f"\n   📋 Tournament {idx+1}/{len(tournament_options)}: {tournament_name}")
                
                # Select this tournament
                tournament_select.select_option(value=tournament_value)
                time.sleep(2)  # Wait for table to reload
                
                # Extract matches from this tournament
                tournament_matches = extract_matches_from_table(page, before_date, num_matches, tournament_name)
                all_matches.extend(tournament_matches)
                
                if len(all_matches) >= num_matches:
                    print(f"   ✅ Collected {len(all_matches)} matches total, stopping")
                    break
            
            matches = all_matches[:num_matches]  # Limit to requested number
            
        except Exception as e:
            print(f"   ⚠️  Could not iterate tournaments: {e}")
            print("   Falling back to single tournament extraction...")
            # Fallback: just extract from current tournament
            matches = extract_matches_from_table(page, before_date, num_matches, "Current Tournament")
        
        return matches
        
    except Exception as e:
        print(f"\n❌ Error getting offensive stats: {e}")
        return []


def extract_matches_from_table(page, before_date, num_matches, tournament_name="Unknown"):
    """
    Extract match data from the currently displayed offensive statistics table.
    
    Args:
        page: Playwright page object
        before_date: datetime - only get matches before this date
        num_matches: Maximum number of matches to extract
        tournament_name: Name of the tournament for labeling
        
    Returns:
        List of match dictionaries
    """
    matches = []
    
    try:
        # Wait for the statistics table to load and be visible
        table_selector = '#player-matches-stats-offensive table, #statistics-table-offensive table, div[id*="offensive"] table'
        page.wait_for_selector(table_selector, state='visible', timeout=5000)
        
        # Get all table rows
        rows = page.locator(f'{table_selector} tbody tr').all()
        print(f"      Found {len(rows)} match rows in {tournament_name}")
        
        matches_found = 0
        for row in rows:
            try:
                cells = row.locator('td').all()
                if len(cells) < 3:
                    continue
                
                # Column structure: 0=Opponent+Score, 1=Tournament, 2=Date, 3+=Stats
                date_text = None
                date_col_idx = None
                for idx in [2, 1, 0, 3]:  # Try different columns
                    if idx < len(cells):
                        text = cells[idx].text_content().strip()
                        if re.search(r'\d+[/-]\d+', text) or re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', text, re.IGNORECASE):
                            date_text = text
                            date_col_idx = idx
                            break
                
                if not date_text:
                    continue
                
                # Parse the date
                match_date = None
                for date_format in ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d %b %Y', '%b %d, %Y']:
                    try:
                        match_date = datetime.strptime(date_text, date_format)
                        break
                    except:
                        continue
                
                # Skip if no date or date is after injury
                if not match_date or match_date >= before_date:
                    continue
                
                # Extract match details from column 0 (Opponent + Score)
                match_text = cells[0].text_content().strip() if len(cells) > 0 else ""
                
                # Extract teams and home/away
                home_team = "Unknown"
                away_team = "Unknown"
                opponent = "Unknown"
                
                # Pattern: "TeamName (H)1 - 2" or "TeamName (A)1 - 2"
                opponent_match = re.search(r'(.+?)\s*\(([HA])\)', match_text)
                if opponent_match:
                    opponent = opponent_match.group(1).strip()
                    location = opponent_match.group(2)  # H or A
                    
                    if location == 'H':
                        home_team = "Player Team"
                        away_team = opponent
                    else:
                        home_team = opponent
                        away_team = "Player Team"
                
                # Extract statistics from remaining columns
                stats = {}
                stat_start_col = (date_col_idx if date_col_idx else 2) + 1
                for idx, cell in enumerate(cells[stat_start_col:], start=stat_start_col):
                    cell_text = cell.text_content().strip()
                    try:
                        stats[f'col_{idx}'] = float(cell_text) if '.' in cell_text else int(cell_text)
                    except:
                        stats[f'col_{idx}'] = cell_text
                
                # Map to data structure
                match_data = {
                    'match_date': match_date,
                    'tournament': tournament_name,
                    'opponent': opponent,
                    'home_team': home_team,
                    'away_team': away_team,
                    'minutes_played': stats.get('col_4'),
                    'goals': stats.get('col_5'),
                    'assists': stats.get('col_6'),
                    'shots': stats.get('col_7'),
                    'key_passes': stats.get('col_8'),
                    'dribbles': stats.get('col_9'),
                    'fouls_won': stats.get('col_10'),
                    'rating': stats.get('col_3'),
                    '_raw_stats': stats
                }
                
                matches.append(match_data)
                matches_found += 1
                
                if matches_found >= num_matches:
                    break
                    
            except Exception as e:
                continue
        
        if matches_found > 0:
            print(f"      ✅ Extracted {matches_found} matches from {tournament_name}")
        
        return matches
        
    except Exception as e:
        print(f"      ❌ Error extracting from {tournament_name}: {e}")
        return []


def get_player_matches_whoscored(page, player_name, before_date, num_matches=5):
    """
    Main function to search for player and get their offensive match statistics.
    
    Args:
        page: Playwright page object
        player_name: Player name to search
        before_date: datetime object - get matches before this date
        num_matches: Number of matches to retrieve
        
    Returns:
        List of match dictionaries with stats, or empty list if no matches found
    """
    print(f"\n{'='*80}")
    print(f"🔍 WhoScored scraper for: {player_name}")
    print(f"📅 Matches before: {before_date.date()}")
    print(f"{'='*80}")
    
    try:
        # Step 1: Search for player
        player_url = search_player_whoscored(page, player_name)
        
        if not player_url:
            print(f"\n⚠️  Could not find player: {player_name}")
            print("   Skipping this injury (no data available)")
            return []
        
        # Step 2: Get offensive stats
        matches = get_player_offensive_stats(page, player_url, before_date, num_matches)
        
        if not matches or len(matches) == 0:
            print(f"\n⚠️  No matches found before {before_date.date()}")
            print("   Skipping this injury (no games before injury in dataset)")
            return []
        
        print(f"\n✅ Successfully retrieved {len(matches)} matches")
        return matches
        
    except Exception as e:
        print(f"\n❌ Error in scraping process: {e}")
        return []


def main():
    """
    WhoScored scraper - creates two separate tables:
    1. Player injuries
    2. Player match history (all matches from current season)
    """
    print("="*80)
    print("🌐 WhoScored Scraper - Season Match History")
    print("="*80)
    print("""
This tool will:
1. Scrape ALL match data from current season (all tournaments) for each player
2. Create one CSV file: player_match_history.csv
""")
    print("="*80)
    
    print(f"\n📊 Players to scrape: {len(PLAYER_SEARCH_NAMES)}")
    for player_name in PLAYER_SEARCH_NAMES.keys():
        print(f"   • {player_name}")
    
    # Launch Playwright
    with sync_playwright() as p:
        print("\n🚀 Launching browser...")
        browser = p.chromium.launch(
            headless=False,
            slow_mo=500,
            args=['--start-maximized']
        )
        
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        )
        
        page = context.new_page()
        print("✅ Browser opened!")
        
        # Storage for match history
        all_matches = []
        
        # For each player, get complete match history
        player_list = list(PLAYER_SEARCH_NAMES.keys())
        for player_idx, player_name in enumerate(player_list, 1):
            print(f"\n{'='*80}")
            print(f"📝 Player {player_idx}/{len(player_list)}: {player_name}")
            print(f"{'='*80}")
            
            try:
                # Search for player  
                player_url = search_player_whoscored(page, player_name)
                
                if not player_url:
                    print(f"   ⚠️  Could not find {player_name}, skipping")
                    continue
                
                # Get all matches for this player (use a far-future date to get all matches)
                far_future = datetime(2030, 1, 1)
                player_matches = get_player_offensive_stats(page, player_url, far_future, num_matches=100)
                
                if player_matches:
                    # Add player name to each match
                    for match in player_matches:
                        match['player'] = player_name
                    
                    all_matches.extend(player_matches)
                    print(f"   ✅ Collected {len(player_matches)} matches for {player_name}")
                else:
                    print(f"   ⚠️  No matches found for {player_name}")
                
                # Small delay between players
                time.sleep(2)
                
            except Exception as e:
                print(f"   ❌ Error scraping {player_name}: {e}")
                continue
        
        # Save results - Player Match History
        print("\n" + "="*80)
        print("💾 Saving results...")
        print("="*80)
        
        if all_matches:
            matches_table = pd.DataFrame(all_matches)
            # Reorder columns for clarity
            cols = ['player', 'match_date', 'tournament', 'opponent', 'home_team', 'away_team',
                    'minutes_played', 'rating', 'goals', 'assists', 'shots', 'key_passes',
                    'dribbles', 'fouls_won']
            # Only include columns that exist
            cols = [c for c in cols if c in matches_table.columns]
            matches_table = matches_table[cols]
            
            matches_table.to_csv('player_match_history.csv', index=False)
            print(f"✅ Saved {len(matches_table)} match records to player_match_history.csv")
            
            # Display sample
            print("\n📊 Sample of match history:")
            print(matches_table.head(10).to_string())
        else:
            print("⚠️  No match data collected")
        
        print("\n" + "="*80)
        print("🎉 Scraping complete!")
        print(f"   Output file: player_match_history.csv")
        print("="*80)
        
        browser.close()
        print("\n✅ Done!")


if __name__ == "__main__":
    main()
