"""
FBRef Match Data Scraper
========================
Scrapes detailed match logs from fbref.com for multiple seasons
"""

import asyncio
from playwright.async_api import async_playwright
import pandas as pd
from datetime import datetime
import re
from pathlib import Path
import json
from io import StringIO
import random


class FBRefScraper:
    """Scrape match data from fbref.com"""
    
    # Player configurations: name -> (fbref_id, display_name)
    PLAYERS = {
        'pedri': ('0d9b2d31', 'Pedri'),
        'odegaard': ('a1c03554', 'Martin-Odegaard'),
        'palmer': ('51c664e4', 'Cole-Palmer'),
        'dembele': ('5400766b', 'Ousmane-Dembele'),
        'doue': ('a117921c', 'Desire-Doue'),
    }
    
    def __init__(self, max_years_back=6, cookies_file='fbref_cookies.json'):
        self.max_years_back = max_years_back
        self.current_year = 2026
        self.all_player_data = {}
        self.cookies_file = Path(cookies_file)
        
    def generate_seasons(self):
        """Generate season strings going back max_years_back"""
        seasons = []
        for i in range(self.max_years_back):
            year = self.current_year - i
            season = f"{year-1}-{year}"
            seasons.append(season)
        return seasons
    
    def build_url(self, player_id, display_name, season):
        """Build the FBRef URL for a player's season"""
        return f"https://fbref.com/en/players/{player_id}/matchlogs/{season}/{display_name}-Match-Logs"
    
    async def save_cookies(self, context):
        """Save cookies to file"""
        cookies = await context.cookies()
        with open(self.cookies_file, 'w') as f:
            json.dump(cookies, f, indent=2)
        print(f"  💾 Cookies saved to {self.cookies_file}")
    
    async def load_cookies(self, context):
        """Load cookies from file if they exist"""
        if self.cookies_file.exists():
            with open(self.cookies_file, 'r') as f:
                cookies = json.load(f)
            await context.add_cookies(cookies)
            print(f"  🔑 Loaded cookies from {self.cookies_file}")
            return True
        return False
    
    def get_player_id_from_url(self, url):
        """Extract player ID from FBRef URL"""
        # URL format: https://fbref.com/en/players/{id}/{name}...
        if '/en/players/' in url:
            parts = url.split('/en/players/')
            if len(parts) > 1:
                id_and_rest = parts[1].split('/')
                if id_and_rest:
                    return id_and_rest[0]
        return None
    
    async def get_player_url_from_user(self, player_name):
        """Prompt user for player URL and extract ID"""
        search_name = player_name.replace('-', ' ')
        print(f"\n  📋 Enter FBRef URL for {search_name}")
        print(f"     (Navigate to the player's page in browser and copy the URL)")
        print(f"     Example: https://fbref.com/en/players/a1c03554/Martin-Odegaard")
        
        # Get input from user (blocking)
        url = await asyncio.get_event_loop().run_in_executor(None, input, "     URL: ")
        
        player_id = self.get_player_id_from_url(url)
        if player_id:
            print(f"  ✅ Extracted ID: {player_id}")
            return player_id
        else:
            print(f"  ❌ Could not extract player ID from URL")
            return None
    
    async def wait_for_manual_verification(self, page):
        """Wait for user to manually complete Cloudflare verification"""
        print("\n" + "="*70)
        print("⚠️  CLOUDFLARE VERIFICATION DETECTED")
        print("="*70)
        print("Please complete the verification in the browser window.")
        print("Press Enter here once you see the page load successfully...")
        print("="*70)
        
        # Wait for user input
        await asyncio.get_event_loop().run_in_executor(None, input)
        
        print("✅ Verification complete. Waiting for page to stabilize...")
        
        # Wait longer to ensure cookies are set and page is fully loaded
        await asyncio.sleep(5)
        
        # Check if we're actually on the right page now
        title = await page.title()
        if 'cloudflare' not in title.lower() and 'just a moment' not in title.lower():
            print("✅ Page loaded successfully!")
        else:
            print("⚠️  Still seeing Cloudflare. Waiting a bit more...")
            await asyncio.sleep(5)
    
    async def scrape_player_season(self, page, player_id, display_name, season, skip_cloudflare_check=False):
        """Scrape a single season for a player"""
        url = self.build_url(player_id, display_name, season)
        
        print(f"  📥 Fetching {season}... ", end='', flush=True)
        
        try:
            # Navigate to the page with more lenient settings
            response = await page.goto(url, wait_until='domcontentloaded', timeout=90000)
            
            if response.status == 404:
                print("❌ Not found")
                return None
            
            # Random delay to let the page settle and avoid bot detection
            await asyncio.sleep(random.uniform(3, 6))
            
            # Check if we hit Cloudflare (skip if using existing browser)
            if not skip_cloudflare_check:
                title = await page.title()
                content = await page.content()
                
                if 'cloudflare' in title.lower() or 'just a moment' in title.lower() or 'checking your browser' in content.lower():
                    await self.wait_for_manual_verification(page)
                    # After verification, give extra time for page to fully load
                    await asyncio.sleep(3)
            
            # Now try to find the table
            try:
                await page.wait_for_selector('#matchlogs_all', timeout=30000)
            except:
                # Try alternative selectors
                print("(trying alt selector)", end=" ")
                await page.wait_for_selector('table', timeout=10000)
            
            # Extract the full table HTML including the <table> tag
            try:
                # Use outerHTML to get the complete table element
                table_html = await page.locator('#matchlogs_all').evaluate('el => el.outerHTML')
            except:
                # Fallback: try inner_html and wrap it
                try:
                    table_html = await page.inner_html('#matchlogs_all')
                    table_html = f'<table>{table_html}</table>'
                except:
                    # Last resort: get all tables
                    all_tables = await page.locator('table').all()
                    if not all_tables:
                        print("❌ No tables on page")
                        return None
                    table_html = await all_tables[0].evaluate('el => el.outerHTML')
            
            # Parse with pandas using StringIO
            try:
                # Read with header=[0,1] to handle multi-level headers, then flatten
                df_list = pd.read_html(StringIO(table_html), header=[0, 1])
                if not df_list:
                    # Try with single header
                    df_list = pd.read_html(StringIO(table_html), header=0)
            except:
                # Fallback: try reading from page URL directly (leverages existing cookies)
                try:
                    print("(trying URL)", end=" ")
                    df_list = pd.read_html(url, header=[0, 1])
                    if not df_list:
                        df_list = pd.read_html(url, header=0)
                except:
                    print("❌ Parse failed")
                    return None
            
            if not df_list:
                print("❌ No data")
                return None
            
            df = df_list[0]
            
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                # Take the last level of column names (most specific) 
                # This handles the Performance/header structure
                df.columns = [col[-1] if isinstance(col, tuple) else col for col in df.columns]
            
            # Strip any whitespace from column names
            df.columns = df.columns.str.strip() if hasattr(df.columns, 'str') else df.columns
            
            # Clean up the dataframe
            # Find the date column (might be 'Date' or 'Unnamed: 0' or similar)
            date_col = None
            for col in df.columns:
                if 'date' in str(col).lower() or col == 'Unnamed: 0':
                    date_col = col
                    break
            
            if date_col is None:
                print(f"❌ No date column found. Columns: {df.columns.tolist()[:5]}")
                return None
            
            # Rename to standard name
            if date_col != 'Date':
                df.rename(columns={date_col: 'Date'}, inplace=True)
            
            # Remove footer and header repeat rows
            df = df[df['Date'].notna()]
            df = df[df['Date'].astype(str).str.contains(r'\d{4}', na=False)]  # Keep only rows with year in date
            
            # Add season column
            df['Season'] = season
            
            print(f"✅ {len(df)} matches")
            return df
            
        except Exception as e:
            import traceback
            error_detail = str(e)[:100]
            print(f"⚠️  Error: {error_detail}")
            # Uncomment for full traceback during debugging:
            # traceback.print_exc()
            return None
    
    async def scrape_player(self, page, player_key, skip_cloudflare_check=False):
        """Scrape all seasons for a player"""
        player_id, display_name = self.PLAYERS[player_key]
        
        print(f"\n{'='*70}")
        print(f"🔍 Scraping {display_name.replace('-', ' ')}")
        print(f"{'='*70}")
        
        # Ask user for player URL to get current ID
        user_provided_id = await self.get_player_url_from_user(display_name)
        if user_provided_id:
            player_id = user_provided_id
        else:
            print(f"  ⚠️  Using fallback ID: {player_id}")
        
        seasons = self.generate_seasons()
        all_season_data = []
        
        for i, season in enumerate(seasons):
            df = await self.scrape_player_season(page, player_id, display_name, season, skip_cloudflare_check=skip_cloudflare_check)
            
            if df is not None and len(df) > 0:
                all_season_data.append(df)
            
            # Be respectful - random wait between requests to avoid bot detection
            await asyncio.sleep(random.uniform(4, 8))
        
        if all_season_data:
            combined_df = pd.concat(all_season_data, ignore_index=True)
            print(f"\n  ✅ Total matches scraped: {len(combined_df)}")
            
            # Clean and standardize the dataframe
            combined_df = self.clean_dataframe(combined_df, display_name)
            
            return combined_df
        else:
            print(f"\n  ❌ No data found for {display_name}")
            return None
    
    def clean_dataframe(self, df, player_name):
        """Clean and standardize the dataframe"""
        # Flatten multi-level columns if present (shouldn't be needed after scraping, but just in case)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[-1] if isinstance(col, tuple) else col for col in df.columns]
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip() if hasattr(df.columns, 'str') else df.columns
        
        # Add player name as first column
        df.insert(0, 'Player', player_name.replace('-', ' '))
        
        # Convert date to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Convert numeric columns
        numeric_cols = ['Min', 'Gls', 'Ast', 'PK', 'PKatt', 'Sh', 'SoT', 
                       'CrdY', 'CrdR', 'Fls', 'Fld', 'Off', 'Crs', 'TklW', 'Int', 'OG']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Sort by date descending (most recent first)
        if 'Date' in df.columns:
            df = df.sort_values('Date', ascending=False)
        
        return df
    
    async def scrape_all_players(self, use_existing_browser=False, debug_port=9222, browser_type='chrome'):
        """Scrape all players"""
        print("="*70)
        print("FBREF MATCH DATA SCRAPER")
        print("="*70)
        print(f"Seasons: {self.generate_seasons()[0]} to {self.generate_seasons()[-1]}")
        print(f"Players: {len(self.PLAYERS)}")
        print("="*70)
        
        async with async_playwright() as p:
            if use_existing_browser:
                # Get Windows host IP from WSL
                import subprocess
                # try:
                #     # Get Windows host IP from /etc/resolv.conf
                #     result = subprocess.run(['cat', '/etc/resolv.conf'], capture_output=True, text=True)
                #     for line in result.stdout.split('\n'):
                #         if 'nameserver' in line:
                #             windows_ip = line.split()[1]
                #             break
                #     else:
                #         windows_ip = 'localhost'
                # except:
                    # windows_ip = 'localhost'
                windows_ip = 'localhost'
                
                cdp_url = f'http://{windows_ip}:{debug_port}'
                print(f"\n🔗 Connecting to {browser_type.upper()} at {cdp_url}...")
                
                # Connect based on browser type
                if browser_type == 'firefox':
                    print("For Firefox: Run with --remote-debugging-port=9222")
                    browser = await p.firefox.connect_over_cdp(cdp_url)
                else:  # chrome or edge (both use chromium)
                    browser = await p.chromium.connect_over_cdp(cdp_url)
                
                context = browser.contexts[0]  # Use existing context
                page = await context.new_page()
            else:
                # Launch browser with better settings
                browser = await p.chromium.launch(
                    headless=False,
                    args=['--disable-blink-features=AutomationControlled']
                )
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080}
                )
                
                # Try to load existing cookies
                await self.load_cookies(context)
                
                page = await context.new_page()
                
                # Remove webdriver flag
                await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            for i, player_key in enumerate(self.PLAYERS.keys()):
                # Skip Cloudflare checks when using existing browser
                df = await self.scrape_player(page, player_key, skip_cloudflare_check=use_existing_browser)
                
                if df is not None:
                    self.all_player_data[player_key] = df
                    # Save immediately after scraping each player
                    self.save_player_data(player_key, df)
                
                # Save cookies after first successful scrape (only for non-existing browser)
                if i == 0 and df is not None and not use_existing_browser:
                    await self.save_cookies(context)
                
                # Random wait between players to avoid bot detection
                await asyncio.sleep(random.uniform(5, 10))
            
            if not use_existing_browser:
                await browser.close()
        
        return self.all_player_data
    
    def save_player_data(self, player_key, df, output_dir='fbref_data'):
        """Save a single player's data to CSV immediately"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename = f"{player_key}_fbref.csv"
        filepath = output_path / filename
        
        df.to_csv(filepath, index=False)
        print(f"  💾 Saved {filename} ({len(df)} matches)")
    
    def save_data(self, output_dir='fbref_data'):
        """Save scraped data to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print("SAVING DATA")
        print(f"{'='*70}")
        
        for player_key, df in self.all_player_data.items():
            filename = f"{player_key}_fbref.csv"
            filepath = output_path / filename
            
            df.to_csv(filepath, index=False)
            print(f"  ✅ Saved {filename} ({len(df)} matches)")
        
        print(f"\n  📁 All files saved to: {output_path.absolute()}")
    
    def get_summary(self):
        """Print summary statistics"""
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        total_matches = 0
        
        for player_key, df in self.all_player_data.items():
            player_name = self.PLAYERS[player_key][1].replace('-', ' ')
            
            if 'Season' in df.columns:
                seasons = df['Season'].unique()
                print(f"\n{player_name}:")
                print(f"  Total matches: {len(df)}")
                print(f"  Seasons: {', '.join(sorted(seasons, reverse=True))}")
                
                if 'Date' in df.columns and df['Date'].notna().any():
                    earliest = df['Date'].min()
                    latest = df['Date'].max()
                    print(f"  Date range: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
                
                total_matches += len(df)
        
        print(f"\n{'='*70}")
        print(f"Total matches across all players: {total_matches}")
        print(f"{'='*70}")


async def main():
    """Main function"""
    import sys
    
    # Check if user wants to use existing browser
    use_existing = '--existing-chrome' in sys.argv or '-e' in sys.argv or '--existing-edge' in sys.argv or '--existing-firefox' in sys.argv
    
    # Get custom port if specified
    custom_port = None
    for arg in sys.argv:
        if arg.startswith('--port='):
            custom_port = int(arg.split('=')[1])
    
    debug_port = custom_port if custom_port else 9222
    
    # Determine browser type
    if '--existing-edge' in sys.argv:
        browser_type = 'edge'
        browser_name = 'Edge'
        browser_command = f'msedge.exe --remote-debugging-port={debug_port} --remote-debugging-address=0.0.0.0'
    elif '--existing-firefox' in sys.argv:
        browser_type = 'firefox'
        browser_name = 'Firefox'
        browser_command = f'firefox.exe --remote-debugging-port={debug_port}'
    else:
        browser_type = 'chrome'
        browser_name = 'Chrome'
        browser_command = f'"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port={debug_port} --remote-debugging-address=0.0.0.0 --user-data-dir=C:\\temp\\chrome-debug-profile'
    
    scraper = FBRefScraper(max_years_back=6)
    
    if use_existing:
        print(f"\n📌 USING EXISTING {browser_name.upper()} BROWSER MODE")
        print("=" * 70)
        print("Steps to set up:")
        print(f"1. On Windows, open CMD or PowerShell and run:")
        print(f'   {browser_command}')
        print(f"2. In {browser_name}, navigate to fbref.com and complete any Cloudflare verification")
        print(f"3. Keep {browser_name} open and press Enter here to continue...")
        print("=" * 70)
        input()
    
    # Scrape all players
    await scraper.scrape_all_players(use_existing_browser=use_existing, browser_type=browser_type, debug_port=debug_port)
    
    # Save data
    scraper.save_data(output_dir='fbref_data')
    
    # Print summary
    scraper.get_summary()
    
    print("\n✅ Scraping complete!")


if __name__ == "__main__":
    asyncio.run(main())
