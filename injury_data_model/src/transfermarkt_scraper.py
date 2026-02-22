"""
Transfermarkt Injury Data Scraper

This module handles scraping injury history from Transfermarkt.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
from typing import List, Dict, Optional


# Player configuration with their Transfermarkt URLs
PLAYERS = {
    "Ousmane Dembélé": "https://www.transfermarkt.com/ousmane-dembele/verletzungen/spieler/288230",
    "Pedri": "https://www.transfermarkt.us/pedri/verletzungen/spieler/683840",
    "Cole Palmer": "https://www.transfermarkt.us/cole-palmer/verletzungen/spieler/568177",
    "Martin Ødegaard": "https://www.transfermarkt.com/martin-odegaard/verletzungen/spieler/316264",
    "Désiré Doué": "https://www.transfermarkt.us/desire-doue/verletzungen/spieler/914562"
}

# Muscular injury keywords
MUSCULAR_INJURY_KEYWORDS = [
    'muscle', 'muscular', 'hamstring', 'thigh', 'calf', 'groin',
    'adductor', 'quadriceps', 'strain', 'tear', 'pulled', 'pelvic'
]

# Number of injuries to collect per player
NUM_INJURIES_PER_PLAYER = 50  # Increased to get more historical data

# Minimum date for injuries
MIN_INJURY_DATE = datetime(2020, 1, 1)


def scrape_transfermarkt_injuries(player_name: str, url: str) -> List[Dict]:
    """
    Scrape injury history from Transfermarkt for a specific player.
    
    Args:
        player_name: Name of the player
        url: Transfermarkt injury page URL
        
    Returns:
        List of injury dictionaries
    """
    print(f"Scraping injuries for {player_name}...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        injuries = []
        
        # Find the injury table
        injury_table = soup.find('table', {'class': 'items'})
        if not injury_table:
            print(f"  Warning: No injury table found for {player_name}")
            return injuries
        
        rows = injury_table.find_all('tr', {'class': ['odd', 'even']})
        
        for row in rows:
            try:
                cols = row.find_all('td')
                if len(cols) < 5:
                    continue
                
                # Extract injury type (column index 1)
                injury_type = cols[1].get_text(strip=True)
                
                # Check if it's a muscular injury
                if not any(keyword in injury_type.lower() for keyword in MUSCULAR_INJURY_KEYWORDS):
                    continue
                
                # Extract dates (columns 2 and 3)
                from_date_str = cols[2].get_text(strip=True)
                until_date_str = cols[3].get_text(strip=True)
                
                # Extract days missed (column 4)
                days_str = cols[4].get_text(strip=True)
                days_missed = None
                days_match = re.search(r'(\d+)', days_str)
                if days_match:
                    days_missed = int(days_match.group(1))
                
                # Parse dates
                injury_start = parse_date(from_date_str)
                injury_end = parse_date(until_date_str)
                
                # Only include injuries after August 1, 2025 (current season)
                if injury_start and injury_start >= MIN_INJURY_DATE:
                    injuries.append({
                        'player': player_name,
                        'injury_type': injury_type,
                        'injury_start': injury_start,
                        'injury_end': injury_end,
                        'days_missed': days_missed
                    })
                    
            except Exception as e:
                print(f"  Error parsing row for {player_name}: {e}")
                continue
        
        # Sort by date (most recent first) and limit to NUM_INJURIES_PER_PLAYER
        injuries.sort(key=lambda x: x['injury_start'], reverse=True)
        injuries = injuries[:NUM_INJURIES_PER_PLAYER]
        
        print(f"  Found {len(injuries)} muscular injuries for {player_name} (since 2020)")
        return injuries
        
    except Exception as e:
        print(f"  Error scraping {player_name}: {e}")
        return []


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse date string from Transfermarkt.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Parsed datetime object or None
    """
    if not date_str or date_str == '-':
        return None
    
    # Common date formats on Transfermarkt
    formats = [
        '%d/%m/%Y',   # 30/01/2026 (most common on Transfermarkt)
        '%d.%m.%Y',   # 15.01.2023
        '%b %d, %Y',  # Jan 15, 2023
        '%Y-%m-%d',   # 2023-01-15
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try to extract just month and year if specific date not available
    try:
        # Handle formats like "Jan 2023"
        return datetime.strptime(date_str, '%b %Y')
    except ValueError:
        pass
    
    return None


def get_all_player_injuries() -> Dict[str, List[Dict]]:
    """
    Scrape injuries for all configured players.
    
    Returns:
        Dictionary mapping player names to their injury lists
    """
    all_injuries = {}
    
    for player_name, injury_url in PLAYERS.items():
        injuries = scrape_transfermarkt_injuries(player_name, injury_url)
        all_injuries[player_name] = injuries
    
    return all_injuries
