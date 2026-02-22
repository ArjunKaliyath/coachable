"""
Manual WhoScored Data Extraction Tool
You navigate to the page, then the script extracts the data
"""

from playwright.sync_api import sync_playwright
import pandas as pd
from datetime import datetime
import json
import re


def extract_from_current_page(page):
    """
    Extract match data from whatever page is currently open
    """
    print("\n" + "="*80)
    print("📄 ANALYZING CURRENT PAGE")
    print("="*80)
    
    current_url = page.url
    print(f"URL: {current_url}")
    
    # Save page for debugging
    html = page.content()
    with open('whoscored_current_page.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("💾 Saved HTML to: whoscored_current_page.html")
    
    page.screenshot(path='whoscored_current_page.png')
    print("📸 Saved screenshot to: whoscored_current_page.png")
    
    # Try to find all tables
    print("\n🔍 Looking for tables...")
    tables = page.locator('table').all()
    print(f"Found {len(tables)} tables on the page")
    
    # Try to extract any visible match data
    print("\n🔍 Looking for match data patterns...")
    
    # Look for date patterns
    all_text = page.inner_text('body')
    dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', all_text)
    print(f"Found {len(dates)} date patterns: {dates[:10]}")
    
    # Try different table selectors
    match_data = []
    
    for i, table in enumerate(tables):
        print(f"\nTable {i+1}:")
        try:
            # Get rows
            rows = table.locator('tr').all()
            print(f"  - {len(rows)} rows")
            
            # Get first 3 rows to see structure
            for j, row in enumerate(rows[:3]):
                cells = row.locator('td, th').all()
                cell_texts = [c.inner_text().strip()[:40] for c in cells]
                print(f"  Row {j}: {cell_texts}")
            
            # If this looks like match data, try to extract
            if len(rows) > 3:
                print(f"  ✅ This table has enough rows, attempting extraction...")
                
                for row_idx, row in enumerate(rows):
                    try:
                        cells = row.locator('td').all()
                        if len(cells) < 3:
                            continue
                        
                        row_text = row.inner_text()
                        
                        # Look for date in first few cells
                        date_found = None
                        for cell in cells[:3]:
                            cell_text = cell.inner_text().strip()
                            # Try to parse as date
                            for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y']:
                                try:
                                    date_found = datetime.strptime(cell_text, fmt)
                                    break
                                except:
                                    continue
                            if date_found:
                                break
                        
                        if date_found:
                            match_info = {
                                'row_index': row_idx,
                                'date': date_found,
                                'raw_text': row_text[:100],
                                'num_cells': len(cells)
                            }
                            match_data.append(match_info)
                    
                    except Exception as e:
                        continue
        
        except Exception as e:
            print(f"  Error analyzing table: {e}")
    
    print(f"\n📊 Extracted {len(match_data)} potential matches:")
    for m in match_data[:10]:
        print(f"  - {m['date'].date()}: {m['raw_text']}")
    
    return match_data


def manual_extraction_tool():
    """
    Manual navigation tool - you go to the page, we extract the data
    """
    print("="*80)
    print("🎯 MANUAL WHOSCORED DATA EXTRACTION")
    print("="*80)
    print("""
This tool will:
1. Open a browser
2. Wait for YOU to navigate to the correct page
3. Once you're on the player's match history page
4. You tell it to extract the data
5. It saves the page and attempts extraction

YOU CONTROL THE NAVIGATION!
""")
    print("="*80)
    
    input("\nPress Enter to open browser...")
    
    with sync_playwright() as p:
        print("\n🚀 Launching browser...")
        browser = p.chromium.launch(
            headless=False,
            slow_mo=100
        )
        
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        
        page = context.new_page()
        
        # Start at WhoScored homepage
        print("📍 Navigating to WhoScored...")
        page.goto("https://www.whoscored.com")
        
        print("\n" + "="*80)
        print("🖱️  YOUR TURN!")
        print("="*80)
        print("""
In the browser window:
1. Solve any bot challenges
2. Search for the player (e.g., "Mason Mount")
3. Click on their name
4. Go to their "Fixtures" or "Match Log" tab
5. Make sure you can see their match history

When you're on the CORRECT page with the match list visible,
come back here and press Enter.
""")
        print("="*80)
        
        input("\n>>> I'm on the correct page now, press Enter to extract...")
        
        # Extract from current page
        match_data = extract_from_current_page(page)
        
        print("\n" + "="*80)
        print("💡 NEXT STEPS")
        print("="*80)
        print("""
Files saved:
- whoscored_current_page.html (page source)
- whoscored_current_page.png (screenshot)

Review these files to see the page structure.
Then we can write a custom extractor for this specific layout.
""")
        
        input("\nPress Enter to close browser...")
        browser.close()
    
    print("\n✅ Done! Check the saved files to debug the page structure.")


if __name__ == "__main__":
    manual_extraction_tool()
