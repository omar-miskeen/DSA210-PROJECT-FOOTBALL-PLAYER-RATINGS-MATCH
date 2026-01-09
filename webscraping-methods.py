        
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import pandas as pd
import json
import time
import random
import os

class SofaScoreScraper:
    def __init__(self):
        # FC Barcelona Players Config
        self.players = {
            'Pedri': {
                'id': '992587',
                'tournaments': {
                    'La Liga': {'tournament_id': '8', 'season_id': '61643'},
                    'Champions League': {'tournament_id': '7', 'season_id': '61644'},
                    'Copa del Rey': {'tournament_id': '329', 'season_id': '66885'},
                    'Supercopa': {'tournament_id': '213', 'season_id': '66001'}
                }
            },
            'Lewandowski': {
                'id': '41789',
                'tournaments': {
                    'La Liga': {'tournament_id': '8', 'season_id': '61643'},
                    'Champions League': {'tournament_id': '7', 'season_id': '61644'},
                    'Copa del Rey': {'tournament_id': '329', 'season_id': '66885'},
                    'Supercopa': {'tournament_id': '213', 'season_id': '66001'}
                }
            },
            'Yamal': {
                'id': '1402912',
                'tournaments': {
                    'La Liga': {'tournament_id': '8', 'season_id': '61643'},
                    'Champions League': {'tournament_id': '7', 'season_id': '61644'},
                    'Copa del Rey': {'tournament_id': '329', 'season_id': '66885'},
                    'Supercopa': {'tournament_id': '213', 'season_id': '66001'}
                }
            },
            'Raphinha': {
                'id': '831005',
                'tournaments': {
                    'La Liga': {'tournament_id': '8', 'season_id': '61643'},
                    'Champions League': {'tournament_id': '7', 'season_id': '61644'},
                    'Copa del Rey': {'tournament_id': '329', 'season_id': '66885'},
                    'Supercopa': {'tournament_id': '213', 'season_id': '66001'}
                }
            }
        }

        opts = Options()
        # opts.add_argument("--headless") # Uncomment for background execution
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--log-level=3") 
        self.driver = webdriver.Chrome(options=opts)

    def get_player_ratings(self, player_name, player_id, tournament_id, season_id, tournament_name):
        url = f"https://www.sofascore.com/api/v1/player/{player_id}/unique-tournament/{tournament_id}/season/{season_id}/ratings/overall"
        
        try:
            self.driver.get(url)
            time.sleep(1.5) 

            print(f"   -> Processing {tournament_name}...", end=' ', flush=True)
            
            # API returns raw JSON in a <pre> tag
            json_text = self.driver.find_element("tag name", "pre").text
            data = json.loads(json_text)

            matches = []
            season_ratings = data.get("seasonRatings", [])
            
            for match in season_ratings:
                ts = match.get("startTimestamp")
                date_str = pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d')
                
                matches.append({
                    "Player": player_name,
                    "Competition": tournament_name,
                    "Date": date_str,
                    "Opponent": match.get("opponent", {}).get("name", "Unknown"),
                    "Rating": match.get("rating")
                })

            print(f"Found {len(matches)} matches.")
            return matches

        except Exception as e:
            print(f"Failed. Error: {e}")
            return []

    def scrape_player(self, player_name, player_info):
        print(f"[{player_name}] Starting data collection...")
        all_matches = []
        player_id = player_info['id']

        for tournament_name, t in player_info['tournaments'].items():
            matches = self.get_player_ratings(
                player_name, 
                player_id, 
                t['tournament_id'], 
                t['season_id'], 
                tournament_name
            )
            all_matches.extend(matches)
            # Random delay to avoid rate limiting
            time.sleep(random.uniform(1, 2))

        if all_matches:
            return pd.DataFrame(all_matches)
        return None

    def scrape_all(self):
        all_data = []
        print("Starting SofaScore scraper...")
        
        for name, info in self.players.items():
            df = self.scrape_player(name, info)
            if df is not None:
                all_data.append(df)
            print("-" * 30)
            time.sleep(random.uniform(2, 4))

        self.driver.quit()

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"Finished. Total rows: {len(combined)}")
            return combined
        return None

    def save_data(self, df, filename='sofascore_ratings.csv'):
        if df is None or df.empty:
            print("No data to save.")
            return

        # Remove potential duplicates
        df = df.drop_duplicates(subset=['Player', 'Date', 'Opponent'])
        
        try:
            df.to_csv(filename, index=False)
            print(f"Data saved to: {filename}")
            print(df.head())
        except IOError as e:
            print(f"Could not save file: {e}")

if __name__ == "__main__":
    scraper = SofaScoreScraper()
    data = scraper.scrape_all()
    
    if data is not None:
        scraper.save_data(data)
    else:
        print("Scraper finished with no data.")

def get_chrome_driver(headless=True):
    """Configures and returns a Chrome driver instance."""
    opts = Options()
    if headless:
        opts.add_argument('--headless')
        opts.add_argument('--disable-gpu')
    
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--disable-blink-features=AutomationControlled')
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option('useAutomationExtension', False)
    opts.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)

class FBrefScraper:
    def __init__(self):
        self.players = {
            'Lewandowski': 'https://fbref.com/en/players/8d78e732/matchlogs/2024-2025/summary/Robert-Lewandowski-Match-Logs',
            'Yamal': 'https://fbref.com/en/players/82ec26c1/matchlogs/2024-2025/Lamine-Yamal-Match-Logs',
            'Raphinha': 'https://fbref.com/en/players/3423f250/matchlogs/2024-2025/Raphinha-Match-Logs',
            'Pedri': 'https://fbref.com/en/players/0d9b2d31/matchlogs/2024-2025/Pedri-Match-Logs'
        }
    
    def scrape_player(self, player_name, url):
        print(f"[{player_name}] Starting scrape...")
        
        driver = None
        try:
            driver = get_chrome_driver(headless=True)
            driver.get(url)
            
            # Random wait for page elements
            wait_time = random.uniform(12, 18)
            print(f"Waiting {wait_time:.1f}s for tables...")
            time.sleep(wait_time)
            
            tables = driver.find_elements(By.CSS_SELECTOR, "table.stats_table")
            matches = []

            for table in tables:
                rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
                for row in rows:
                    try:
                        # Skip header rows or empty dates
                        th = row.find_element(By.TAG_NAME, "th")
                        date = th.text.strip()
                        if not date:
                            continue
                    except:
                        continue

                    match_data = {'Player': player_name, 'Date': date}
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    for cell in cells:
                        stat_name = cell.get_attribute('data-stat')
                        value = cell.text.strip()
                        if stat_name:
                            match_data[stat_name] = value

                    # Check validity
                    if 'team' in match_data and match_data['team']:
                        matches.append(match_data)
            
            df = pd.DataFrame(matches)

            # Filter for Barcelona games only
            if not df.empty and 'team' in df.columns:
                df = df[df['team'].str.contains('Barcelona', case=False, na=False)].copy()

            print(f"Matches found: {len(df)}")
            return df
            
        except Exception as e:
            print(f"Error scraping {player_name}: {e}")
            return None
            
        finally:
            if driver:
                driver.quit()
    
    def scrape_all(self):
        all_data = []
        
        for player_name, url in self.players.items():
            df = self.scrape_player(player_name, url)
            if df is not None and len(df) > 0:
                all_data.append(df)
            
            # Pause between players to be polite to the server
            wait_time = random.uniform(10, 15)
            print(f"Sleeping {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"Scrape complete. Total rows: {len(combined)}")
            return combined
        return None
    
    def save_data(self, df, filename='fbref_barcelona_only.csv'):
        if df is None or df.empty:
            print("No data to save.")
            return

        # Basic column cleaning
        rename_map = { 
            'date':'Date', 'comp':'Competition', 'team':'Team', 
            'opponent':'Opponent', 'game_started':'Started', 
            'minutes':'Min', 'goals':'Gls', 'assists':'Ast' 
        }
        df = df.rename(columns=rename_map)

        # Double check filtering
        if 'Team' in df.columns:
            df = df[df['Team'].str.contains('Barcelona', case=False, na=False)]

        try:
            df.to_csv(filename, index=False)
            print(f"File saved: {filename}")
            print(df.head())
        except IOError as e:
            print(f"Save failed: {e}")

def main():
    print("Starting FBref Scraper (Barcelona Filter)...")
    
    scraper = FBrefScraper()
    data = scraper.scrape_all()
    
    if data is not None:
        scraper.save_data(data)
    else:
        print("Finished with no data.")

if __name__ == "__main__":
    main()
    
    
# Load datasets
fbref_df = pd.read_csv("fbref_barcelona_only.csv")
sofascore_df = pd.read_csv("sofascore_ratings.csv")

# Standardize dates
fbref_df['Date'] = pd.to_datetime(fbref_df['Date'])
sofascore_df['Date'] = pd.to_datetime(sofascore_df['Date'])

# Merge on Player + Date only
merged_df = pd.merge(
    fbref_df,
    sofascore_df[['Player', 'Date', 'Rating']],  # only keep relevant columns
    how='left',
    on=['Player', 'Date']
)

# Sort by player and date
merged_df = merged_df.sort_values(['Player', 'Date']).reset_index(drop=True)

# Save
merged_df.to_csv("fbref_sofascore_merged_by_date_only.csv", index=False)

print(f"Merged dataset saved. Total rows: {len(merged_df)}")
print(merged_df.head(10))    