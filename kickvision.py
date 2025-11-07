#!/usr/bin/env python3
"""
KickVision v1.3.0 — Enhanced Free Edition
Added: Results comparison, live scores, FPL section, improved formatting
Enhanced: Goal distribution, menu system, fixture display
"""

import os
import re
import time
import zipfile
import logging
import json
import random
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta

import numpy as np
import requests
import difflib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import telebot
from telebot import types
from flask import Flask, request

# === CONFIG ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
API_BASE = 'https://api.football-data.org/v4'
FPL_BASE = 'https://fantasy.premierleague.com/api'
ZIP_FILE = 'clubs.zip'
CACHE_FILE = 'team_cache.json'
LEAGUES_CACHE_FILE = 'leagues_cache.json'
CACHE_TTL = 86400
SIMS_PER_MODEL = 500
TOTAL_MODELS = 50

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger('kickvision')
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# === GLOBAL STATE ===
user_rate = defaultdict(list)
TEAM_ALIASES = {}
TEAM_CACHE = {}
LEAGUES_CACHE = {}
PENDING_MATCH = {}
USER_SESSIONS = set()
ODDS_CACHE = {}
LOADING_MSGS = {}
HELP_STATE = {}

# === PERFORMANCE OPTIMIZATIONS ===
PREDICTION_CACHE = {}
TEAM_RESOLVE_CACHE = {}
USER_HISTORY = defaultdict(list)
LEAGUES_LOADED = {}

# === ENHANCED FEATURES CACHE ===
STANDINGS_CACHE = {}
SCORERS_CACHE = {}
MATCHDAY_CACHE = {}
LIVE_SCORES_CACHE = {}
FPL_CACHE = {}
RESULTS_CACHE = {}

# Cache durations (seconds)
CACHE_DURATIONS = {
    'standings': 7200,      # 2 hours
    'scorers': 43200,      # 12 hours
    'matchday': 3600,      # 1 hour
    'predictions': 3600,   # 1 hour
    'live': 300,           # 5 minutes
    'fpl': 1800,           # 30 minutes
    'results': 3600,       # 1 hour
}

# Educational content
EDUCATIONAL_TIPS = [
    "💡 **Tip**: Never bet more than 5% of your bankroll on a single match",
    "🔍 **Strategy**: Look for value bets where bookmakers underestimate teams",
    "⚡ **Discipline**: Don't chase losses - stick to your strategy",
    "📊 **Research**: Always check team news and lineups before betting",
    "🎯 **Focus**: Specialize in 2-3 leagues you know well",
    "💎 **Patience**: Wait for the right opportunities, don't force bets",
    "📈 **Tracking**: Keep a record of all your bets to analyze performance",
    "🛡️ **Safety**: Use reputable bookmakers with proper licenses"
]

# Match preview templates
MATCH_PREVIEWS = [
    "Key battle: Midfield control could decide this encounter",
    "Watch for set pieces - both teams have aerial threats",
    "Recent form suggests we might see goals in this one",
    "Defensive solidity vs attacking flair - classic matchup",
    "Team news could be crucial with some key players doubtful",
    "Historical meetings between these sides have been entertaining",
    "Both managers known for tactical flexibility - intriguing matchup",
    "Critical 3 points at stake with table position implications"
]

# === LEAGUE MAP ===
LEAGUE_MAP = {
    "premier league": 2021, "epl": 2021, "pl": 2021,
    "la liga": 2014, "laliga": 2014, "liga": 2014,
    "bundesliga": 2002, "bundes": 2002,
    "serie a": 2019, "seria": 2019,
    "ligue 1": 2015, "ligue": 2015,
    "uefa champions league": 2001, "ucl": 2001, "champions": 2001,
    "europa league": 2018, "uel": 2018, "europa": 2018,
    "championship": 2016, "efl": 2016,
    "eredivisie": 2003,
    "primeira liga": 2017, "portugal": 2017,
    "super lig": 2036, "turkey": 2036,
    "mls": 2011, "usa": 2011,
    "brasileirao": 2013, "brazil": 2013,
    "liga mx": 2012, "mexico": 2012
}

# League names for display
LEAGUE_DISPLAY_NAMES = {
    2021: "Premier League",
    2014: "La Liga", 
    2002: "Bundesliga",
    2019: "Serie A",
    2015: "Ligue 1",
    2001: "Champions League",
    2018: "Europa League",
    2016: "Championship",
    2003: "Eredivisie",
    2017: "Primeira Liga", 
    2036: "Süper Lig",
    2011: "MLS",
    2013: "Brasileirão",
    2012: "Liga MX"
}

# === LOAD ALIASES FROM ZIP ===
log.info(f"Loading aliases from {ZIP_FILE}...")
if not os.path.exists(ZIP_FILE):
    log.error(f"{ZIP_FILE} NOT FOUND!")
    raise SystemExit(1)

try:
    with zipfile.ZipFile(ZIP_FILE, 'r') as z:
        for file in z.namelist():
            if not file.endswith('.txt'): continue
            if any(x in file.lower() for x in ['alphabet','duplicate','license','notes','readme']): continue
            with z.open(file) as f:
                lines = [l.decode('utf-8').strip() for l in f if l.strip()]
            for line in lines:
                parts = [p.strip() for p in re.split(r'\s*[|,]\s*', line.strip()) if p.strip()]
                if not parts: continue
                official = parts[0]
                for alias in parts:
                    TEAM_ALIASES[alias.lower()] = official
                clean_file = os.path.basename(file).replace('.txt','').replace('_',' ').lower()
                TEAM_ALIASES[clean_file] = official
    for off in set(TEAM_ALIASES.values()):
        TEAM_ALIASES[off.lower()] = off
    log.info(f"Loaded {len(TEAM_ALIASES)} aliases")
except Exception as e:
    log.exception("ZIP ERROR")
    raise SystemExit(1)

# === HTTP SESSION ===
session = requests.Session()
session.headers.update({'X-Auth-Token': API_KEY})
retries = Retry(total=5, backoff_factor=2, status_forcelist=[429,500,502,503,504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# === TELEBOT ===
bot = telebot.TeleBot(BOT_TOKEN)
time.sleep(2)

# === CACHE ===
def load_cache():
    global TEAM_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                now = time.time()
                new_cache = {}
                for k, v in data.items():
                    if now - v['time'] < CACHE_TTL:
                        if k.startswith("league_"):
                            lid = int(k.split("_")[1])
                            fixed_teams = [team + (lid,) if len(team) == 4 else team for team in v['data']]
                            new_cache[k] = {'time': v['time'], 'data': fixed_teams}
                        else:
                            new_cache[k] = v
                TEAM_CACHE = new_cache
            log.info(f"Loaded cache: {len(TEAM_CACHE)} entries")
        except Exception as e:
            log.exception("Cache load error")

def save_cache():
    with open(CACHE_FILE, 'w') as f:
        json.dump(TEAM_CACHE, f)

load_cache()

# === FAST TEAM RESOLUTION ===
def fast_resolve_alias(name):
    low = re.sub(r'[^a-z0-9\s]', '', str(name).lower().strip())
    if low in TEAM_RESOLVE_CACHE:
        return TEAM_RESOLVE_CACHE[low]
    if low in TEAM_ALIASES: 
        result = TEAM_ALIASES[low]
        TEAM_RESOLVE_CACHE[low] = result
        return result
    for alias, official in TEAM_ALIASES.items():
        if low in alias or alias in low: 
            TEAM_RESOLVE_CACHE[low] = official
            return official
    TEAM_RESOLVE_CACHE[low] = name
    return name

# === NEW: MATCH IMPORTANCE INDICATOR ===
def get_match_importance(hname, aname):
    big_matches = [
        "man city vs liverpool", "liverpool vs man city",
        "barcelona vs real madrid", "real madrid vs barcelona", 
        "man united vs chelsea", "chelsea vs man united",
        "arsenal vs tottenham", "tottenham vs arsenal",
        "bayern vs dortmund", "dortmund vs bayern",
        "milan vs inter", "inter vs milan",
        "psg vs marseille", "marseille vs psg"
    ]
    match_key = f"{hname.lower()} vs {aname.lower()}"
    if match_key in big_matches:
        return "🔥 **BIG MATCH ALERT** - High stakes encounter!"
    
    derby_matches = [
        "man united vs man city", "man city vs man united",
        "liverpool vs everton", "everton vs liverpool",
        "arsenal vs chelsea", "chelsea vs arsenal",
        "celtic vs rangers", "rangers vs celtic"
    ]
    if match_key in derby_matches:
        return "⚔️ **DERBY DAY** - Local rivalry intensifies!"
    
    return "⚽ **Regular Match** - Good betting opportunity"

# === NEW: MATCH PREVIEW GENERATOR ===
def generate_match_preview():
    return random.choice(MATCH_PREVIEWS)

# === NEW: EDUCATIONAL TIP ===
def get_educational_tip():
    return random.choice(EDUCATIONAL_TIPS)

# === NEW: LEAGUE STANDINGS ===
def get_league_standings(league_name):
    lid = LEAGUE_MAP.get(league_name.lower())
    if not lid:
        return "League not supported."
    
    cache_key = f"standings_{lid}"
    now = time.time()
    
    # Check cache
    if cache_key in STANDINGS_CACHE and now - STANDINGS_CACHE[cache_key]['time'] < CACHE_DURATIONS['standings']:
        return STANDINGS_CACHE[cache_key]['data']
    
    data = safe_get(f"{API_BASE}/competitions/{lid}/standings")
    if not data or 'standings' not in data:
        return "Could not fetch standings at the moment."
    
    try:
        table = data['standings'][0]['table']  # Total standings
        standings_text = [f"*{LEAGUE_DISPLAY_NAMES.get(lid, league_name.title())} Standings* 📊\n"]
        standings_text.append("═" * 40)
        
        for i, team in enumerate(table[:10]):  # Top 10 teams
            position = team['position']
            team_name = team['team']['name']
            played = team['playedGames']
            points = team['points']
            goals_for = team['goalsFor']
            goals_against = team['goalsAgainst']
            goal_diff = team['goalDifference']
            form = team.get('form', '')
            
            emoji = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else "🔸"
            
            standings_text.append(
                f"{emoji} `{position:2d}. {team_name[:15]:15s} P:{played:2d} PTS:{points:2d} GD:{goal_diff:+2d}`"
            )
        
        # Add relegation zone if available
        if len(table) > 15:
            standings_text.append("\n*Relegation Zone:*")
            standings_text.append("─" * 20)
            for team in table[-3:]:
                position = team['position']
                team_name = team['team']['name']
                points = team['points']
                standings_text.append(f"🔻 `{position:2d}. {team_name[:15]:15s} PTS:{points:2d}`")
        
        result = '\n'.join(standings_text)
        STANDINGS_CACHE[cache_key] = {'time': now, 'data': result}
        return result
        
    except Exception as e:
        log.error(f"Error parsing standings: {e}")
        return "Error parsing standings data."

# === NEW: TOP SCORERS ===
def get_top_scorers(league_name):
    lid = LEAGUE_MAP.get(league_name.lower())
    if not lid:
        return "League not supported."
    
    cache_key = f"scorers_{lid}"
    now = time.time()
    
    if cache_key in SCORERS_CACHE and now - SCORERS_CACHE[cache_key]['time'] < CACHE_DURATIONS['scorers']:
        return SCORERS_CACHE[cache_key]['data']
    
    data = safe_get(f"{API_BASE}/competitions/{lid}/scorers")
    if not data or 'scorers' not in data:
        return "Could not fetch top scorers at the moment."
    
    try:
        scorers_text = [f"*{LEAGUE_DISPLAY_NAMES.get(lid, league_name.title())} - Top Scorers* ⚽\n"]
        scorers_text.append("═" * 40)
        
        for i, scorer in enumerate(data['scorers'][:10]):  # Top 10 scorers
            player_name = scorer['player']['name']
            team_name = scorer['team']['name']
            goals = scorer['goals'] or 0
            assists = scorer.get('assists', 0) or 0
            
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🔹"
            scorers_text.append(f"{emoji} `{player_name[:18]:18s} - {goals:2d} goals, {assists:2d} assists ({team_name[:12]:12s})`")
        
        result = '\n'.join(scorers_text)
        SCORERS_CACHE[cache_key] = {'time': now, 'data': result}
        return result
        
    except Exception as e:
        log.error(f"Error parsing scorers: {e}")
        return "Error loading top scorers."

# === NEW: CURRENT MATCHDAY ===
def get_current_matchday(league_name):
    lid = LEAGUE_MAP.get(league_name.lower())
    if not lid:
        return "League not supported."
    
    cache_key = f"matchday_{lid}"
    now = time.time()
    
    if cache_key in MATCHDAY_CACHE and now - MATCHDAY_CACHE[cache_key]['time'] < CACHE_DURATIONS['matchday']:
        return MATCHDAY_CACHE[cache_key]['data']
    
    # Get competition info to find current matchday
    comp_data = safe_get(f"{API_BASE}/competitions/{lid}")
    if not comp_data:
        return "Could not fetch competition data."
    
    current_matchday = comp_data.get('currentSeason', {}).get('currentMatchday')
    if not current_matchday:
        return "No current matchday information available."
    
    # Get matches for current matchday
    data = safe_get(f"{API_BASE}/competitions/{lid}/matches", {
        'matchday': current_matchday,
        'status': 'SCHEDULED'
    })
    
    if not data or 'matches' not in data:
        return f"No fixtures found for matchday {current_matchday}."
    
    try:
        matchday_text = [f"*{LEAGUE_DISPLAY_NAMES.get(lid, league_name.title())} - Matchday {current_matchday}* 📅\n"]
        matchday_text.append("═" * 40)
        
        for match in data['matches'][:8]:  # First 8 matches
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            date_str = match['utcDate'][:10]
            time_str = match['utcDate'][11:16]
            
            matchday_text.append(f"`📅 {date_str} ⏰ {time_str} UTC`")
            matchday_text.append(f"**{home_team}** vs **{away_team}**")
            matchday_text.append("─" * 30)
        
        result = '\n'.join(matchday_text)
        MATCHDAY_CACHE[cache_key] = {'time': now, 'data': result}
        return result
        
    except Exception as e:
        log.error(f"Error parsing matchday: {e}")
        return "Error loading matchday fixtures."

# === NEW: LIVE SCORES ===
def get_live_scores():
    cache_key = "live_scores"
    now = time.time()
    
    if cache_key in LIVE_SCORES_CACHE and now - LIVE_SCORES_CACHE[cache_key]['time'] < CACHE_DURATIONS['live']:
        return LIVE_SCORES_CACHE[cache_key]['data']
    
    try:
        data = safe_get(f"{API_BASE}/matches", {'status': 'LIVE', 'limit': 20})
        if not data or 'matches' not in data or not data['matches']:
            return "No live matches at the moment."
        
        live_text = ["*⚽ LIVE SCORES ⚽*\n"]
        live_text.append("═" * 40)
        
        for match in data['matches'][:10]:  # First 10 live matches
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            home_score = match['score']['fullTime']['home'] or 0
            away_score = match['score']['fullTime']['away'] or 0
            minute = match.get('minute', 'LIVE')
            competition = match['competition']['name']
            
            live_text.append(f"**{competition}**")
            live_text.append(f"🕒 **{minute}'**")
            live_text.append(f"**{home_team}** {home_score} - {away_score} **{away_team}**")
            live_text.append("─" * 30)
        
        result = '\n'.join(live_text)
        LIVE_SCORES_CACHE[cache_key] = {'time': now, 'data': result}
        return result
        
    except Exception as e:
        log.error(f"Error fetching live scores: {e}")
        return "Error fetching live scores."

# === NEW: FANTASY PREMIER LEAGUE (PUBLIC API - NO LOGIN REQUIRED) ===
def get_fpl_data():
    cache_key = "fpl_data"
    now = time.time()
    
    if cache_key in FPL_CACHE and now - FPL_CACHE[cache_key]['time'] < CACHE_DURATIONS['fpl']:
        return FPL_CACHE[cache_key]['data']
    
    try:
        # Get current gameweek from public API
        bootstrap = requests.get(f"{FPL_BASE}/bootstrap-static/", timeout=10).json()
        current_gw = None
        for event in bootstrap['events']:
            if event['is_current']:
                current_gw = event['id']
                break
        
        if not current_gw:
            return "Could not fetch FPL data at the moment."
        
        # Get top players for current gameweek from elements
        elements = bootstrap['elements']
        elements_dict = {elem['id']: elem for elem in elements}
        
        # Get player performances - try to get live data, fall back to static
        try:
            gw_data = requests.get(f"{FPL_BASE}/event/{current_gw}/live/", timeout=10).json()
            player_performances = []
            
            for element in gw_data['elements']:
                player_id = element['id']
                stats = element['stats']
                player_info = elements_dict.get(player_id, {})
                
                if stats['total_points'] > 0:
                    player_performances.append({
                        'name': player_info.get('web_name', 'Unknown'),
                        'team': player_info.get('team', 0),
                        'points': stats['total_points'],
                        'goals': stats['goals_scored'],
                        'assists': stats['assists'],
                        'bonus': stats['bonus']
                    })
        except:
            # Fallback: use static data from bootstrap
            player_performances = []
            for element in elements:
                if element['event_points'] and element['event_points'] > 0:
                    player_performances.append({
                        'name': element.get('web_name', 'Unknown'),
                        'team': element.get('team', 0),
                        'points': element['event_points'],
                        'goals': element.get('goals_scored', 0),
                        'assists': element.get('assists', 0),
                        'bonus': element.get('bonus', 0)
                    })
        
        # Sort by points
        player_performances.sort(key=lambda x: x['points'], reverse=True)
        
        if not player_performances:
            return f"*FPL Gameweek {current_gw}*\n\nNo player data available yet."
        
        fpl_text = [f"*🏆 FPL Gameweek {current_gw} - Top Performers* 🏆\n"]
        fpl_text.append("═" * 50)
        
        team_names = {
            1: 'Arsenal', 2: 'Aston Villa', 3: 'Bournemouth', 4: 'Brentford',
            5: 'Brighton', 6: 'Chelsea', 7: 'Crystal Palace', 8: 'Everton',
            9: 'Fulham', 10: 'Leicester', 11: 'Leeds', 12: 'Liverpool',
            13: 'Man City', 14: 'Man Utd', 15: 'Newcastle', 16: 'Nottm Forest',
            17: 'Southampton', 18: 'Spurs', 19: 'West Ham', 20: 'Wolves'
        }
        
        for i, player in enumerate(player_performances[:15]):
            emoji = "🔥" if player['points'] >= 10 else "⭐" if player['points'] >= 7 else "🔸"
            team = team_names.get(player['team'], 'Unknown')
            fpl_text.append(
                f"{emoji} `{player['name'][:15]:15s} - {player['points']:2d} pts "
                f"({player['goals']}G {player['assists']}A {player['bonus']}B) - {team}`"
            )
        
        result = '\n'.join(fpl_text)
        FPL_CACHE[cache_key] = {'time': now, 'data': result}
        return result
        
    except Exception as e:
        log.error(f"Error fetching FPL data: {e}")
        return "Fantasy Premier League data temporarily unavailable."

# === NEW: RESULTS COMPARISON ===
def get_todays_results_with_comparison():
    cache_key = "todays_results"
    now = time.time()
    
    if cache_key in RESULTS_CACHE and now - RESULTS_CACHE[cache_key]['time'] < CACHE_DURATIONS['results']:
        return RESULTS_CACHE[cache_key]['data']
    
    try:
        today = date.today().isoformat()
        data = safe_get(f"{API_BASE}/matches", {'dateFrom': today, 'dateTo': today, 'status': 'FINISHED'})
        
        if not data or 'matches' not in data or not data['matches']:
            return "No finished matches today yet."
        
        results_text = ["*📊 TODAY'S RESULTS vs PREDICTIONS* 📊\n"]
        results_text.append("═" * 50)
        
        for match in data['matches'][:8]:  # First 8 finished matches
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            home_score = match['score']['fullTime']['home'] or 0
            away_score = match['score']['fullTime']['away'] or 0
            competition = match['competition']['name']
            
            # Get prediction for comparison
            hid = match['homeTeam']['id']
            aid = match['awayTeam']['id']
            pred = predict_with_ids(hid, aid, home_team, away_team, '', '')
            
            # Extract verdict from prediction
            verdict = "Unknown"
            for line in pred.split('\n'):
                if "Verdict:" in line:
                    verdict = line.split("Verdict:")[1].strip().replace('*', '')
                    break
            
            # Determine if prediction was correct
            actual_result = "Draw" if home_score == away_score else "Home Win" if home_score > away_score else "Away Win"
            prediction_correct = "✅" if actual_result in verdict else "❌"
            
            results_text.append(f"**{competition}**")
            results_text.append(f"**{home_team} {home_score} - {away_score} {away_team}**")
            results_text.append(f"Predicted: {verdict} {prediction_correct}")
            results_text.append("─" * 40)
        
        # Add summary
        total_matches = len(data['matches'])
        results_text.append(f"\n*Summary: {total_matches} matches completed today*")
        
        result = '\n'.join(results_text)
        RESULTS_CACHE[cache_key] = {'time': now, 'data': result}
        return result
        
    except Exception as e:
        log.error(f"Error fetching results: {e}")
        return "Error loading today's results."

# === SAFE GET ===
def safe_get(url, params=None):
    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = 60 * (2 ** attempt)
                log.debug(f"429 -> sleep {wait}s")
                time.sleep(wait)
            else:
                log.debug(f"API {r.status_code}")
                return None
        except Exception as e:
            log.debug(f"Request failed: {e}")
            time.sleep(5)
    return None

# === LEAGUES CACHE ===
def load_leagues_cache():
    global LEAGUES_CACHE
    if os.path.exists(LEAGUES_CACHE_FILE):
        try:
            with open(LEAGUES_CACHE_FILE, 'r') as f:
                data = json.load(f)
                now = time.time()
                if now - data['time'] < CACHE_TTL:
                    LEAGUES_CACHE = {int(k): v for k, v in data['leagues'].items()}
                    log.info(f"Loaded leagues cache: {len(LEAGUES_CACHE)}")
                    return True
        except Exception as e:
            log.exception("Leagues cache error")
    return False

def save_leagues_cache():
    with open(LEAGUES_CACHE_FILE, 'w') as f:
        json.dump({'time': time.time(), 'leagues': LEAGUES_CACHE}, f)

def fetch_all_leagues():
    data = safe_get(f"{API_BASE}/competitions")
    if data and 'competitions' in data:
        for comp in data['competitions']:
            lid = comp['id']
            LEAGUES_CACHE[lid] = comp['name']
        save_leagues_cache()
        log.info(f"Fetched {len(LEAGUES_CACHE)} leagues")
        return True
    log.warning("Failed to fetch leagues")
    return False

if not load_leagues_cache():
    fetch_all_leagues()

# === LAZY LEAGUE LOADING ===
def get_league_teams_lazy(league_id):
    if league_id in LEAGUES_LOADED:
        return LEAGUES_LOADED[league_id]
    teams = get_league_teams(league_id)
    LEAGUES_LOADED[league_id] = teams
    return teams

# === GET LEAGUE TEAMS ===
def get_league_teams(league_id):
    key = f"league_{league_id}"
    now = time.time()
    if key in TEAM_CACHE and now - TEAM_CACHE[key]['time'] < CACHE_TTL:
        return TEAM_CACHE[key]['data']
    data = safe_get(f"{API_BASE}/competitions/{league_id}/teams")
    if data and 'teams' in data:
        teams = [(t['id'], t['name'], t.get('shortName',''), t.get('tla',''), league_id) for t in data['teams']]
        TEAM_CACHE[key] = {'time': now, 'data': teams}
        save_cache()
        return teams
    return []

# === FIND CANDIDATES ===
def find_team_candidates(name):
    name_resolved = fast_resolve_alias(name)
    search_key = re.sub(r'[^a-z0-9\s]', '', name_resolved.lower())
    candidates = []
    for lid in LEAGUE_MAP.values():
        teams = get_league_teams_lazy(lid)
        for team in teams:
            tid, tname, tshort, tla, _ = team
            score = max(
                difflib.SequenceMatcher(None, search_key, tname.lower()).ratio(),
                difflib.SequenceMatcher(None, search_key, tshort.lower()).ratio() if tshort else 0,
                1.0 if search_key == tla.lower() else 0
            )
            if score > 0.4:
                league_name = LEAGUES_CACHE.get(lid, f"League {lid}")
                candidates.append((score, tname, tid, tla or tname[:3].upper(), lid, league_name))
    candidates.sort(reverse=True)
    return candidates[:5]

# === AUTO DETECT LEAGUE ===
def auto_detect_league(hid, aid):
    h_matches = safe_get(f"{API_BASE}/teams/{hid}/matches", {'limit': 20, 'status': 'FINISHED'})
    a_matches = safe_get(f"{API_BASE}/teams/{aid}/matches", {'limit': 20, 'status': 'FINISHED'})
    h_leagues = set()
    a_leagues = set()
    if h_matches and 'matches' in h_matches:
        for m in h_matches['matches']:
            lid = m.get('competition', {}).get('id')
            if lid: h_leagues.add(lid)
    if a_matches and 'matches' in a_matches:
        for m in a_matches['matches']:
            lid = m.get('competition', {}).get('id')
            if lid: a_leagues.add(lid)
    common = h_leagues & a_leagues
    if common:
        lid = next(iter(common))
        return lid, LEAGUES_CACHE.get(lid, "League")
    if h_leagues:
        lid = next(iter(h_leagues))
        return lid, LEAGUES_CACHE.get(lid, "League")
    return 0, "Unknown League"

# === IMPROVED WEIGHTED STATS WITH BETTER GOAL DISTRIBUTION ===
def get_weighted_stats(team_id, is_home):
    cache_key = f"stats_{team_id}_{'h' if is_home else 'a'}"
    now = time.time()
    if cache_key in TEAM_CACHE and now - TEAM_CACHE[cache_key]['time'] < 3600:
        return TEAM_CACHE[cache_key]['data']
    data = safe_get(f"{API_BASE}/teams/{team_id}/matches", {'status': 'FINISHED', 'limit': 8})
    if not data or len(data.get('matches', [])) < 3:
        # More realistic default values that allow for more goals
        return (2.2, 1.3) if is_home else (1.6, 1.8)
    
    gf, ga, weights = [], [], []
    for i, m in enumerate(reversed(data['matches'][:8])):
        try:
            home_id = m['homeTeam']['id']
            sh = m['score']['fullTime']['home'] or 0
            sa = m['score']['fullTime']['away'] or 0
            # More balanced weighting
            weight = 1.5 if i < 3 else 1.2 if i < 6 else 1.0
            if home_id == team_id:
                gf.append(sh * weight)
                ga.append(sa * weight)
                weights.append(weight)
            else:
                gf.append(sa * weight)
                ga.append(sh * weight)
                weights.append(weight)
        except: pass
    
    total_weight = sum(weights) if weights else 1
    avg_gf = sum(gf) / total_weight
    avg_ga = sum(ga) / total_weight
    
    # Apply home/away adjustment with more variance
    if is_home:
        avg_gf = avg_gf * 1.15  # Increased home advantage
        avg_ga = avg_ga * 0.9
    else:
        avg_gf = avg_gf * 0.85
        avg_ga = avg_ga * 1.1
    
    stats = (round(max(avg_gf, 0.8), 2), round(max(avg_ga, 0.8), 2))
    TEAM_CACHE[cache_key] = {'time': now, 'data': stats}
    save_cache()
    return stats

# === MARKET ODDS ===
def get_market_odds(hname, aname):
    key = f"{hname.lower()} vs {aname.lower()}"
    if key in ODDS_CACHE and time.time() - ODDS_CACHE[key]['time'] < 1800:
        return ODDS_CACHE[key]['data']
    if not ODDS_API_KEY:
        return None
    try:
        url = "https://api.the-odds-api.com/v4/sports/football_england_premier_league/odds/"
        params = {'apiKey': ODDS_API_KEY, 'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200: return None
        data = r.json()
        for game in data:
            if hname.lower() in game['home_team'].lower() and aname.lower() in game['away_team'].lower():
                for book in game['bookmakers']:
                    if 'bet' in book['key'].lower():
                        odds = book['markets'][0]['outcomes']
                        result = {
                            'home': odds[0]['price'],
                            'draw': odds[1]['price'] if len(odds) > 2 else None,
                            'away': odds[1]['price'] if len(odds) == 2 else odds[2]['price']
                        }
                        ODDS_CACHE[key] = {'time': time.time(), 'data': result}
                        return result
        ODDS_CACHE[key] = {'time': time.time(), 'data': None}
        return None
    except:
        return None

# === IMPROVED SIMULATION WITH BETTER GOAL DISTRIBUTION ===
def run_single_model(seed, h_gf, h_ga, a_gf, a_ga):
    random.seed(seed)
    np.random.seed(seed)
    
    # Enhanced goal expectation calculation with more variance
    home_attack = h_gf * 1.1  # Home advantage
    away_attack = a_gf * 0.95  # Away disadvantage
    
    home_defense = h_ga * 0.9  # Home defensive boost
    away_defense = a_ga * 1.1  # Away defensive weakness
    
    # More dynamic xG calculation with higher potential for goals
    home_xg = (home_attack * away_defense * 1.2) ** 0.6 * random.uniform(0.7, 1.4)
    away_xg = (away_attack * home_defense * 0.9) ** 0.6 * random.uniform(0.7, 1.4)
    
    # Add more randomness for high-scoring games
    if random.random() < 0.1:  # 10% chance for high-scoring game
        home_xg *= random.uniform(1.2, 2.0)
        away_xg *= random.uniform(1.2, 2.0)
    
    # Ensure minimum xG values
    home_xg = max(home_xg, 0.3)
    away_xg = max(away_xg, 0.3)
    
    # Use Poisson distribution for goals
    hg = np.random.poisson(home_xg, SIMS_PER_MODEL)
    ag = np.random.poisson(away_xg, SIMS_PER_MODEL)
    
    return hg, ag

def ensemble_100_models(h_gf, h_ga, a_gf, a_ga):
    seeds = list(range(TOTAL_MODELS))
    all_home_goals = []
    all_away_goals = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(lambda s: run_single_model(s, h_gf, h_ga, a_gf, a_ga), seeds)
        for hg, ag in results:
            all_home_goals.extend(hg)
            all_away_goals.extend(ag)
    
    total_sims = len(all_home_goals) if all_home_goals else 1
    home_win = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h > a) / total_sims
    draw = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h == a) / total_sims
    away_win = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h < a) / total_sims
    
    # More sophisticated score prediction
    score_counts = Counter(zip(all_home_goals, all_away_goals))
    if score_counts:
        most_likely = score_counts.most_common(1)[0][0]
        # Ensure we don't always get low scores
        if most_likely[0] + most_likely[1] < 2 and random.random() < 0.3:
            # Sometimes show a higher scoring alternative
            high_scoring = [score for score in score_counts if score[0] + score[1] >= 3]
            if high_scoring:
                most_likely = random.choice(high_scoring)
    else:
        most_likely = (1, 1)
    
    return {
        'home_win': round(home_win * 100),
        'draw': round(draw * 100),
        'away_win': round(away_win * 100),
        'score': f"{most_likely[0]}-{most_likely[1]}"
    }

# === IMPROVED VERDICT (LESS BIASED) ===
def get_verdict(model, market=None):
    h, d, a = model['home_win'], model['draw'], model['away_win']
    
    # Only use market odds if they're available and reasonable
    if market and market.get('home') and market.get('away') and market.get('draw'):
        mh = 1/market['home']
        ma = 1/market['away']
        md = 1/market['draw']
        total = mh + md + ma
        if total > 0.8:  # Only use if market probabilities make sense
            # Less aggressive blending (40% market, 60% model)
            h = int(h * 0.6 + (mh/total*100 * 0.4))
            d = int(d * 0.6 + (md/total*100 * 0.4))
            a = int(a * 0.6 + (ma/total*100 * 0.4))
    
    # Find the maximum with some tolerance for draws
    max_pct = max(h, d, a)
    
    # Give draws a better chance - if draw is close to max, prefer draw for balanced games
    if d >= max_pct * 0.85 and abs(h - a) < 15:
        return "Draw", h, d, a
    elif h == max_pct:
        return "Home Win", h, d, a
    elif a == max_pct:
        return "Away Win", h, d, a
    else:
        return "Draw", h, d, a  # Fallback to draw

# === CACHED PREDICTION ===
def cached_prediction(hid, aid, hname, aname, h_tla, a_tla):
    prediction_key = f"pred_{hid}_{aid}"
    now = time.time()
    if prediction_key in PREDICTION_CACHE and now - PREDICTION_CACHE[prediction_key]['time'] < CACHE_DURATIONS['predictions']:
        log.info(f"Using cached prediction for {hname} vs {aname}")
        return PREDICTION_CACHE[prediction_key]['data']
    
    lid, league_name = auto_detect_league(hid, aid)
    h_gf, h_ga = get_weighted_stats(hid, True)
    a_gf, a_ga = get_weighted_stats(aid, False)
    model = ensemble_100_models(h_gf, h_ga, a_gf, a_ga)
    market = get_market_odds(hname, aname)
    verdict, h_pct, d_pct, a_pct = get_verdict(model, market)
    
    # Enhanced output with new features
    importance = get_match_importance(hname, aname)
    preview = generate_match_preview()
    education = get_educational_tip()
    
    out = [
        f"*{hname} vs {aname}*",
        f"_{league_name}_",
        "",
        f"**Match Type:** {importance}",
        f"**xG Analysis:** `{h_gf:.2f}` — `{a_gf:.2f}`",
        f"**Win Probability:** `{h_pct}%` | `{d_pct}%` | `{a_pct}%`",
        "",
        f"**Most Likely Score:** `{model['score']}`",
        f"**Verdict:** *{verdict}*",
        "",
        f"**Match Insight:** {preview}",
        "",
        f"**Betting Tip:** {education}"
    ]
    result = '\n'.join(out)
    PREDICTION_CACHE[prediction_key] = {'time': now, 'data': result}
    return result

# === PREDICTION HISTORY ===
def add_to_history(user_id, match, prediction):
    USER_HISTORY[user_id].append({
        'match': match,
        'prediction': prediction,
        'time': time.time()
    })
    if len(USER_HISTORY[user_id]) > 5:
        USER_HISTORY[user_id] = USER_HISTORY[user_id][-5:]

def get_user_history(user_id):
    if user_id not in USER_HISTORY or not USER_HISTORY[user_id]:
        return "No prediction history yet. Make some predictions first!"
    history_text = ["*Your Recent Predictions:* 📊"]
    history_text.append("═" * 40)
    for i, pred in enumerate(reversed(USER_HISTORY[user_id])):
        match = pred['match']
        prediction = pred['prediction']
        time_str = datetime.fromtimestamp(pred['time']).strftime("%H:%M")
        lines = prediction.split('\n')
        verdict_line = next((line for line in lines if "Verdict:" in line), "Verdict: Unknown")
        verdict = verdict_line.split("Verdict:")[1].strip() if "Verdict:" in verdict_line else "Unknown"
        history_text.append(f"{i+1}. {match} → {verdict} ({time_str})")
        history_text.append("─" * 30)
    return '\n'.join(history_text)

# === PREDICT (USES CACHED VERSION) ===
def predict_with_ids(hid, aid, hname, aname, h_tla, a_tla):
    return cached_prediction(hid, aid, hname, aname, h_tla, a_tla)

# === IMPROVED LEAGUE FIXTURES WITH BETTER FORMATTING ===
def get_league_fixtures(league_name):
    lid = LEAGUE_MAP.get(league_name.lower())
    if not lid:
        return "League not supported."
    data = safe_get(f"{API_BASE}/competitions/{lid}/matches", {'status': 'SCHEDULED', 'limit': 10})
    if not data or not data.get('matches'):
        return "No upcoming fixtures found."
    fixtures = []
    for m in data['matches'][:5]:
        date = m['utcDate'][:10]
        time_str = m['utcDate'][11:16]
        home = m['homeTeam']['name']
        away = m['awayTeam']['name']
        hid = m['homeTeam']['id']
        aid = m['awayTeam']['id']
        pred = predict_with_ids(hid, aid, home, away, '', '')
        pred_lines = pred.splitlines()
        body = '\n'.join(pred_lines[2:]) if len(pred_lines) > 2 else pred
        
        fixtures.append(f"📅 *{date}* ⏰ *{time_str} UTC*")
        fixtures.append(f"**{home}** vs **{away}**")
        fixtures.append("─" * 40)
        fixtures.append(body)
        fixtures.append("═" * 50)
    
    return '\n'.join(fixtures)

# === FUN LOADING ANIMATIONS ===
def fun_loading(chat_id, base_text="Loading", reply_to_message_id=None, stages_count=3):
    stages = [
        "Loading data ⚙️",
        "Analyzing formations 🧠",
        "Crunching xG stats 📊",
        "Poisson digging 🔍",
        "Hold my beer 🍺",
        "Running Monte Carlo chaos 🎲",
        "Calibrating models 🤖",
        "Almost there… ⚡",
        "Finalizing predictions 🎲"
    ]
    random.shuffle(stages)
    try:
        if reply_to_message_id:
            msg = bot.send_message(chat_id, f"{base_text}...", reply_to_message_id=reply_to_message_id, parse_mode='Markdown')
        else:
            msg = bot.send_message(chat_id, f"{base_text}...", parse_mode='Markdown')
    except Exception:
        msg = bot.send_message(chat_id, f"{base_text}...", parse_mode='Markdown')
    for stage in stages[:stages_count]:
        time.sleep(random.uniform(0.9, 1.4))
        try:
            bot.edit_message_text(stage, chat_id, msg.message_id, parse_mode='Markdown')
        except Exception:
            pass
    return msg

# === NEW: LEAGUE SELECTION FOR STANDINGS ===
def show_standings_leagues(message):
    markup = types.InlineKeyboardMarkup(row_width=2)
    buttons = []
    leagues = [
        ("Premier League", "premier league"),
        ("La Liga", "la liga"),
        ("Bundesliga", "bundesliga"),
        ("Serie A", "serie a"),
        ("Ligue 1", "ligue 1"),
        ("Champions League", "champions"),
        ("Europa League", "europa"),
        ("Championship", "championship")
    ]
    
    for name, key in leagues:
        buttons.append(types.InlineKeyboardButton(name, callback_data=f"standings_{key}"))
    
    # Add buttons in rows of 2
    for i in range(0, len(buttons), 2):
        if i + 1 < len(buttons):
            markup.add(buttons[i], buttons[i+1])
        else:
            markup.add(buttons[i])
    
    markup.add(types.InlineKeyboardButton("⬅️ Back", callback_data="menu_2"))
    
    bot.send_message(
        message.chat.id,
        "📊 *Select League for Standings:*",
        reply_markup=markup,
        parse_mode='Markdown'
    )

# === NEW: LEAGUE SELECTION FOR TOP SCORERS ===
def show_scorers_leagues(message):
    markup = types.InlineKeyboardMarkup(row_width=2)
    buttons = []
    leagues = [
        ("Premier League", "premier league"),
        ("La Liga", "la liga"), 
        ("Bundesliga", "bundesliga"),
        ("Serie A", "serie a"),
        ("Ligue 1", "ligue 1"),
        ("Champions League", "champions"),
        ("Europa League", "europa")
    ]
    
    for name, key in leagues:
        buttons.append(types.InlineKeyboardButton(name, callback_data=f"scorers_{key}"))
    
    for i in range(0, len(buttons), 2):
        if i + 1 < len(buttons):
            markup.add(buttons[i], buttons[i+1])
        else:
            markup.add(buttons[i])
    
    markup.add(types.InlineKeyboardButton("⬅️ Back", callback_data="menu_2"))
    
    bot.send_message(
        message.chat.id,
        "⚽ *Select League for Top Scorers:*",
        reply_markup=markup,
        parse_mode='Markdown'
    )

# === STANDINGS COMMAND ===
@bot.message_handler(commands=['standings'])
def standings_command(m):
    show_standings_leagues(m)

# === TOP SCORERS COMMAND ===
@bot.message_handler(commands=['topscorers'])
def topscorers_command(m):
    show_scorers_leagues(m)

# === NEW: LIVE SCORES COMMAND ===
@bot.message_handler(commands=['live'])
def live_scores_command(m):
    loading = fun_loading(m.chat.id, "Checking live matches...", reply_to_message_id=m.message_id, stages_count=2)
    live_scores = get_live_scores()
    try:
        bot.edit_message_text(
            chat_id=m.chat.id,
            message_id=loading.message_id,
            text=live_scores,
            parse_mode='Markdown'
        )
    except Exception:
        bot.send_message(m.chat.id, live_scores, parse_mode='Markdown')

# === NEW: FPL COMMAND ===
@bot.message_handler(commands=['fpl'])
def fpl_command(m):
    loading = fun_loading(m.chat.id, "Fetching FPL data...", reply_to_message_id=m.message_id, stages_count=2)
    fpl_data = get_fpl_data()
    try:
        bot.edit_message_text(
            chat_id=m.chat.id,
            message_id=loading.message_id,
            text=fpl_data,
            parse_mode='Markdown'
        )
    except Exception:
        bot.send_message(m.chat.id, fpl_data, parse_mode='Markdown')

# === NEW: RESULTS COMMAND ===
@bot.message_handler(commands=['results'])
def results_command(m):
    loading = fun_loading(m.chat.id, "Checking today's results...", reply_to_message_id=m.message_id, stages_count=2)
    results = get_todays_results_with_comparison()
    try:
        bot.edit_message_text(
            chat_id=m.chat.id,
            message_id=loading.message_id,
            text=results,
            parse_mode='Markdown'
        )
    except Exception:
        bot.send_message(m.chat.id, results, parse_mode='Markdown')

# === MATCHDAY COMMAND ===
@bot.message_handler(commands=['matchday'])
def matchday_command(m):
    if len(m.text.split()) < 2:
        bot.reply_to(m,
            "Please specify a league. Examples:\n"
            "• `/matchday premier league`\n"
            "• `/matchday la liga`\n"
            "• `/matchday bundesliga`",
            parse_mode='Markdown'
        )
        return
    
    league_name = ' '.join(m.text.split()[1:])
    loading = fun_loading(m.chat.id, "Getting matchday...", reply_to_message_id=m.message_id, stages_count=2)
    matchday = get_current_matchday(league_name)
    try:
        bot.edit_message_text(
            chat_id=m.chat.id,
            message_id=loading.message_id,
            text=matchday,
            parse_mode='Markdown'
        )
    except Exception:
        bot.send_message(m.chat.id, matchday, parse_mode='Markdown')

# === ENHANCED /today WITH BETTER FORMATTING ===
def run_today(chat_id, reply_to_id=None):
    uid = chat_id
    if uid in LOADING_MSGS:
        return
    loading = fun_loading(chat_id, "Fetching today's fixtures", reply_to_message_id=reply_to_id, stages_count=3)
    LOADING_MSGS[uid] = loading.message_id
    try:
        today = date.today().isoformat()
        all_fixtures = []
        def fetch_and_predict(lid, name):
            data = safe_get(f"{API_BASE}/competitions/{lid}/matches", {'dateFrom': today, 'dateTo': today})
            if not data or not data.get('matches'): return [], 0
            results = []
            for m in data['matches'][:3]:
                hname = m['homeTeam']['name']
                aname = m['awayTeam']['name']
                hid = m['homeTeam']['id']
                aid = m['awayTeam']['id']
                t = m['utcDate'][11:16]
                pred = predict_with_ids(hid, aid, hname, aname, '', '')
                pred_lines = pred.splitlines()
                body = '\n'.join(pred_lines[2:]) if len(pred_lines) > 2 else pred
                results.append(f"📅 *{today}* ⏰ *{t} UTC*\n**{hname}** vs **{aname}**\n{body}\n{'═'*50}")
            return results, len(data['matches'])
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_and_predict, lid, name): name for name, lid in LEAGUE_MAP.items() if ' ' in name}
            for future in as_completed(futures):
                league_name = futures[future]
                try:
                    matches, total = future.result()
                    if matches:
                        all_fixtures.append(f"🏆 **{league_name.title()}** 🏆")
                        all_fixtures.extend(matches)
                        if total > 3:
                            all_fixtures.append(f"_+{total-3} more matches..._")
                        all_fixtures.append("")
                except: pass
        if not all_fixtures:
            result = "No fixtures today in major leagues."
        else:
            result = "*📅 TODAY'S FIXTURES & PREDICTIONS* 📅\n\n" + "\n".join(all_fixtures).strip()
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading.message_id,
            text=result,
            parse_mode='Markdown'
        )
    except Exception as e:
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading.message_id,
            text="Error loading fixtures.",
            parse_mode='Markdown'
        )
    finally:
        LOADING_MSGS.pop(uid, None)

# === ENHANCED /users ===
def run_users(chat_id, reply_to_id=None):
    uid = chat_id
    if uid in LOADING_MSGS:
        return
    loading_msg = bot.send_message(
        chat_id, "Compiling active users... 🔍", 
        reply_to_message_id=reply_to_id, 
        parse_mode='Markdown'
    )
    LOADING_MSGS[uid] = loading_msg.message_id
    try:
        time.sleep(random.uniform(1.2, 1.8))
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading_msg.message_id,
            text="Hold my beer 🍺",
            parse_mode='Markdown'
        )
        time.sleep(random.uniform(0.8, 1.3))
        active = len(USER_SESSIONS)
        total_predictions = sum(len(history) for history in USER_HISTORY.values())
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading_msg.message_id,
            text=(
                f"**Community Stats** 📊\n"
                f"═══════════════════════\n\n"
                f"👥 Active Users: `{active}`\n"
                f"📈 Predictions Made: `{total_predictions}`\n\n"
                f"⚽ **Now with enhanced features!**\n"
                f"• Live scores ⚡\n• FPL integration 🏆\n• Results comparison 📊\n• Better formatting 🎨"
            ),
            parse_mode='Markdown'
        )
    except Exception:
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading_msg.message_id,
            text="Error counting users.",
            parse_mode='Markdown'
        )
    finally:
        LOADING_MSGS.pop(uid, None)

# === ENHANCED /history ===
@bot.message_handler(commands=['history'])
def show_history(m):
    user_id = m.from_user.id
    history_text = get_user_history(user_id)
    predictions_made = len(USER_HISTORY.get(user_id, []))
    enhanced_text = f"{history_text}\n\n📊 Total Predictions: {predictions_made}"
    bot.reply_to(m, enhanced_text, parse_mode='Markdown')

# === ENHANCED START MENU ===
@bot.message_handler(commands=['start'])
def start(m):
    user_id = m.from_user.id
    USER_SESSIONS.add(user_id)
    
    show_menu_page(m, 1)

def show_menu_page(m, page=1):
    markup = types.InlineKeyboardMarkup(row_width=2)
    
    if page == 1:
        text = (
            f"⚽ *KickVision Football Predictions* ⚽\n\n"
            f"✨ *Advanced AI-powered match predictions*\n"
            f"🔮 *Proven statistical models*\n"
            f"🎯 *Professional betting insights*\n\n"
            f"*Page 1: Major Leagues*"
        )
        row1 = [
            types.InlineKeyboardButton("Premier League", callback_data="cmd_/premierleague"),
            types.InlineKeyboardButton("La Liga", callback_data="cmd_/laliga")
        ]
        row2 = [
            types.InlineKeyboardButton("Bundesliga", callback_data="cmd_/bundesliga"),
            types.InlineKeyboardButton("Serie A", callback_data="cmd_/seriea")
        ]
        row3 = [
            types.InlineKeyboardButton("Ligue 1", callback_data="cmd_/ligue1"),
            types.InlineKeyboardButton("Champions", callback_data="cmd_/champions")
        ]
        nav_row = [types.InlineKeyboardButton("Next ➡️", callback_data="menu_2")]
        markup.add(*row1, *row2, *row3, *nav_row)
    
    elif page == 2:
        text = (
            f"*KickVision Menu*\n\n"
            f"*Page 2: Quick Actions*"
        )
        row1 = [
            types.InlineKeyboardButton("Today", callback_data="cmd_/today"),
            types.InlineKeyboardButton("Users", callback_data="cmd_/users")
        ]
        row2 = [
            types.InlineKeyboardButton("History", callback_data="cmd_/history"),
            types.InlineKeyboardButton("Standings", callback_data="standings_menu")
        ]
        row3 = [
            types.InlineKeyboardButton("Top Scorers", callback_data="scorers_menu"),
            types.InlineKeyboardButton("Live Scores", callback_data="cmd_/live")
        ]
        row4 = [types.InlineKeyboardButton("Help", callback_data="help_1")]
        nav_row = [
            types.InlineKeyboardButton("Prev ⬅️", callback_data="menu_1"),
            types.InlineKeyboardButton("Next ➡️", callback_data="menu_3")
        ]
        markup.add(*row1, *row2, *row3, *row4, *nav_row)
    
    elif page == 3:
        text = (
            f"*KickVision Menu*\n\n"
            f"*Page 3: Advanced Features*"
        )
        row1 = [
            types.InlineKeyboardButton("FPL", callback_data="cmd_/fpl"),
            types.InlineKeyboardButton("Results", callback_data="cmd_/results")
        ]
        row2 = [
            types.InlineKeyboardButton("Matchday", callback_data="cmd_/matchday"),
            types.InlineKeyboardButton("Live Scores", callback_data="cmd_/live")
        ]
        row3 = [
            types.InlineKeyboardButton("Today", callback_data="cmd_/today"),
            types.InlineKeyboardButton("Users", callback_data="cmd_/users")
        ]
        nav_row = [
            types.InlineKeyboardButton("Prev ⬅️", callback_data="menu_2"),
            types.InlineKeyboardButton("Close", callback_data="menu_close")
        ]
        markup.add(*row1, *row2, *row3, *nav_row)
    
    bot.send_message(m.chat.id, text, reply_markup=markup, parse_mode='Markdown')

# === ENHANCED HELP PAGES ===
def build_help_page(page):
    markup = types.InlineKeyboardMarkup(row_width=3)
    prev_btn = types.InlineKeyboardButton("⬅️ Prev", callback_data=f"help_{max(1, page-1)}")
    next_btn = types.InlineKeyboardButton("Next ➡️", callback_data=f"help_{page+1}")
    close_btn = types.InlineKeyboardButton("Close", callback_data="menu_2")
    
    if page == 1:
        text = (
            "📃 *KickVision — Help (Page 1/3)*\n\n"
            "*Main Commands*\n"
            "• `/today` — Today's fixtures & predictions\n"
            "• `/premierleague` etc — League predictions\n"
            "• `Team A vs Team B` — Specific match prediction\n\n"
            "*New Features*\n"
            "• `/standings` — League table selector\n"
            "• `/topscorers` — Top goal scorers selector\n"
            "• `/live` — Live match scores\n\n"
            "_Tap Next for more._"
        )
        markup.add(next_btn, close_btn)
    elif page == 2:
        text = (
            "📃 *KickVision — Help (Page 2/3)*\n\n"
            "*Advanced Commands*\n"
            "• `/fpl` — Fantasy Premier League top performers\n"
            "• `/results` — Today's results vs predictions\n"
            "• `/matchday <league>` — Current matchday fixtures\n"
            "• `/history` — Your prediction history\n"
            "• `/users` — Community statistics\n\n"
            "*How It Works*\n"
            "• Uses advanced statistical models\n"
            "• Analyzes team form & xG data\n"
            "• Runs Monte Carlo simulations\n\n"
            "_Tap Next for tips._"
        )
        markup.add(prev_btn, next_btn, close_btn)
    elif page == 3:
        text = (
            "📃 *KickVision — Help (Page 3/3)*\n\n"
            "*Betting Tips* 💡\n"
            "• Never bet more than 5% of your bankroll\n"
            "• Look for value in underestimated teams\n"
            "• Stay disciplined - don't chase losses\n"
            "• Research team news and lineups\n\n"
            "*Supported Leagues*\n"
            "• Premier League, La Liga, Bundesliga\n"
            "• Serie A, Ligue 1, Champions League\n"
            "• And 8+ more major leagues!\n\n"
            "Enjoy! ⚽"
        )
        markup.add(prev_btn, close_btn)
    else:
        return build_help_page(1)
    return text, markup

def show_help_page(message, page=1):
    text, markup = build_help_page(page)
    try:
        bot.edit_message_text(chat_id=message.chat.id, message_id=message.message_id, text=text, reply_markup=markup, parse_mode='Markdown')
    except Exception:
        bot.send_message(message.chat.id, text, reply_markup=markup, parse_mode='Markdown')

@bot.message_handler(commands=['help'])
def run_help_cmd(message):
    show_help_page(message, 1)

# === ENHANCED CALLBACK HANDLER ===
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    chat_id = call.message.chat.id
    reply_to_id = call.message.message_id
    data = call.data
    bot.answer_callback_query(call.id)

    if data.startswith("cmd_/"):
        cmd = data[4:]
        if cmd == "/today":
            run_today(chat_id, reply_to_id)
        elif cmd == "/users":
            run_users(chat_id, reply_to_id)
        elif cmd == "/history":
            user_id = call.from_user.id
            history_text = get_user_history(user_id)
            bot.send_message(chat_id, history_text, parse_mode='Markdown')
        elif cmd == "/live":
            loading = fun_loading(chat_id, "Checking live matches...", reply_to_message_id=reply_to_id, stages_count=2)
            live_scores = get_live_scores()
            try:
                bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=loading.message_id,
                    text=live_scores,
                    parse_mode='Markdown'
                )
            except Exception:
                bot.send_message(chat_id, live_scores, parse_mode='Markdown')
        elif cmd == "/fpl":
            loading = fun_loading(chat_id, "Fetching FPL data...", reply_to_message_id=reply_to_id, stages_count=2)
            fpl_data = get_fpl_data()
            try:
                bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=loading.message_id,
                    text=fpl_data,
                    parse_mode='Markdown'
                )
            except Exception:
                bot.send_message(chat_id, fpl_data, parse_mode='Markdown')
        elif cmd == "/results":
            loading = fun_loading(chat_id, "Checking today's results...", reply_to_message_id=reply_to_id, stages_count=2)
            results = get_todays_results_with_comparison()
            try:
                bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=loading.message_id,
                    text=results,
                    parse_mode='Markdown'
                )
            except Exception:
                bot.send_message(chat_id, results, parse_mode='Markdown')
        else:
            real_msg = types.Message(
                message_id=call.message.message_id,
                from_user=call.from_user,
                date=datetime.now(),
                chat=call.message.chat,
                content_type='text',
                options=[],
                json_string=None
            )
            real_msg.text = cmd
            dynamic_league_handler(real_msg)

    elif data.startswith("standings_"):
        if data == "standings_menu":
            show_standings_leagues(call.message)
        else:
            league_name = data[10:]
            loading = fun_loading(chat_id, "Fetching standings...", reply_to_message_id=reply_to_id, stages_count=2)
            standings = get_league_standings(league_name)
            try:
                bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=loading.message_id,
                    text=standings,
                    parse_mode='Markdown'
                )
            except Exception:
                bot.send_message(chat_id, standings, parse_mode='Markdown')
    
    elif data.startswith("scorers_"):
        if data == "scorers_menu":
            show_scorers_leagues(call.message)
        else:
            league_name = data[8:]
            loading = fun_loading(chat_id, "Fetching top scorers...", reply_to_message_id=reply_to_id, stages_count=2)
            scorers = get_top_scorers(league_name)
            try:
                bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=loading.message_id,
                    text=scorers,
                    parse_mode='Markdown'
                )
            except Exception:
                bot.send_message(chat_id, scorers, parse_mode='Markdown')
    
    elif data.startswith("menu_"):
        if data == "menu_close":
            try:
                bot.delete_message(chat_id, call.message.message_id)
            except:
                pass
        else:
            page = int(data.split("_")[1])
            show_menu_page(call.message, page)
    
    elif data.startswith("help_"):
        page = int(data.split("_")[1])
        show_help_page(call.message, page)

# === DYNAMIC LEAGUE HANDLER ===
@bot.message_handler(func=lambda m: any(m.text and (m.text.lower().startswith(f"/{k.replace(' ', '')}") or m.text.lower() == k) for k in LEAGUE_MAP))
def dynamic_league_handler(m):
    if not m.text: return
    txt = m.text.strip().lower()
    if txt.startswith('/'):
        txt = txt[1:]
    matched = next((k for k in LEAGUE_MAP if txt == k.replace(' ', '') or txt == k), None)
    if not matched:
        return
    display_name = matched.title() if ' ' in matched else matched.upper()
    reply_id = m.message_id if hasattr(m, 'message_id') else None
    loading = fun_loading(m.chat.id, "Loading fixtures...", reply_to_message_id=reply_id, stages_count=3)
    fixtures = get_league_fixtures(matched)
    try:
        bot.edit_message_text(
            chat_id=m.chat.id,
            message_id=loading.message_id,
            text=f"*{display_name} Upcoming*\n\n{fixtures}" if fixtures else "No fixtures.",
            parse_mode='Markdown'
        )
    except Exception:
        bot.send_message(m.chat.id, f"*{display_name} Upcoming*\n\n{fixtures}" if fixtures else "No fixtures.", parse_mode='Markdown')

# === RATE LIMIT ===
def is_allowed(uid):
    now = time.time()
    user_rate[uid] = [t for t in user_rate[uid] if now - t < 5]
    if len(user_rate[uid]) >= 3: return False
    user_rate[uid].append(now)
    return True

# === ENHANCED MAIN HANDLER ===
@bot.message_handler(func=lambda m: True)
def handle(m):
    if not m.text: return
    uid = m.from_user.id
    txt = m.text.strip()

    USER_SESSIONS.add(uid)

    if txt.strip().lower() == '/cancel':
        if uid in PENDING_MATCH:
            del PENDING_MATCH[uid]
            bot.reply_to(m, "Cancelled.")
        return

    if uid in PENDING_MATCH:
        parts = txt.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            h_choice = int(parts[0])
            a_choice = int(parts[1])
            home_input, away_input, home_opts, away_opts = PENDING_MATCH[uid]
            if 1 <= h_choice <= len(home_opts) and 1 <= a_choice <= len(away_opts):
                h = home_opts[h_choice-1]
                a = away_opts[a_choice-1]
                loading = fun_loading(m.chat.id, "Predicting...", reply_to_message_id=m.message_id, stages_count=3)
                r = predict_with_ids(h[2], a[2], h[1], a[1], h[3], a[3])
                add_to_history(uid, f"{h[1]} vs {a[1]}", r)
                try:
                    bot.edit_message_text(
                        chat_id=m.chat.id,
                        message_id=loading.message_id,
                        text=r,
                        parse_mode='Markdown'
                    )
                except Exception:
                    bot.send_message(m.chat.id, r, parse_mode='Markdown')
                del PENDING_MATCH[uid]
            else:
                bot.reply_to(m, "Invalid. Try `1 2` or /cancel")
        else:
            bot.reply_to(m, "Reply with two numbers: `1 3`")
        return

    if not is_allowed(uid):
        bot.reply_to(m, "Wait 5s...")
        return

    # Quick searching animation
    searching_msg = bot.reply_to(m, "Checking 🔍 ...", parse_mode='Markdown')
    for _ in range(4):
        time.sleep(0.55)
        icon = "🔍" if _ % 2 == 0 else "🔎"
        try:
            bot.edit_message_text(
                chat_id=m.chat.id,
                message_id=searching_msg.message_id,
                text=f"Checking {icon} ...",
                parse_mode='Markdown'
            )
        except Exception:
            pass

    # Team vs team logic
    txt = re.sub(r'[|\[\](){}]', ' ', txt)
    if not re.search(r'\s+vs\s+|\s+[-–—]\s+', txt, re.IGNORECASE):
        try:
            bot.delete_message(m.chat.id, searching_msg.message_id)
        except Exception:
            pass
        return

    parts = re.split(r'\s+vs\s+|\s+[-–—]\s+', txt, re.IGNORECASE)
    home = parts[0].strip()
    away = ' '.join(parts[1:]).strip()

    home_cands = find_team_candidates(home)
    away_cands = find_team_candidates(away)

    try:
        bot.delete_message(m.chat.id, searching_msg.message_id)
    except Exception:
        pass

    if not home_cands or not away_cands:
        bot.reply_to(m, f"*{home} vs {away}*\n\n_Not found._", parse_mode='Markdown')
        return

    if home_cands[0][0] > 0.9 and away_cands[0][0] > 0.9:
        h = home_cands[0]; a = away_cands[0]
        loading = fun_loading(m.chat.id, "Predicting...", reply_to_message_id=m.message_id, stages_count=3)
        r = predict_with_ids(h[2], a[2], h[1], a[1], h[3], a[3])
        add_to_history(uid, f"{h[1]} vs {a[1]}", r)
        try:
            bot.edit_message_text(
                chat_id=m.chat.id,
                message_id=loading.message_id,
                text=r,
                parse_mode='Markdown'
            )
        except Exception:
            bot.send_message(m.chat.id, r, parse_mode='Markdown')
        return

    msg = [f"*Did you mean?*"]
    msg.append(f"**Home:** {home}")
    for i, (_, name, _, tla, _, lname) in enumerate(home_cands, 1):
        msg.append(f"`{i}.` {name} `({tla})` — _{lname}_")
    msg.append(f"**Away:** {away}")
    for i, (_, name, _, tla, _, lname) in enumerate(away_cands, 1):
        msg.append(f"`{i}.` {name} `({tla})` — _{lname}_")
    msg.append("\nReply with two numbers: `1 3`")
    bot.reply_to(m, '\n'.join(msg), parse_mode='Markdown')
    PENDING_MATCH[uid] = (home, away, home_cands, away_cands)

# === FLASK WEBHOOK ===
app = Flask(__name__)

@app.route(f'/{BOT_TOKEN}', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        update = telebot.types.Update.de_json(request.get_data().decode('utf-8'))
        bot.process_new_updates([update])
        return 'OK', 200
    return 'Invalid', 403

@app.route('/')
def index():
    return 'KickVision Bot v1.3.0 - Enhanced Free Edition is running!'

if __name__ == '__main__':
    log.info("KickVision v1.3.0 — ENHANCED FREE EDITION READY")
    bot.remove_webhook()
    time.sleep(1)
    bot.set_webhook(url=f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/{BOT_TOKEN}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
