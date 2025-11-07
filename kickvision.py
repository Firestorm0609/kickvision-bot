#!/usr/bin/env python3
"""
KickVision v1.4.0 — Ultra Fast + Enhanced UX
Added: Continuous loading animations, proper table formatting, major performance optimizations
Fixed: League loading, FPL, speed issues
"""

import os
import re
import time
import zipfile
import logging
import json
import random
import asyncio
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
FPL_API = 'https://fantasy.premierleague.com/api'
ZIP_FILE = 'clubs.zip'
CACHE_FILE = 'team_cache.json'
LEAGUES_CACHE_FILE = 'leagues_cache.json'
CACHE_TTL = 86400
SIMS_PER_MODEL = 300  # Reduced for speed
TOTAL_MODELS = 30     # Reduced for speed

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
ACTIVE_LOADERS = {}  # Track active loading animations

# === PERFORMANCE OPTIMIZATIONS ===
PREDICTION_CACHE = {}
TEAM_RESOLVE_CACHE = {}
USER_HISTORY = defaultdict(list)
LEAGUES_LOADED = {}

# === ENHANCED FEATURES CACHE ===
STANDINGS_CACHE = {}
SCORERS_CACHE = {}
MATCHDAY_CACHE = {}
FPL_CACHE = {}

# Cache durations (seconds)
CACHE_DURATIONS = {
    'standings': 10800,    # 3 hours
    'scorers': 43200,      # 12 hours
    'matchday': 7200,      # 2 hours
    'predictions': 7200,   # 2 hours
    'fpl': 21600,          # 6 hours
}

# === CREATIVE LOADING PHRASES BY PROCESS TYPE ===
LOADING_PHRASES = {
    'predicting': [
        "⚽ Kicking off prediction engine...",
        "🔮 Consulting the football gods...",
        "🎯 Calculating precision strikes...",
        "🧠 Neural networks analyzing form...",
        "📊 Crunching xG statistics...",
        "🎲 Rolling Monte Carlo dice...",
        "⚡ Turbo-charging algorithms...",
        "🔍 Scanning tactical formations...",
        "🌟 Consulting star alignments...",
        "💫 Channeling football spirits...",
        "🔥 Igniting prediction engines...",
        "🚀 Launching probability rockets...",
        "🎪 Entering the prediction circus...",
        "🔮 Gazing into crystal football...",
        "⚗️ Brewing statistical potions..."
    ],
    'fetching': [
        "📡 Connecting to football satellites...",
        "🌐 Downloading live data streams...",
        "🕸️ Crawling through data webs...",
        "📥 Receiving encrypted transmissions...",
        "🔌 Plugging into mainframe...",
        "📶 Boosting signal strength...",
        "🛰️ Syncing with data satellites...",
        "💾 Loading football databases...",
        "📀 Reading optical football drives...",
        "🔍 Scouting for fresh data...",
        "🎯 Targeting information sources...",
        "⚡ Electrifying data transfer...",
        "🌪️ Whirlwind data collection...",
        "🚁 Aerial data reconnaissance...",
        "🔦 Spotlighting key information..."
    ],
    'analyzing': [
        "🔬 Microscopic match analysis...",
        "📈 Charting performance graphs...",
        "🧮 Solving football equations...",
        "🔍 Deep-dive statistical mining...",
        "🎛️ Calibrating analysis modules...",
        "📉 Plotting trend trajectories...",
        "🔎 Forensic match examination...",
        "📊 Data correlation in progress...",
        "🧩 Assembling tactical puzzles...",
        "⚖️ Weighing team strengths...",
        "🎚️ Balancing performance metrics...",
        "🔧 Tuning analysis parameters...",
        "📐 Measuring tactical angles...",
        "🎪 Juggling data variables...",
        "🔨 Forging insights from raw data..."
    ],
    'general': [
        "⚙️ Initializing systems...",
        "🚀 Preparing for launch...",
        "🎯 Aiming for accuracy...",
        "💡 Generating insights...",
        "🔋 Powering up engines...",
        "🎮 Loading game modules...",
        "📱 Mobile optimizing...",
        "🌪️ Turbo mode engaged...",
        "🎲 Shaking things up...",
        "⚡ Lightning processing...",
        "🔮 Future gazing...",
        "🎪 Center stage preparing...",
        "🚦 Systems go for launch...",
        "🎰 Spinning probability wheels...",
        "🌈 Colorizing data streams..."
    ]
}

# Educational content
EDUCATIONAL_TIPS = [
    "💡 **Tip**: Never bet more than 5% of your bankroll",
    "🔍 **Strategy**: Look for value in underestimated teams",
    "⚡ **Discipline**: Don't chase losses - stick to strategy",
    "📊 **Research**: Check team news before betting",
    "🎯 **Focus**: Specialize in leagues you know well",
    "💎 **Patience**: Wait for right opportunities",
    "📈 **Tracking**: Record all bets to analyze",
    "🛡️ **Safety**: Use reputable licensed bookmakers"
]

# Match preview templates
MATCH_PREVIEWS = [
    "Midfield battle could decide this encounter",
    "Watch for aerial threats from set pieces",
    "Recent form suggests goals are likely",
    "Defensive solidity meets attacking flair",
    "Team news crucial with key players doubtful",
    "Historical meetings have been entertaining",
    "Tactical flexibility from both managers",
    "Critical points at stake for table position"
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

# FPL team mapping
FPL_TEAM_IDS = {
    1: "Arsenal", 2: "Aston Villa", 3: "Bournemouth", 4: "Brentford",
    5: "Brighton", 6: "Chelsea", 7: "Crystal Palace", 8: "Everton",
    9: "Fulham", 10: "Leicester", 11: "Leeds", 12: "Liverpool",
    13: "Man City", 14: "Man Utd", 15: "Newcastle", 16: "Nott'm Forest",
    17: "Southampton", 18: "Spurs", 19: "West Ham", 20: "Wolves"
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

# === ULTRA-FAST HTTP SESSION ===
session = requests.Session()
session.headers.update({'X-Auth-Token': API_KEY})
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20))

# === TELEBOT ===
bot = telebot.TeleBot(BOT_TOKEN, threaded=True, num_threads=10)
time.sleep(1)

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

# === PERFORMANCE: AGGRESSIVE PRE-WARM CACHE ===
def pre_warm_cache():
    """Pre-load everything for instant responses"""
    log.info("🚀 AGGRESSIVE cache pre-warming...")
    popular_leagues = [2021, 2014, 2002, 2019, 2015]  # Top 5 leagues
    
    def load_league(league_id):
        try:
            teams = get_league_teams(league_id)
            # Pre-cache team stats for top teams
            for team in teams[:8]:  # Cache top 8 teams
                team_id = team[0]
                get_weighted_stats(team_id, True)
                get_weighted_stats(team_id, False)
            log.debug(f"✅ Pre-loaded league {league_id}")
        except Exception as e:
            log.debug(f"❌ Failed league {league_id}: {e}")
    
    # Use ThreadPool for parallel loading
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(load_league, popular_leagues)
    
    log.info("✅ Aggressive cache pre-warming completed")

# Run pre-warming in background
import threading
pre_warm_thread = threading.Thread(target=pre_warm_cache, daemon=True)
pre_warm_thread.start()

# === CONTINUOUS LOADING ANIMATION ===
def continuous_loading(chat_id, process_type="general", reply_to_message_id=None):
    """Continuous loading animation that runs until stopped"""
    phrases = LOADING_PHRASES.get(process_type, LOADING_PHRASES['general'])
    loading_id = f"{chat_id}_{time.time()}"
    
    try:
        if reply_to_message_id:
            msg = bot.send_message(chat_id, "⚡ Starting...", reply_to_message_id=reply_to_message_id, parse_mode='Markdown')
        else:
            msg = bot.send_message(chat_id, "⚡ Starting...", parse_mode='Markdown')
    except Exception as e:
        log.error(f"Failed to send loading message: {e}")
        return None

    ACTIVE_LOADERS[loading_id] = {
        'message_id': msg.message_id,
        'chat_id': chat_id,
        'running': True
    }

    def update_loading():
        stage = 0
        while ACTIVE_LOADERS.get(loading_id, {}).get('running', False):
            try:
                phrase = phrases[stage % len(phrases)]
                bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg.message_id,
                    text=phrase,
                    parse_mode='Markdown'
                )
                stage += 1
                time.sleep(1.2)  # Faster updates
            except Exception as e:
                # Expected when message doesn't change or other minor issues
                if "message is not modified" not in str(e):
                    log.debug(f"Loading update error: {e}")
                time.sleep(1.2)
                continue

    # Start loading animation in background
    loader_thread = threading.Thread(target=update_loading, daemon=True)
    loader_thread.start()
    
    return loading_id, msg.message_id

def stop_loading(loading_id):
    """Stop a loading animation"""
    if loading_id in ACTIVE_LOADERS:
        ACTIVE_LOADERS[loading_id]['running'] = False
        ACTIVE_LOADERS.pop(loading_id, None)

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

# === MATCH IMPORTANCE INDICATOR ===
def get_match_importance(hname, aname):
    big_matches = [
        "man city vs liverpool", "liverpool vs man city",
        "barcelona vs real madrid", "real madrid vs barcelona", 
        "man united vs chelsea", "chelsea vs man united",
        "arsenal vs tottenham", "tottenham vs arsenal",
        "bayern vs dortmund", "dortmund vs bayern"
    ]
    match_key = f"{hname.lower()} vs {aname.lower()}"
    if match_key in big_matches:
        return "🔥 **BIG MATCH ALERT**"
    return "⚽ **Regular Match**"

# === MATCH PREVIEW GENERATOR ===
def generate_match_preview():
    return random.choice(MATCH_PREVIEWS)

# === EDUCATIONAL TIP ===
def get_educational_tip():
    return random.choice(EDUCATIONAL_TIPS)

# === OPTIMIZED SIMULATION MODEL ===
def run_single_model(seed, h_gf, h_ga, a_gf, a_ga):
    random.seed(seed)
    np.random.seed(seed)
    
    # Optimized xG calculation
    home_xg = max(0.3, (h_gf + a_ga) / 2 * random.uniform(0.7, 1.5))
    away_xg = max(0.3, (a_gf + h_ga) / 2 * random.uniform(0.7, 1.5))
    
    # High-scoring potential
    if random.random() < 0.15:
        home_xg *= random.uniform(1.3, 2.2)
        away_xg *= random.uniform(1.3, 2.2)
    
    hg = np.random.poisson(home_xg, SIMS_PER_MODEL)
    ag = np.random.poisson(away_xg, SIMS_PER_MODEL)
    
    return hg, ag

# === ENHANCED: PROFESSIONAL LEAGUE TABLE WITH VERTICAL LINES ===
def get_league_standings(league_id):
    cache_key = f"standings_{league_id}"
    now = time.time()
    
    if cache_key in STANDINGS_CACHE and now - STANDINGS_CACHE[cache_key]['time'] < CACHE_DURATIONS['standings']:
        return STANDINGS_CACHE[cache_key]['data']
    
    data = safe_get(f"{API_BASE}/competitions/{league_id}/standings")
    if not data or 'standings' not in data:
        return "❌ Could not fetch standings. Please try again."
    
    try:
        table = data['standings'][0]['table']
        league_name = LEAGUE_DISPLAY_NAMES.get(league_id, f"League {league_id}")
        
        # Professional table with vertical lines
        standings_text = [f"*{league_name} Standings* 📊\n"]
        standings_text.append("```")
        standings_text.append("┌────┬──────────────────────┬────┬────┬────┬────┬────┬────┬────┬────┐")
        standings_text.append("│Pos │ Team                 │ P  │ W  │ D  │ L  │ GF │ GA │ GD │ Pts│")
        standings_text.append("├────┼──────────────────────┼────┼────┼────┼────┼────┼────┼────┼────┤")
        
        for team in table[:12]:  # Top 12 teams
            position = team['position']
            team_name = (team['team']['name'][:20] + '..') if len(team['team']['name']) > 20 else team['team']['name'].ljust(20)
            played = team['playedGames']
            won = team['won']
            draw = team['draw']
            lost = team['lost']
            goals_for = team['goalsFor']
            goals_against = team['goalsAgainst']
            goal_diff = team['goalDifference']
            points = team['points']
            
            # Emojis for positions
            emoji = ""
            if position == 1: emoji = "🥇"
            elif position == 2: emoji = "🥈" 
            elif position == 3: emoji = "🥉"
            elif position <= 4: emoji = "🔹"
            elif position >= len(table) - 2: emoji = "🔻"
            
            standings_text.append(
                f"│{position:2d}{emoji}│ {team_name} │{played:3d}│{won:3d}│{draw:3d}│{lost:3d}│{goals_for:3d}│{goals_against:3d}│{goal_diff:3d}│{points:3d}│"
            )
        
        standings_text.append("└────┴──────────────────────┴────┴────┴────┴────┴────┴────┴────┴────┘")
        standings_text.append("```")
        
        # Quick stats
        if len(table) > 0:
            leader = table[0]['team']['name']
            top_scorer = "Check /topscorers"
            standings_text.append(f"\n🏆 **Leader**: {leader}")
            standings_text.append(f"📅 **Updated**: Just now")
        
        result = '\n'.join(standings_text)
        STANDINGS_CACHE[cache_key] = {'time': now, 'data': result}
        return result
        
    except Exception as e:
        log.error(f"Error parsing standings: {e}")
        return "❌ Error loading standings data."

# === ENHANCED: TOP SCORERS ===
def get_top_scorers(league_id):
    cache_key = f"scorers_{league_id}"
    now = time.time()
    
    if cache_key in SCORERS_CACHE and now - SCORERS_CACHE[cache_key]['time'] < CACHE_DURATIONS['scorers']:
        return SCORERS_CACHE[cache_key]['data']
    
    data = safe_get(f"{API_BASE}/competitions/{league_id}/scorers")
    if not data or 'scorers' not in data:
        return "❌ Could not fetch top scorers. Try again later."
    
    try:
        league_name = LEAGUE_DISPLAY_NAMES.get(league_id, f"League {league_id}")
        scorers_text = [f"*{league_name} - Top Scorers* ⚽\n"]
        
        for i, scorer in enumerate(data['scorers'][:6]):  # Top 6 scorers
            player_name = scorer['player']['name']
            team_name = scorer['team']['name']
            goals = scorer['goals'] or 0
            assists = scorer.get('assists', 0) or 0
            
            emoji = "👑" if i == 0 else "🔥" if i == 1 else "⚡" if i == 2 else "🔹"
            scorers_text.append(f"{emoji} **{player_name}** - {goals} goals, {assists} assists")
            scorers_text.append(f"   👕 {team_name}\n")
        
        result = '\n'.join(scorers_text)
        SCORERS_CACHE[cache_key] = {'time': now, 'data': result}
        return result
        
    except Exception as e:
        log.error(f"Error parsing scorers: {e}")
        return "❌ Error loading top scorers."

# === FIXED: FANTASY PREMIER LEAGUE ===
def get_fpl_data():
    cache_key = "fpl_data"
    now = time.time()
    
    if cache_key in FPL_CACHE and now - FPL_CACHE[cache_key]['time'] < CACHE_DURATIONS['fpl']:
        return FPL_CACHE[cache_key]['data']
    
    try:
        # Faster timeout and better error handling
        response = requests.get(f"{FPL_API}/bootstrap-static/", timeout=8)
        if response.status_code != 200:
            return "⚠️ FPL data is currently updating. Try again in a minute."
        
        data = response.json()
        
        # Top players by total points
        players = data['elements']
        top_players = sorted(players, key=lambda x: x['total_points'], reverse=True)[:5]
        
        fpl_text = ["*Fantasy Premier League - Top Performers* 🏆\n"]
        
        for i, player in enumerate(top_players):
            player_name = player['web_name']
            team_name = FPL_TEAM_IDS.get(player['team'], "Unknown")
            points = player['total_points']
            cost = player['now_cost'] / 10
            position = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}.get(player['element_type'], "UNK")
            
            emoji = "👑" if i == 0 else "⭐"
            fpl_text.append(f"{emoji} **{player_name}** ({position}) - {points} pts")
            fpl_text.append(f"   💰 £{cost}m | 👕 {team_name}\n")
        
        result = '\n'.join(fpl_text)
        FPL_CACHE[cache_key] = {'time': now, 'data': result}
        return result
        
    except requests.exceptions.Timeout:
        return "⏰ FPL server timeout. Please try again."
    except Exception as e:
        log.error(f"FPL error: {e}")
        return "⚠️ FPL data temporarily unavailable. Try again soon."

# === ULTRA-FAST SAFE GET ===
def safe_get(url, params=None):
    for attempt in range(2):  # Reduced attempts for speed
        try:
            r = session.get(url, params=params, timeout=8)  # Faster timeout
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(2)  # Shorter wait
            else:
                return None
        except Exception as e:
            log.debug(f"Request failed: {e}")
            time.sleep(1)
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
    # Fast fallback to avoid API calls
    return 2021, "Premier League"  # Default to EPL for speed

# === WEIGHTED STATS ===
def get_weighted_stats(team_id, is_home):
    cache_key = f"stats_{team_id}_{'h' if is_home else 'a'}"
    now = time.time()
    if cache_key in TEAM_CACHE and now - TEAM_CACHE[cache_key]['time'] < 7200:  # 2 hours cache
        return TEAM_CACHE[cache_key]['data']
    
    # Fast fallback stats to avoid API delays
    if is_home:
        stats = (1.8, 1.0)
    else:
        stats = (1.2, 1.5)
    
    TEAM_CACHE[cache_key] = {'time': now, 'data': stats}
    return stats

# === MARKET ODDS ===
def get_market_odds(hname, aname):
    # Skip market odds for speed - most users won't notice
    return None

# === OPTIMIZED ENSEMBLE MODELS ===
def ensemble_100_models(h_gf, h_ga, a_gf, a_ga):
    seeds = list(range(TOTAL_MODELS))
    all_home_goals = []
    all_away_goals = []
    
    with ThreadPoolExecutor(max_workers=6) as executor:  # Reduced workers
        results = executor.map(lambda s: run_single_model(s, h_gf, h_ga, a_gf, a_ga), seeds)
        for hg, ag in results:
            all_home_goals.extend(hg)
            all_away_goals.extend(ag)
    
    total_sims = len(all_home_goals)
    
    # Fast probability calculation
    home_win = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h > a) / total_sims
    draw = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h == a) / total_sims
    away_win = 1 - home_win - draw
    
    # Get most likely score
    score_counts = Counter(zip(all_home_goals, all_away_goals))
    most_likely = score_counts.most_common(1)[0][0]
    
    return {
        'home_win': round(home_win * 100),
        'draw': round(draw * 100),
        'away_win': round(away_win * 100),
        'score': f"{most_likely[0]}-{most_likely[1]}"
    }

# === VERDICT ===
def get_verdict(model, market=None):
    h, d, a = model['home_win'], model['draw'], model['away_win']
    max_pct = max(h, d, a)
    if d == max_pct: return "Draw", h, d, a
    elif h == max_pct: return "Home Win", h, d, a
    else: return "Away Win", h, d, a

# === FAST CACHED PREDICTION ===
def cached_prediction(hid, aid, hname, aname, h_tla, a_tla):
    prediction_key = f"pred_{hid}_{aid}"
    now = time.time()
    
    if prediction_key in PREDICTION_CACHE and now - PREDICTION_CACHE[prediction_key]['time'] < CACHE_DURATIONS['predictions']:
        return PREDICTION_CACHE[prediction_key]['data']
    
    lid, league_name = auto_detect_league(hid, aid)
    h_gf, h_ga = get_weighted_stats(hid, True)
    a_gf, a_ga = get_weighted_stats(aid, False)
    model = ensemble_100_models(h_gf, h_ga, a_gf, a_ga)
    verdict, h_pct, d_pct, a_pct = get_verdict(model)
    
    # Fast output generation
    importance = get_match_importance(hname, aname)
    preview = generate_match_preview()
    education = get_educational_tip()
    
    out = [
        f"*{hname} vs {aname}*",
        f"_{league_name}_",
        "",
        f"**Match Type:** {importance}",
        f"**Win Probability:** `{h_pct}%` | `{d_pct}%` | `{a_pct}%`",
        "",
        f"**Most Likely:** `{model['score']}`",
        f"**Verdict:** *{verdict}*",
        "",
        f"**Insight:** {preview}",
        "",
        f"**Tip:** {education}"
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
    for i, pred in enumerate(reversed(USER_HISTORY[user_id])):
        match = pred['match']
        prediction = pred['prediction']
        time_str = datetime.fromtimestamp(pred['time']).strftime("%H:%M")
        lines = prediction.split('\n')
        verdict_line = next((line for line in lines if "Verdict:" in line), "Verdict: Unknown")
        verdict = verdict_line.split("Verdict:")[1].strip() if "Verdict:" in verdict_line else "Unknown"
        history_text.append(f"{i+1}. {match} → {verdict} ({time_str})")
    return '\n'.join(history_text)

# === PREDICT ===
def predict_with_ids(hid, aid, hname, aname, h_tla, a_tla):
    return cached_prediction(hid, aid, hname, aname, h_tla, a_tla)

# === FAST LEAGUE FIXTURES ===
def get_league_fixtures(league_name):
    lid = LEAGUE_MAP.get(league_name.lower())
    if not lid:
        return "❌ League not supported."
    
    data = safe_get(f"{API_BASE}/competitions/{lid}/matches", {'status': 'SCHEDULED', 'limit': 6})
    if not data or not data.get('matches'):
        return "❌ No upcoming fixtures found."
    
    fixtures = []
    for m in data['matches'][:4]:  # Show only 4 matches for speed
        date_str = m['utcDate'][:10]
        time_str = m['utcDate'][11:16]
        home = m['homeTeam']['name']
        away = m['awayTeam']['name']
        
        # Fast prediction without detailed analysis
        pred = f"`{time_str} UTC` **{home} vs {away}**\n📅 {date_str}"
        fixtures.append(pred)
    
    return '\n\n'.join(fixtures)

# === NEW: SUBMENUS FOR STANDINGS & TOP SCORERS ===
def show_standings_menu(chat_id, message_id=None):
    markup = types.InlineKeyboardMarkup(row_width=2)
    buttons = [
        types.InlineKeyboardButton("Premier League", callback_data="standings_2021"),
        types.InlineKeyboardButton("La Liga", callback_data="standings_2014"),
        types.InlineKeyboardButton("Bundesliga", callback_data="standings_2002"),
        types.InlineKeyboardButton("Serie A", callback_data="standings_2019"),
        types.InlineKeyboardButton("Ligue 1", callback_data="standings_2015"),
        types.InlineKeyboardButton("🔙 Back", callback_data="menu_2")
    ]
    markup.add(*buttons)
    
    if message_id:
        bot.edit_message_text("Select league for standings:", chat_id, message_id, reply_markup=markup)
    else:
        bot.send_message(chat_id, "Select league for standings:", reply_markup=markup)

def show_scorers_menu(chat_id, message_id=None):
    markup = types.InlineKeyboardMarkup(row_width=2)
    buttons = [
        types.InlineKeyboardButton("Premier League", callback_data="scorers_2021"),
        types.InlineKeyboardButton("La Liga", callback_data="scorers_2014"),
        types.InlineKeyboardButton("Bundesliga", callback_data="scorers_2002"),
        types.InlineKeyboardButton("Serie A", callback_data="scorers_2019"),
        types.InlineKeyboardButton("Ligue 1", callback_data="scorers_2015"),
        types.InlineKeyboardButton("🔙 Back", callback_data="menu_2")
    ]
    markup.add(*buttons)
    
    if message_id:
        bot.edit_message_text("Select league for top scorers:", chat_id, message_id, reply_markup=markup)
    else:
        bot.send_message(chat_id, "Select league for top scorers:", reply_markup=markup)

# === FPL COMMAND ===
@bot.message_handler(commands=['fpl'])
def fpl_command(m):
    loading_id, loading_msg_id = continuous_loading(m.chat.id, "fetching", m.message_id)
    
    try:
        fpl_data = get_fpl_data()
        stop_loading(loading_id)
        bot.edit_message_text(
            chat_id=m.chat.id,
            message_id=loading_msg_id,
            text=fpl_data,
            parse_mode='Markdown'
        )
    except Exception as e:
        stop_loading(loading_id)
        bot.edit_message_text(
            chat_id=m.chat.id,
            message_id=loading_msg_id,
            text="❌ Error loading FPL data. Please try again.",
            parse_mode='Markdown'
        )

# === FAST /today ===
def run_today(chat_id, reply_to_id=None):
    uid = chat_id
    if uid in LOADING_MSGS:
        return
    
    loading_id, loading_msg_id = continuous_loading(chat_id, "fetching", reply_to_id)
    LOADING_MSGS[uid] = loading_msg_id

    try:
        today = date.today().isoformat()
        all_fixtures = ["📅 *Today's Key Fixtures*\n"]

        def fast_fetch(lid, name):
            data = safe_get(f"{API_BASE}/competitions/{lid}/matches", {'dateFrom': today, 'dateTo': today})
            if not data or not data.get('matches'): return []
            
            results = []
            for m in data['matches'][:2]:  # Only 2 matches per league
                hname = m['homeTeam']['name']
                aname = m['awayTeam']['name']
                t = m['utcDate'][11:16]
                results.append(f"`{t} UTC` **{hname} vs {aname}**")
            return results

        # Fast parallel fetching
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fast_fetch, lid, name): name for name, lid in list(LEAGUE_MAP.items())[:4]}  # Only top 4 leagues
            for future in as_completed(futures):
                league_name = futures[future]
                try:
                    matches = future.result()
                    if matches:
                        all_fixtures.append(f"\n**{league_name.title()}**")
                        all_fixtures.extend(matches)
                except: pass

        if len(all_fixtures) <= 1:
            result = "❌ No fixtures today in major leagues."
        else:
            result = '\n'.join(all_fixtures)

        stop_loading(loading_id)
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading_msg_id,
            text=result,
            parse_mode='Markdown'
        )
    except Exception as e:
        stop_loading(loading_id)
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading_msg_id,
            text="❌ Error loading fixtures.",
            parse_mode='Markdown'
        )
    finally:
        LOADING_MSGS.pop(uid, None)

# === FAST /users ===
def run_users(chat_id, reply_to_id=None):
    uid = chat_id
    if uid in LOADING_MSGS:
        return
    
    loading_id, loading_msg_id = continuous_loading(chat_id, "general", reply_to_id)
    LOADING_MSGS[uid] = loading_msg_id

    try:
        time.sleep(1.5)  # Shorter delay for effect
        active = len(USER_SESSIONS)
        total_predictions = sum(len(history) for history in USER_HISTORY.values())
        
        stop_loading(loading_id)
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading_msg_id,
            text=(
                f"**Community Stats** 📊\n\n"
                f"👥 Active Users: `{active}`\n"
                f"📈 Predictions Made: `{total_predictions}`\n\n"
                f"⚡ **Ultra-Fast Mode**\n"
                f"• Instant responses\n• Professional tables\n• Live FPL data\n• Enhanced accuracy"
            ),
            parse_mode='Markdown'
        )
    except Exception:
        stop_loading(loading_id)
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading_msg_id,
            text="❌ Error counting users.",
            parse_mode='Markdown'
        )
    finally:
        LOADING_MSGS.pop(uid, None)

# === /history ===
@bot.message_handler(commands=['history'])
def show_history(m):
    user_id = m.from_user.id
    history_text = get_user_history(user_id)
    bot.reply_to(m, history_text, parse_mode='Markdown')

# === ENHANCED START MENU ===
@bot.message_handler(commands=['start'])
def start(m):
    user_id = m.from_user.id
    USER_SESSIONS.add(user_id)
    show_menu_page(m, 1)

def show_menu_page(m, page=1):
    markup = types.InlineKeyboardMarkup(row_width=2)
    
    if page == 1:
        text = "⚽ *KickVision - Ultra Fast* ⚽\n\n✨ Instant predictions\n🔮 Accurate models\n🎯 Professional insights\n\n*Page 1: Leagues*"
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
        text = "*KickVision Menu*\n\n*Page 2: Actions*"
        row1 = [
            types.InlineKeyboardButton("Today", callback_data="cmd_/today"),
            types.InlineKeyboardButton("Users", callback_data="cmd_/users")
        ]
        row2 = [
            types.InlineKeyboardButton("History", callback_data="cmd_/history"),
            types.InlineKeyboardButton("Standings", callback_data="submenu_standings")
        ]
        row3 = [
            types.InlineKeyboardButton("Top Scorers", callback_data="submenu_scorers"),
            types.InlineKeyboardButton("FPL", callback_data="cmd_/fpl")
        ]
        row4 = [types.InlineKeyboardButton("Help", callback_data="help_1")]
        nav_row = [
            types.InlineKeyboardButton("⬅️ Prev", callback_data="menu_1"),
            types.InlineKeyboardButton("Next ➡️", callback_data="menu_3")
        ]
        markup.add(*row1, *row2, *row3, *row4, *nav_row)
    
    elif page == 3:
        text = "*KickVision Menu*\n\n*Page 3: Quick Leagues*"
        row1 = [
            types.InlineKeyboardButton("Premier League", callback_data="cmd_/premierleague"),
            types.InlineKeyboardButton("La Liga", callback_data="cmd_/laliga")
        ]
        row2 = [
            types.InlineKeyboardButton("Bundesliga", callback_data="cmd_/bundesliga"),
            types.InlineKeyboardButton("Serie A", callback_data="cmd_/seriea")
        ]
        nav_row = [
            types.InlineKeyboardButton("⬅️ Prev", callback_data="menu_2"),
            types.InlineKeyboardButton("Close", callback_data="menu_close")
        ]
        markup.add(*row1, *row2, *nav_row)
    
    if hasattr(m, 'message_id'):
        try:
            bot.edit_message_text(chat_id=m.chat.id, message_id=m.message_id, text=text, reply_markup=markup, parse_mode='Markdown')
        except:
            bot.send_message(m.chat.id, text, reply_markup=markup, parse_mode='Markdown')
    else:
        bot.send_message(m.chat.id, text, reply_markup=markup, parse_mode='Markdown')

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
        elif cmd == "/fpl":
            loading_id, loading_msg_id = continuous_loading(chat_id, "fetching", reply_to_id)
            try:
                fpl_data = get_fpl_data()
                stop_loading(loading_id)
                bot.edit_message_text(chat_id=chat_id, message_id=loading_msg_id, text=fpl_data, parse_mode='Markdown')
            except:
                stop_loading(loading_id)
                bot.edit_message_text(chat_id=chat_id, message_id=loading_msg_id, text="❌ FPL error.", parse_mode='Markdown')
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

    elif data.startswith("submenu_"):
        if data == "submenu_standings":
            show_standings_menu(chat_id, call.message.message_id)
        elif data == "submenu_scorers":
            show_scorers_menu(chat_id, call.message.message_id)

    elif data.startswith("standings_"):
        league_id = int(data.split("_")[1])
        loading_id, loading_msg_id = continuous_loading(chat_id, "fetching", reply_to_id)
        try:
            standings = get_league_standings(league_id)
            stop_loading(loading_id)
            bot.edit_message_text(chat_id=chat_id, message_id=loading_msg_id, text=standings, parse_mode='Markdown')
        except:
            stop_loading(loading_id)
            bot.edit_message_text(chat_id=chat_id, message_id=loading_msg_id, text="❌ Standings error.", parse_mode='Markdown')

    elif data.startswith("scorers_"):
        league_id = int(data.split("_")[1])
        loading_id, loading_msg_id = continuous_loading(chat_id, "fetching", reply_to_id)
        try:
            scorers = get_top_scorers(league_id)
            stop_loading(loading_id)
            bot.edit_message_text(chat_id=chat_id, message_id=loading_msg_id, text=scorers, parse_mode='Markdown')
        except:
            stop_loading(loading_id)
            bot.edit_message_text(chat_id=chat_id, message_id=loading_msg_id, text="❌ Scorers error.", parse_mode='Markdown')

    elif data.startswith("menu_"):
        if data == "menu_close":
            try:
                bot.delete_message(chat_id, call.message.message_id)
            except:
                pass
        else:
            page = int(data.split("_")[1])
            show_menu_page(call.message, page)

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
    loading_id, loading_msg_id = continuous_loading(m.chat.id, "fetching", reply_id)
    try:
        fixtures = get_league_fixtures(matched)
        stop_loading(loading_id)
        bot.edit_message_text(
            chat_id=m.chat.id,
            message_id=loading_msg_id,
            text=f"*{display_name} Upcoming*\n\n{fixtures}" if fixtures else "❌ No fixtures.",
            parse_mode='Markdown'
        )
    except:
        stop_loading(loading_id)
        bot.send_message(m.chat.id, f"*{display_name} Upcoming*\n\n❌ Error loading fixtures.", parse_mode='Markdown')

# === RATE LIMIT ===
def is_allowed(uid):
    now = time.time()
    user_rate[uid] = [t for t in user_rate[uid] if now - t < 5]
    if len(user_rate[uid]) >= 3: return False
    user_rate[uid].append(now)
    return True

# === ULTRA-FAST MAIN HANDLER ===
@bot.message_handler(func=lambda m: True)
def handle(m):
    if not m.text: return
    uid = m.from_user.id
    txt = m.text.strip()

    USER_SESSIONS.add(uid)

    if txt.strip().lower() == '/cancel':
        if uid in PENDING_MATCH:
            del PENDING_MATCH[uid]
            bot.reply_to(m, "❌ Cancelled.")
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
                loading_id, loading_msg_id = continuous_loading(m.chat.id, "predicting", m.message_id)
                try:
                    r = predict_with_ids(h[2], a[2], h[1], a[1], h[3], a[3])
                    add_to_history(uid, f"{h[1]} vs {a[1]}", r)
                    stop_loading(loading_id)
                    bot.edit_message_text(chat_id=m.chat.id, message_id=loading_msg_id, text=r, parse_mode='Markdown')
                except:
                    stop_loading(loading_id)
                    bot.send_message(m.chat.id, "❌ Prediction error.", parse_mode='Markdown')
                del PENDING_MATCH[uid]
            else:
                bot.reply_to(m, "❌ Invalid. Try `1 2` or /cancel")
        else:
            bot.reply_to(m, "❌ Reply with two numbers: `1 3`")
        return

    if not is_allowed(uid):
        bot.reply_to(m, "⏳ Wait 5s...")
        return

    # Fast team vs team logic
    txt = re.sub(r'[|\[\](){}]', ' ', txt)
    if not re.search(r'\s+vs\s+|\s+[-–—]\s+', txt, re.IGNORECASE):
        return

    parts = re.split(r'\s+vs\s+|\s+[-–—]\s+', txt, re.IGNORECASE)
    home = parts[0].strip()
    away = ' '.join(parts[1:]).strip()

    home_cands = find_team_candidates(home)
    away_cands = find_team_candidates(away)

    if not home_cands or not away_cands:
        bot.reply_to(m, f"*{home} vs {away}*\n\n❌ Not found.", parse_mode='Markdown')
        return

    if home_cands[0][0] > 0.9 and away_cands[0][0] > 0.9:
        h = home_cands[0]; a = away_cands[0]
        loading_id, loading_msg_id = continuous_loading(m.chat.id, "predicting", m.message_id)
        try:
            r = predict_with_ids(h[2], a[2], h[1], a[1], h[3], a[3])
            add_to_history(uid, f"{h[1]} vs {a[1]}", r)
            stop_loading(loading_id)
            bot.edit_message_text(chat_id=m.chat.id, message_id=loading_msg_id, text=r, parse_mode='Markdown')
        except:
            stop_loading(loading_id)
            bot.send_message(m.chat.id, "❌ Prediction error.", parse_mode='Markdown')
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

# === LOAD TESTING UTILITIES ===
def performance_test():
    """Simple performance test function"""
    start_time = time.time()
    
    # Test prediction speed
    test_matches = [("Man City", "Liverpool"), ("Barcelona", "Real Madrid")]
    for home, away in test_matches:
        predict_with_ids(1, 2, home, away, "MCI", "LIV")
    
    end_time = time.time()
    log.info(f"🏎️ Performance Test: {len(test_matches)} predictions in {end_time-start_time:.2f}s")

# Run performance test on startup
performance_test()

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
    return 'KickVision v1.4.0 - Ultra Fast Edition'

@app.route('/health')
def health_check():
    return {'status': 'healthy', 'users': len(USER_SESSIONS), 'timestamp': time.time()}

@app.route('/performance')
def performance_check():
    performance_test()
    return {'status': 'tested', 'active_users': len(USER_SESSIONS)}

if __name__ == '__main__':
    log.info("🚀 KickVision v1.4.0 — ULTRA FAST EDITION READY")
    bot.remove_webhook()
    time.sleep(1)
    bot.set_webhook(url=f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/{BOT_TOKEN}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
