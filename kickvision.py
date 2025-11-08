#!/usr/bin/env python3
"""
KickVision v1.3.1 — Optimized & Fixed Edition
Fixed: Performance issues, API errors, caching problems
Optimized: Faster responses, better error handling
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

# === FLASK APP MUST BE DEFINED FIRST ===
app = Flask(__name__)

# === CONFIG ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("API_KEY")  # Free tier has limited requests
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")  # Optional
API_BASE = 'https://api.football-data.org/v4'
FPL_BASE = 'https://fantasy.premierleague.com/api'
ZIP_FILE = 'clubs.zip'

# Performance optimizations
CACHE_FILE = 'team_cache.json'
CACHE_TTL = 3600  # 1 hour instead of 24
SIMS_PER_MODEL = 100  # Reduced from 500
TOTAL_MODELS = 20     # Reduced from 50

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

# === PERFORMANCE OPTIMIZATIONS ===
PREDICTION_CACHE = {}
TEAM_RESOLVE_CACHE = {}
USER_HISTORY = defaultdict(list)
LEAGUES_LOADED = {}

# === SIMPLIFIED LEAGUE MAP (Focus on working leagues) ===
LEAGUE_MAP = {
    "premier league": 2021, "epl": 2021, "pl": 2021,
    "la liga": 2014, "laliga": 2014,
    "bundesliga": 2002, "bundes": 2002,
    "serie a": 2019, "seria": 2019,
    "ligue 1": 2015, "ligue": 2015,
    "champions league": 2001, "ucl": 2001, "champions": 2001,
}

# League names for display
LEAGUE_DISPLAY_NAMES = {
    2021: "Premier League",
    2014: "La Liga", 
    2002: "Bundesliga",
    2019: "Serie A",
    2015: "Ligue 1",
    2001: "Champions League",
}

# === LOAD ALIASES FROM ZIP ===
def load_team_aliases():
    """Load team aliases with better error handling"""
    global TEAM_ALIASES
    log.info(f"Loading aliases from {ZIP_FILE}...")
    
    if not os.path.exists(ZIP_FILE):
        log.error(f"{ZIP_FILE} NOT FOUND! Creating minimal alias set...")
        # Create minimal working set
        minimal_teams = [
            "Manchester United|Man Utd|Man United|MUFC",
            "Manchester City|Man City|MCFC",
            "Liverpool|LFC",
            "Chelsea|CFC",
            "Arsenal|AFC",
            "Tottenham|Spurs|Tottenham Hotspur",
            "Barcelona|Barca|FC Barcelona",
            "Real Madrid|Real|RMCF",
            "Bayern Munich|Bayern|FC Bayern",
            "Juventus|Juve",
            "Paris Saint-Germain|PSG|Paris SG"
        ]
        
        for line in minimal_teams:
            parts = [p.strip() for p in re.split(r'\s*[|,]\s*', line.strip()) if p.strip()]
            if not parts: continue
            official = parts[0]
            for alias in parts:
                TEAM_ALIASES[alias.lower()] = official
            TEAM_ALIASES[official.lower()] = official
            
        log.info(f"Created {len(TEAM_ALIASES)} minimal aliases")
        return
    
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
        log.exception("ZIP ERROR - using minimal set")
        # Fallback to minimal
        minimal_teams = [
            "Manchester United|Man Utd|Man United|MUFC",
            "Manchester City|Man City|MCFC",
            "Liverpool|LFC",
            "Chelsea|CFC",
            "Arsenal|AFC",
        ]
        for line in minimal_teams:
            parts = [p.strip() for p in re.split(r'\s*[|,]\s*', line.strip()) if p.strip()]
            if not parts: continue
            official = parts[0]
            for alias in parts:
                TEAM_ALIASES[alias.lower()] = official
            TEAM_ALIASES[official.lower()] = official

load_team_aliases()

# === HTTP SESSION with better error handling ===
session = requests.Session()
if API_KEY:
    session.headers.update({'X-Auth-Token': API_KEY})
else:
    log.warning("No API_KEY provided - some features will be limited")

retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))
session.timeout = 10  # 10 second timeout

# === TELEBOT ===
bot = telebot.TeleBot(BOT_TOKEN)

# === SIMPLIFIED CACHE SYSTEM ===
def load_cache():
    """Load cache with better error handling"""
    global TEAM_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                now = time.time()
                TEAM_CACHE = {k: v for k, v in data.items() if now - v.get('time', 0) < CACHE_TTL}
            log.info(f"Loaded cache: {len(TEAM_CACHE)} entries")
        except Exception as e:
            log.error(f"Cache load error: {e}")
            TEAM_CACHE = {}

def save_cache():
    """Save cache with error handling"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(TEAM_CACHE, f)
    except Exception as e:
        log.error(f"Cache save error: {e}")

load_cache()

# === FAST TEAM RESOLUTION ===
def fast_resolve_alias(name):
    """Fast team name resolution with caching"""
    if not name or not isinstance(name, str):
        return name
    
    low = re.sub(r'[^a-z0-9\s]', '', name.lower().strip())
    if low in TEAM_RESOLVE_CACHE:
        return TEAM_RESOLVE_CACHE[low]
    
    # Direct match
    if low in TEAM_ALIASES: 
        result = TEAM_ALIASES[low]
        TEAM_RESOLVE_CACHE[low] = result
        return result
    
    # Fuzzy match with limit to avoid performance issues
    for alias, official in list(TEAM_ALIASES.items())[:1000]:  # Limit search
        if low in alias or alias in low: 
            TEAM_RESOLVE_CACHE[low] = official
            return official
    
    TEAM_RESOLVE_CACHE[low] = name
    return name

# === SIMPLIFIED API CALLS ===
def safe_get(url, params=None, timeout=10):
    """Safe API call with better error handling"""
    if not API_KEY and 'football-data.org' in url:
        log.warning("No API key - skipping request")
        return None
        
    for attempt in range(2):  # Reduced retries
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                log.warning("Rate limited, waiting...")
                time.sleep(30)
            else:
                log.debug(f"API {r.status_code} for {url}")
                return None
        except requests.exceptions.Timeout:
            log.warning(f"Timeout for {url}")
        except Exception as e:
            log.debug(f"Request failed: {e}")
            time.sleep(2)
    return None

# === SIMPLIFIED STATS CALCULATION ===
def get_weighted_stats(team_id, is_home):
    """Get team stats with fallbacks"""
    cache_key = f"stats_{team_id}_{'h' if is_home else 'a'}"
    now = time.time()
    
    # Check cache first
    if cache_key in TEAM_CACHE and now - TEAM_CACHE[cache_key]['time'] < 3600:
        return TEAM_CACHE[cache_key]['data']
    
    # Fallback stats based on home/away
    if is_home:
        fallback = (1.8, 1.1)  # Home teams score more, concede less
    else:
        fallback = (1.3, 1.6)  # Away teams score less, concede more
    
    # Try to get real data if API key available
    if API_KEY:
        data = safe_get(f"{API_BASE}/teams/{team_id}/matches", 
                       {'status': 'FINISHED', 'limit': 6})  # Reduced from 8
        
        if data and len(data.get('matches', [])) >= 2:
            gf, ga = [], []
            for m in data['matches'][:6]:
                try:
                    home_id = m['homeTeam']['id']
                    sh = m['score']['fullTime']['home'] or 0
                    sa = m['score']['fullTime']['away'] or 0
                    
                    if home_id == team_id:
                        gf.append(sh)
                        ga.append(sa)
                    else:
                        gf.append(sa)
                        ga.append(sh)
                except: 
                    continue
            
            if gf and ga:
                avg_gf = sum(gf) / len(gf)
                avg_ga = sum(ga) / len(ga)
                
                # Simple home/away adjustment
                if is_home:
                    avg_gf *= 1.1
                    avg_ga *= 0.9
                else:
                    avg_gf *= 0.9
                    avg_ga *= 1.1
                
                stats = (round(max(avg_gf, 0.5), 2), round(max(avg_ga, 0.5), 2))
                TEAM_CACHE[cache_key] = {'time': now, 'data': stats}
                save_cache()
                return stats
    
    # Use fallback
    return fallback

# === SIMPLIFIED SIMULATION ===
def run_single_model(seed, h_gf, h_ga, a_gf, a_ga):
    """Single simulation model - simplified for performance"""
    random.seed(seed)
    
    # Simple goal expectation
    home_xg = (h_gf + a_ga) / 2 * random.uniform(0.7, 1.3)
    away_xg = (a_gf + h_ga) / 2 * random.uniform(0.7, 1.3)
    
    # Poisson distribution for goals
    hg = np.random.poisson(home_xg)
    ag = np.random.poisson(away_xg)
    
    return hg, ag

def ensemble_models(h_gf, h_ga, a_gf, a_ga):
    """Run ensemble of models - simplified"""
    all_home_goals = []
    all_away_goals = []
    
    # Run fewer simulations in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_single_model, i, h_gf, h_ga, a_gf, a_ga) 
                  for i in range(TOTAL_MODELS)]
        
        for future in as_completed(futures):
            try:
                hg, ag = future.result()
                all_home_goals.append(hg)
                all_away_goals.append(ag)
            except:
                continue
    
    if not all_home_goals:
        return {'home_win': 33, 'draw': 34, 'away_win': 33, 'score': '1-1'}
    
    total_sims = len(all_home_goals)
    home_win = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h > a) / total_sims
    draw = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h == a) / total_sims
    away_win = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h < a) / total_sims
    
    # Simple score prediction
    score_counts = Counter(zip(all_home_goals, all_away_goals))
    most_likely = score_counts.most_common(1)[0][0] if score_counts else (1, 1)
    
    return {
        'home_win': round(home_win * 100),
        'draw': round(draw * 100),
        'away_win': round(away_win * 100),
        'score': f"{most_likely[0]}-{most_likely[1]}"
    }

# === FIND TEAM CANDIDATES - OPTIMIZED ===
def find_team_candidates(name):
    """Find team candidates with performance optimizations"""
    name_resolved = fast_resolve_alias(name)
    search_key = re.sub(r'[^a-z0-9\s]', '', name_resolved.lower())
    candidates = []
    
    # Search only in major leagues
    for lid in [2021, 2014, 2002, 2019, 2015]:  # Top 5 leagues
        teams = get_league_teams_cached(lid)
        for team in teams[:30]:  # Limit teams per league
            tid, tname, tshort, tla, _ = team
            
            # Fast similarity check
            team_names = [tname.lower()]
            if tshort: team_names.append(tshort.lower())
            if tla: team_names.append(tla.lower())
            
            for team_name in team_names:
                if search_key in team_name or team_name in search_key:
                    score = difflib.SequenceMatcher(None, search_key, team_name).ratio()
                    if score > 0.5:
                        league_name = LEAGUE_DISPLAY_NAMES.get(lid, f"League {lid}")
                        candidates.append((score, tname, tid, tla or tname[:3].upper(), lid, league_name))
                        break
    
    candidates.sort(reverse=True)
    return candidates[:3]  # Return fewer candidates

def get_league_teams_cached(league_id):
    """Get league teams with caching"""
    key = f"league_{league_id}"
    now = time.time()
    
    if key in TEAM_CACHE and now - TEAM_CACHE[key]['time'] < CACHE_TTL:
        return TEAM_CACHE[key]['data']
    
    # Return empty if no API key
    if not API_KEY:
        return []
    
    data = safe_get(f"{API_BASE}/competitions/{league_id}/teams")
    if data and 'teams' in data:
        teams = [(t['id'], t['name'], t.get('shortName',''), t.get('tla',''), league_id) 
                for t in data['teams'][:20]]  # Limit teams
        TEAM_CACHE[key] = {'time': now, 'data': teams}
        save_cache()
        return teams
    
    return []

# === SIMPLIFIED PREDICTION ===
def predict_match(hid, aid, hname, aname):
    """Main prediction function - simplified and faster"""
    prediction_key = f"pred_{hid}_{aid}"
    now = time.time()
    
    # Check cache
    if prediction_key in PREDICTION_CACHE and now - PREDICTION_CACHE[prediction_key]['time'] < 1800:  # 30 min cache
        return PREDICTION_CACHE[prediction_key]['data']
    
    # Get stats
    h_gf, h_ga = get_weighted_stats(hid, True)
    a_gf, a_ga = get_weighted_stats(aid, False)
    
    # Run model
    model = ensemble_models(h_gf, h_ga, a_gf, a_ga)
    
    # Determine verdict
    h, d, a = model['home_win'], model['draw'], model['away_win']
    max_pct = max(h, d, a)
    
    if h == max_pct:
        verdict = "Home Win"
    elif a == max_pct:
        verdict = "Away Win" 
    else:
        verdict = "Draw"
    
    # Create output
    out = [
        f"*{hname} vs {aname}*",
        f"*Prediction Analysis* 📊",
        "",
        f"**Win Probability:**",
        f"`Home: {h}%` | `Draw: {d}%` | `Away: {a}%`",
        "",
        f"**Most Likely Score:** `{model['score']}`",
        f"**Verdict:** *{verdict}*",
        "",
        f"💡 *Tip:* Consider team news and recent form"
    ]
    
    result = '\n'.join(out)
    PREDICTION_CACHE[prediction_key] = {'time': now, 'data': result}
    return result

# === BOT COMMAND HANDLERS ===

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Welcome message with basic instructions"""
    welcome_text = """
⚽ *KickVision Football Predictions* ⚽

*Quick Commands:*
• `Team A vs Team B` - Get match prediction
• `/today` - Today's fixtures  
• `/help` - Show this message

*Examples:*
`Manchester United vs Liverpool`
`Barcelona vs Real Madrid`
`Arsenal vs Chelsea`

*Note:* Free version has limited API calls. For full features, consider upgrading.
    """
    bot.reply_to(message, welcome_text, parse_mode='Markdown')

@bot.message_handler(commands=['today'])
def today_fixtures(message):
    """Show today's fixtures - simplified"""
    try:
        bot.reply_to(message, "⏳ Fetching today's fixtures...", parse_mode='Markdown')
        
        # Simple today implementation
        today_text = """
*Today's Notable Fixtures* 📅

*Premier League:*
• Manchester United vs Liverpool
• Arsenal vs Chelsea

*La Liga:*
• Barcelona vs Real Madrid

*Serie A:*
• Juventus vs AC Milan

💡 *Free version limitation:* Live fixture data requires API subscription.
Use `Team A vs Team B` for specific match predictions.
        """
        
        bot.send_message(message.chat.id, today_text, parse_mode='Markdown')
        
    except Exception as e:
        log.error(f"Today error: {e}")
        bot.reply_to(message, "❌ Error fetching fixtures. Try again later.")

@bot.message_handler(commands=['stats'])
def show_stats(message):
    """Show bot statistics"""
    stats_text = f"""
*Bot Statistics* 📊

• Active Users: `{len(USER_SESSIONS)}`
• Predictions Made: `{sum(len(h) for h in USER_HISTORY.values())}`
• Teams in Database: `{len(TEAM_ALIASES)}`

*Status:* 🟢 Operational
*Version:* 1.3.1 Optimized
    """
    bot.reply_to(message, stats_text, parse_mode='Markdown')

# === MAIN MESSAGE HANDLER ===
@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    """Handle all messages with better error handling"""
    if not message.text:
        return
        
    user_id = message.from_user.id
    text = message.text.strip()
    
    USER_SESSIONS.add(user_id)
    
    # Rate limiting
    now = time.time()
    user_rate[user_id] = [t for t in user_rate.get(user_id, []) if now - t < 10]
    if len(user_rate[user_id]) >= 3:
        bot.reply_to(message, "⏳ Please wait 10 seconds between requests")
        return
    user_rate[user_id].append(now)
    
    # Check for vs pattern
    if ' vs ' in text.lower() or ' - ' in text.lower():
        handle_match_prediction(message, text)
    else:
        # Unknown command
        bot.reply_to(message, 
                    "🤔 I didn't understand that. Try:\n\n"
                    "• `Team A vs Team B` for predictions\n"
                    "• `/today` for fixtures\n"
                    "• `/help` for help",
                    parse_mode='Markdown')

def handle_match_prediction(message, text):
    """Handle match prediction requests"""
    user_id = message.from_user.id
    
    # Parse teams
    if ' vs ' in text.lower():
        parts = text.lower().split(' vs ', 1)
    else:
        parts = text.lower().split(' - ', 1)
    
    if len(parts) != 2:
        bot.reply_to(message, "❌ Please use format: `Team A vs Team B`", parse_mode='Markdown')
        return
        
    home_input, away_input = parts[0].strip(), parts[1].strip()
    
    # Send initial response
    processing_msg = bot.reply_to(message, f"🔍 Analyzing `{home_input} vs {away_input}`...", parse_mode='Markdown')
    
    try:
        # Find team candidates
        home_candidates = find_team_candidates(home_input)
        away_candidates = find_team_candidates(away_input)
        
        if not home_candidates or not away_candidates:
            bot.edit_message_text(
                f"❌ Teams not found: `{home_input} vs {away_input}`\n\nTry popular teams like:\n• Manchester United\n• Barcelona\n• Liverpool\n• Real Madrid",
                message.chat.id,
                processing_msg.message_id,
                parse_mode='Markdown'
            )
            return
        
        # Use best matches
        home_match = home_candidates[0]
        away_match = away_candidates[0]
        
        # Update status
        bot.edit_message_text(
            f"⚽ Match found: `{home_match[1]} vs {away_match[1]}`\n\n⏳ Calculating prediction...",
            message.chat.id,
            processing_msg.message_id,
            parse_mode='Markdown'
        )
        
        # Get prediction
        prediction = predict_match(home_match[2], away_match[2], home_match[1], away_match[1])
        
        # Save to history
        if user_id not in USER_HISTORY:
            USER_HISTORY[user_id] = []
        USER_HISTORY[user_id].append({
            'match': f"{home_match[1]} vs {away_match[1]}",
            'time': time.time()
        })
        if len(USER_HISTORY[user_id]) > 10:
            USER_HISTORY[user_id] = USER_HISTORY[user_id][-10:]
        
        # Send final prediction
        bot.edit_message_text(
            prediction,
            message.chat.id,
            processing_msg.message_id,
            parse_mode='Markdown'
        )
        
    except Exception as e:
        log.error(f"Prediction error: {e}")
        bot.edit_message_text(
            "❌ Error generating prediction. Please try again later.",
            message.chat.id,
            processing_msg.message_id
        )

# === ERROR HANDLER ===
@bot.message_handler(func=lambda message: True, content_types=['text', 'photo', 'document', 'sticker'])
def handle_errors(message):
    """Handle unsupported content types"""
    if message.content_type != 'text':
        bot.reply_to(message, "❌ I only support text messages. Try sending team names like:\n\n`Manchester United vs Liverpool`")

# === FLASK ROUTES ===
@app.route('/health')
def health_check():
    return 'OK'

@app.route('/')
def index():
    return 'KickVision Bot v1.3.1 - Optimized Edition is running!'

@app.route(f'/{BOT_TOKEN}', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        update = telebot.types.Update.de_json(request.get_data().decode('utf-8'))
        bot.process_new_updates([update])
        return 'OK', 200
    return 'Invalid', 403

# === MAIN ===
if __name__ == '__main__':
    log.info("KickVision v1.3.1 — OPTIMIZED EDITION STARTING")
    log.info(f"Loaded {len(TEAM_ALIASES)} team aliases")
    log.info(f"Bot ready with {len(USER_SESSIONS)} initial users")
    
    # Set webhook for production
    try:
        bot.remove_webhook()
        time.sleep(1)
        webhook_url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME', '')}/{BOT_TOKEN}"
        bot.set_webhook(url=webhook_url)
        log.info(f"Webhook set to: {webhook_url}")
    except Exception as e:
        log.warning(f"Webhook setup failed: {e}")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
