#!/usr/bin/env python3
"""
KickVision v1.1.0 — Enhanced with Free Premium Features
Added: Match importance, previews, educational tips, referral system, premium tiers
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

# === NEW: PREMIUM FEATURES (FREE) ===
PREMIUM_USERS = set()  # Manually activated premium users
USER_STATS = defaultdict(lambda: {'predictions_made': 0, 'premium_until': None})
REFERRAL_CODES = {}  # code -> user_id
REFERRALS = defaultdict(list)  # user_id -> list of referred users
WAITLIST = set()  # Users waiting for premium

# Free tier limits
FREE_TIER = {
    'predictions_per_day': 8,
    'leagues': ['Premier League', 'La Liga', 'Champions League'],
    'cache_time': 3600
}

PREMIUM_FEATURES = {
    'predictions_per_day': 50,
    'all_leagues': True,
    'cache_time': 300,  # 5 minutes
    'priority_support': True
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

# === NEW: REFERRAL SYSTEM ===
def generate_referral_code(user_id):
    code = f"KV{user_id % 10000:04d}"
    REFERRAL_CODES[code] = user_id
    return code

def get_referral_message(user_id):
    code = generate_referral_code(user_id)
    return (
        f"**Invite Friends & Earn Rewards!** 🤝\n\n"
        f"Share your code: `{code}`\n"
        f"• Friends get +3 free predictions\n"
        f"• You get premium features after 3 referrals\n"
        f"• Help build our betting community!\n\n"
        f"Just tell them to use /start {code}"
    )

# === NEW: PREMIUM STATUS CHECK ===
def is_premium_user(user_id):
    if user_id in PREMIUM_USERS:
        return True
    # Check if user has enough referrals
    if len(REFERRALS.get(user_id, [])) >= 3:
        PREMIUM_USERS.add(user_id)
        return True
    return False

def get_user_tier(user_id):
    if is_premium_user(user_id):
        return "premium"
    return "free"

def get_predictions_remaining(user_id):
    tier = get_user_tier(user_id)
    if tier == "premium":
        return "Unlimited"  # Or a high number
    # Simple daily limit check (reset not implemented here)
    today = date.today().isoformat()
    user_predictions_today = USER_STATS[user_id].get('today', {}).get('count', 0)
    remaining = max(0, FREE_TIER['predictions_per_day'] - user_predictions_today)
    return f"{remaining}/{FREE_TIER['predictions_per_day']}"

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

# === WEIGHTED STATS ===
def get_weighted_stats(team_id, is_home):
    cache_key = f"stats_{team_id}_{'h' if is_home else 'a'}"
    now = time.time()
    if cache_key in TEAM_CACHE and now - TEAM_CACHE[cache_key]['time'] < 3600:
        return TEAM_CACHE[cache_key]['data']
    data = safe_get(f"{API_BASE}/teams/{team_id}/matches", {'status': 'FINISHED', 'limit': 6})
    if not data or len(data.get('matches', [])) < 3:
        return (1.8, 1.0) if is_home else (1.2, 1.5)
    gf, ga, weights = [], [], []
    for i, m in enumerate(reversed(data['matches'][:6])):
        try:
            home_id = m['homeTeam']['id']
            sh = m['score']['fullTime']['home'] or 0
            sa = m['score']['fullTime']['away'] or 0
            weight = 2.0 if i < 2 else 1.0
            if home_id == team_id:
                gf.append(sh * weight); ga.append(sa * weight); weights.append(weight)
            else:
                gf.append(sa * weight); ga.append(sh * weight); weights.append(weight)
        except: pass
    total_weight = sum(weights)
    stats = (round(sum(gf)/total_weight, 2), round(sum(ga)/total_weight, 2)) if total_weight > 0 else ((1.8, 1.0) if is_home else (1.2, 1.5))
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

# === REAL 50×500 SIMS ===
def run_single_model(seed, h_gf, h_ga, a_gf, a_ga):
    random.seed(seed)
    np.random.seed(seed)
    home_xg = (h_gf * a_ga * 1.1) ** 0.5 * random.uniform(0.9, 1.1)
    away_xg = (a_gf * h_ga * 0.9) ** 0.5 * random.uniform(0.9, 1.1)
    if home_xg < 2.0 and away_xg < 2.0:
        tau = 1 - 0.05 * home_xg * away_xg
        home_xg *= tau; away_xg *= tau
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
    score_counts = Counter(zip(all_home_goals, all_away_goals)) if all_home_goals else Counter({(1,0):1})
    most_likely = score_counts.most_common(1)[0][0]
    return {
        'home_win': round(home_win * 100),
        'draw': round(draw * 100),
        'away_win': round(away_win * 100),
        'score': f"{most_likely[0]}-{most_likely[1]}"
    }

# === VERDICT (NO BIAS) ===
def get_verdict(model, market=None):
    h, d, a = model['home_win'], model['draw'], model['away_win']
    if market and market.get('home') and market.get('away'):
        mh = 1/market['home']; ma = 1/market['away']
        md = 1/market['draw'] if market.get('draw') else (mh + ma) * 0.1
        total = mh + md + ma
        if total > 0:
            h = int(h * 0.7 + (mh/total*100 * 0.3))
            d = int(d * 0.7 + (md/total*100 * 0.3))
            a = int(a * 0.7 + (ma/total*100 * 0.3))
    max_pct = max(h, d, a)
    if d == max_pct: return "Draw", h, d, a
    elif h == max_pct: return "Home Win", h, d, a
    else: return "Away Win", h, d, a

# === CACHED PREDICTION ===
def cached_prediction(hid, aid, hname, aname, h_tla, a_tla):
    prediction_key = f"pred_{hid}_{aid}"
    now = time.time()
    if prediction_key in PREDICTION_CACHE and now - PREDICTION_CACHE[prediction_key]['time'] < 3600:
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
    # Update user stats
    USER_STATS[user_id]['predictions_made'] += 1

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

# === PREDICT (USES CACHED VERSION) ===
def predict_with_ids(hid, aid, hname, aname, h_tla, a_tla):
    return cached_prediction(hid, aid, hname, aname, h_tla, a_tla)

# === LEAGUE FIXTURES ===
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
        home = m['homeTeam']['name']
        away = m['awayTeam']['name']
        hid = m['homeTeam']['id']
        aid = m['awayTeam']['id']
        pred = predict_with_ids(hid, aid, home, away, '', '')
        pred_lines = pred.splitlines()
        body = '\n'.join(pred_lines[2:]) if len(pred_lines) > 2 else pred
        fixtures.append(f"*{date}*\n{home} vs {away}\n{body}")
    return '\n\n'.join(fixtures)

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

# === NEW: PREMIUM COMMAND ===
@bot.message_handler(commands=['premium'])
def premium_info(m):
    user_id = m.from_user.id
    tier = get_user_tier(user_id)
    referrals_count = len(REFERRALS.get(user_id, []))
    
    if tier == "premium":
        bot.reply_to(m, 
            "🎉 **You're a Premium User!** 🎉\n\n"
            "✅ Unlimited predictions\n"
            "✅ All leagues available\n"
            "✅ Priority processing\n"
            "✅ Advanced insights\n\n"
            "Thank you for being part of KickVision Premium!",
            parse_mode='Markdown'
        )
    else:
        remaining = get_predictions_remaining(user_id)
        bot.reply_to(m,
            "🚀 **KickVision Premium** 🚀\n\n"
            "✨ *What you get:*\n"
            "• Unlimited daily predictions\n"
            "• All leagues worldwide\n" 
            "• Faster processing\n"
            "• Priority support\n"
            "• Advanced betting insights\n\n"
            f"📊 *Your Status:*\n"
            f"• Tier: {tier.title()}\n"
            f"• Predictions left today: {remaining}\n"
            f"• Referrals: {referrals_count}/3 needed\n\n"
            "🎁 *How to upgrade:*\n"
            "1. Refer 3 friends using /referral\n"
            "2. OR Contact admin for direct upgrade\n\n"
            "Use /referral to start earning premium!",
            parse_mode='Markdown'
        )

# === NEW: REFERRAL COMMAND ===
@bot.message_handler(commands=['referral'])
def referral_command(m):
    user_id = m.from_user.id
    bot.reply_to(m, get_referral_message(user_id), parse_mode='Markdown')

# === NEW: STATS COMMAND ===
@bot.message_handler(commands=['stats'])
def user_stats(m):
    user_id = m.from_user.id
    tier = get_user_tier(user_id)
    predictions_made = USER_STATS[user_id].get('predictions_made', 0)
    referrals_count = len(REFERRALS.get(user_id, []))
    remaining = get_predictions_remaining(user_id)
    
    stats_text = [
        "📈 **Your KickVision Stats**",
        "",
        f"👤 **Tier:** {tier.title()}",
        f"📊 **Predictions Made:** {predictions_made}",
        f"🎯 **Today's Limit:** {remaining}",
        f"🤝 **Referrals:** {referrals_count}/3",
        "",
        "Use /premium to upgrade your account!",
        "Use /referral to invite friends!"
    ]
    
    bot.reply_to(m, '\n'.join(stats_text), parse_mode='Markdown')

# === ENHANCED /today ===
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
                results.append(f"`{t} UTC` {hname} vs {aname}\n{body}")
            return results, len(data['matches'])
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_and_predict, lid, name): name for name, lid in LEAGUE_MAP.items() if ' ' in name}
            for future in as_completed(futures):
                league_name = futures[future]
                try:
                    matches, total = future.result()
                    if matches:
                        all_fixtures.append(f"**{league_name.title()}**")
                        all_fixtures.extend(matches)
                        if total > 3:
                            all_fixtures.append(f"_+{total-3} more..._")
                        all_fixtures.append("")
                except: pass
        if not all_fixtures:
            result = "No fixtures today in major leagues."
        else:
            result = "*Today's Fixtures & Predictions*\n\n" + "\n".join(all_fixtures).strip()
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
        premium_count = len(PREMIUM_USERS)
        total_predictions = sum(stats.get('predictions_made', 0) for stats in USER_STATS.values())
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading_msg.message_id,
            text=(
                f"**Community Stats** 📊\n\n"
                f"👥 Active Users: `{active}`\n"
                f"⭐ Premium Members: `{premium_count}`\n"
                f"📈 Predictions Made: `{total_predictions}`\n"
                f"🤝 Referrals Active: `{sum(len(refs) for refs in REFERRALS.values())}`"
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
    predictions_made = USER_STATS[user_id].get('predictions_made', 0)
    enhanced_text = f"{history_text}\n\n📊 Total Predictions: {predictions_made}"
    bot.reply_to(m, enhanced_text, parse_mode='Markdown')

# === ENHANCED START MENU ===
@bot.message_handler(commands=['start'])
def start(m):
    user_id = m.from_user.id
    USER_SESSIONS.add(user_id)
    
    # Check for referral code
    if len(m.text.split()) > 1:
        referral_code = m.text.split()[1]
        if referral_code in REFERRAL_CODES and REFERRAL_CODES[referral_code] != user_id:
            referrer_id = REFERRAL_CODES[referral_code]
            if user_id not in REFERRALS[referrer_id]:
                REFERRALS[referrer_id].append(user_id)
                # Notify referrer
                try:
                    bot.send_message(
                        referrer_id,
                        f"🎉 New referral! {m.from_user.first_name} joined using your code.\n"
                        f"You now have {len(REFERRALS[referrer_id])}/3 referrals for premium!",
                        parse_mode='Markdown'
                    )
                except:
                    pass
    
    show_menu_page(m, 1)

def show_menu_page(m, page=1):
    user_id = m.from_user.id
    tier = get_user_tier(user_id)
    remaining = get_predictions_remaining(user_id)
    
    markup = types.InlineKeyboardMarkup(row_width=2)
    
    if page == 1:
        text = (
            f"⚽ *KickVision Football Predictions* ⚽\n\n"
            f"✨ *Advanced AI-powered match predictions*\n"
            f"🔮 *Proven statistical models*\n"
            f"🎯 *Professional betting insights*\n\n"
            f"👤 *Your Status: {tier.title()}*\n"
            f"📊 *Predictions Today: {remaining}*\n\n"
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
            f"👤 *Your Status: {tier.title()}*\n"
            f"📊 *Predictions Today: {remaining}*\n\n"
            f"*Page 2: Quick Actions*"
        )
        row1 = [
            types.InlineKeyboardButton("Today", callback_data="cmd_/today"),
            types.InlineKeyboardButton("Users", callback_data="cmd_/users")
        ]
        row2 = [
            types.InlineKeyboardButton("History", callback_data="cmd_/history"),
            types.InlineKeyboardButton("Stats", callback_data="cmd_/stats")
        ]
        row3 = [
            types.InlineKeyboardButton("Premium", callback_data="cmd_/premium"),
            types.InlineKeyboardButton("Referral", callback_data="cmd_/referral")
        ]
        row4 = [types.InlineKeyboardButton("Help", callback_data="help_1")]
        nav_row = [types.InlineKeyboardButton("Prev ⬅️", callback_data="menu_1")]
        markup.add(*row1, *row2, *row3, *row4, *nav_row)
    
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
            "*Account Commands*\n"
            "• `/stats` — Your prediction statistics\n"
            "• `/history` — Your prediction history\n"
            "• `/premium` — Premium features info\n"
            "• `/referral` — Invite friends\n\n"
            "_Tap Next for more._"
        )
        markup.add(next_btn, close_btn)
    elif page == 2:
        text = (
            "📃 *KickVision — Help (Page 2/3)*\n\n"
            "*How It Works*\n"
            "• Uses advanced statistical models\n"
            "• Analyzes team form & xG data\n"
            "• Runs Monte Carlo simulations\n"
            "• Provides win probabilities\n\n"
            "*Free Tier*\n"
            "• 8 predictions per day\n"
            "• Major leagues only\n"
            "• Standard processing\n\n"
            "_Tap Next for premium._"
        )
        markup.add(prev_btn, next_btn, close_btn)
    elif page == 3:
        text = (
            "📃 *KickVision — Help (Page 3/3)*\n\n"
            "*Premium Features* 🚀\n"
            "• Unlimited predictions\n"
            "• All leagues worldwide\n"
            "• Faster processing\n"
            "• Priority support\n\n"
            "*How to Upgrade*\n"
            "• Refer 3 friends OR\n"
            "• Contact admin\n\n"
            "Use /referral to start earning!"
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
        elif cmd == "/stats":
            user_id = call.from_user.id
            tier = get_user_tier(user_id)
            predictions_made = USER_STATS[user_id].get('predictions_made', 0)
            referrals_count = len(REFERRALS.get(user_id, []))
            remaining = get_predictions_remaining(user_id)
            stats_text = (
                f"📈 **Your Stats**\n\n"
                f"👤 Tier: {tier.title()}\n"
                f"📊 Predictions: {predictions_made}\n"
                f"🎯 Today: {remaining}\n"
                f"🤝 Referrals: {referrals_count}/3"
            )
            bot.send_message(chat_id, stats_text, parse_mode='Markdown')
        elif cmd == "/premium":
            user_id = call.from_user.id
            tier = get_user_tier(user_id)
            if tier == "premium":
                bot.send_message(chat_id, "🎉 You're already a Premium user! Enjoy the features!", parse_mode='Markdown')
            else:
                referrals_count = len(REFERRALS.get(user_id, []))
                bot.send_message(chat_id, 
                    f"🚀 Upgrade to Premium!\n\n"
                    f"Referrals: {referrals_count}/3\n"
                    f"Use /referral to invite friends!\n\n"
                    f"Or contact admin for direct upgrade.",
                    parse_mode='Markdown'
                )
        elif cmd == "/referral":
            user_id = call.from_user.id
            bot.send_message(chat_id, get_referral_message(user_id), parse_mode='Markdown')
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

    elif data.startswith("menu_"):
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

    # Check daily limit for free users
    if not is_premium_user(uid):
        today = date.today().isoformat()
        if 'today' not in USER_STATS[uid]:
            USER_STATS[uid]['today'] = {'date': today, 'count': 0}
        elif USER_STATS[uid]['today']['date'] != today:
            USER_STATS[uid]['today'] = {'date': today, 'count': 0}
        
        if USER_STATS[uid]['today']['count'] >= FREE_TIER['predictions_per_day']:
            remaining = get_predictions_remaining(uid)
            bot.reply_to(m, 
                f"❌ Daily limit reached!\n\n"
                f"You've used {FREE_TIER['predictions_per_day']}/8 free predictions today.\n\n"
                f"🚀 Upgrade to Premium for unlimited predictions!\n"
                f"Use /premium to learn more.",
                parse_mode='Markdown'
            )
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
        
        # Increment prediction count for free users
        if not is_premium_user(uid):
            USER_STATS[uid]['today']['count'] += 1
        
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
    return 'KickVision Bot v1.1.0 is running!'

if __name__ == '__main__':
    log.info("KickVision v1.1.0 — ENHANCED & READY")
    bot.remove_webhook()
    time.sleep(1)
    bot.set_webhook(url=f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/{BOT_TOKEN}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
