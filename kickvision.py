#!/usr/bin/env python3
"""
KickVision v1.3.0 — SMOOTH UNIVERSAL ANIMATIONS
- 0.8s smooth cycle
- League-specific animation in ALL predictions
- Auto-detects league for 'vs' matches
- Animated: /today, /users, vs, /league
"""

import os
import re
import time
import zipfile
import logging
import json
import random
import threading
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date

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
SIMS_PER_MODEL = 1000
TOTAL_MODELS = 100

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

# === SMOOTH ANIMATION ENGINE ===
def animate_smooth(chat_id, message_id, frames, stop_event, delay=0.8):
    """Smooth animation with clean stop"""
    idx = 0
    while not stop_event.is_set():
        try:
            text = frames[idx % len(frames)]
            bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode='Markdown'
            )
            idx += 1
            time.sleep(delay)
        except Exception as e:
            if "message is not modified" not in str(e).lower():
                break
    # Final edit happens outside

# === LEAGUE-THEMED ANIMATIONS ===
LEAGUE_ANIM = {
    "premier league": [
        "Scanning Premier League fixtures...",
        "xG models loading...",
        "Simulating 100,000 outcomes...",
        "Calculating goal probabilities...",
        "Finalizing predictions..."
    ],
    "la liga": [
        "Tiki-taka analysis in progress...",
        "Possession vectors recalculating...",
        "La Liga tempo simulation...",
        "xG chains building...",
        "Finalizing forecast..."
    ],
    "bundesliga": [
        "Gegenpressing scan active...",
        "Transition lanes analyzed...",
        "Poisson model running...",
        "Bayern factor applied...",
        "Bundesliga verdict ready..."
    ],
    "serie a": [
        "Catenaccio defense scan...",
        "Set-piece probability check...",
        "Italian tempo adjustment...",
        "xG from distance model...",
        "Serie A prediction complete..."
    ],
    "ligue 1": [
        "French flair simulation...",
        "PSG attack vectors loaded...",
        "Ligue 1 variance model...",
        "Tempo calibrated...",
        "Prediction locked..."
    ],
    "uefa champions league": [
        "Elite xG models activated...",
        "Knockout intensity applied...",
        "Monte Carlo at 100k sims...",
        "Continental pressure factor...",
        "Champions League verdict..."
    ],
    "europa league": [
        "Thursday night drama loading...",
        "Travel fatigue factor...",
        "Underdog upset potential...",
        "Europa chaos model...",
        "Final prediction ready..."
    ],
    "mls": [
        "DP impact analysis...",
        "MLS parity model running...",
        "Cross-country travel factor...",
        "Playoff intensity mode...",
        "Prediction locked..."
    ],
    "brasileirao": [
        "Samba rhythm analysis...",
        "Flamengo attack vectors...",
        "Goal-heavy model active...",
        "Brazilian flair simulation...",
        "Brasileirão verdict..."
    ],
    "liga mx": [
        "Liguilla pressure scan...",
        "Clásico intensity factor...",
        "Altitude advantage applied...",
        "Liga MX chaos model...",
        "Final prediction ready..."
    ]
}

# Fallback
DEFAULT_ANIM = [
    "Loading data...",
    "Running simulations...",
    "Calculating xG...",
    "Predicting outcome...",
    "Finalizing..."
]

# === USERS ANIMATION ===
USERS_FRAMES = ["Counting users.", "Counting users..", "Counting users..."]

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

# === RESOLVE ALIAS ===
def resolve_alias(name):
    low = re.sub(r'[^a-z0-9\s]', '', str(name).lower().strip())
    if low in TEAM_ALIASES: return TEAM_ALIASES[low]
    for alias, official in TEAM_ALIASES.items():
        if low in alias or alias in low: return official
    return name

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
    name_resolved = resolve_alias(name)
    search_key = re.sub(r'[^a-z0-9\s]', '', name_resolved.lower())
    candidates = []
    
    for lid in LEAGUE_MAP.values():
        teams = get_league_teams(lid)
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

# === SIMULATION ENGINE ===
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
    
    total_sims = len(all_home_goals)
    home_win = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h > a) / total_sims
    draw = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h == a) / total_sims
    away_win = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h < a) / total_sims
    
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

# === PREDICT WITH ANIMATION ===
def predict_with_animation(m, h, a):
    hid, hname, _, h_tla, h_lid, _ = h
    aid, aname, _, a_tla, a_lid, _ = a

    # Auto-detect league
    lid, league_name = auto_detect_league(hid, aid)
    league_key = next((k for k, v in LEAGUE_MAP.items() if v == lid), "premier league")
    frames = LEAGUE_ANIM.get(league_key, DEFAULT_ANIM)

    loading = bot.reply_to(m, frames[0])
    stop_event = threading.Event()
    th = threading.Thread(
        target=animate_smooth,
        args=(m.chat.id, loading.message_id, frames, stop_event),
        daemon=True
    )
    th.start()

    try:
        h_gf, h_ga = get_weighted_stats(hid, True)
        a_gf, a_ga = get_weighted_stats(aid, False)
        model = ensemble_100_models(h_gf, h_ga, a_gf, a_ga)
        market = get_market_odds(hname, aname)
        verdict, h_pct, d_pct, a_pct = get_verdict(model, market)

        out = [
            f"*{hname} vs {aname}*",
            f"_{league_name}_",
            "",
            f"**xG:** `{h_gf:.2f}` — `{a_gf:.2f}`",
            f"**Win:** `{h_pct}%` | `{d_pct}%` | `{a_pct}%`",
            "",
            f"**Most Likely:** `{model['score']}`",
            f"**Verdict:** *{verdict}*"
        ]
        result = '\n'.join(out)
    except:
        result = "Prediction failed."

    stop_event.set()
    bot.edit_message_text(
        chat_id=m.chat.id,
        message_id=loading.message_id,
        text=result,
        parse_mode='Markdown'
    )

# === LEAGUE FIXTURES WITH ANIMATION ===
def get_league_fixtures_with_anim(m, league_name):
    lid = LEAGUE_MAP.get(league_name.lower())
    if not lid:
        bot.reply_to(m, "League not supported.")
        return

    loading = bot.reply_to(m, "Loading fixtures...")
    frames = LEAGUE_ANIM.get(league_name.lower(), DEFAULT_ANIM)
    stop_event = threading.Event()
    th = threading.Thread(
        target=animate_smooth,
        args=(m.chat.id, loading.message_id, frames, stop_event),
        daemon=True
    )
    th.start()

    data = safe_get(f"{API_BASE}/competitions/{lid}/matches", {'status': 'SCHEDULED', 'limit': 10})
    fixtures = []
    if data and data.get('matches'):
        for match in data['matches'][:5]:
            date = match['utcDate'][:10]
            home = match['homeTeam']['name']
            away = match['awayTeam']['name']
            hid = match['homeTeam']['id']
            aid = match['awayTeam']['id']
            pred = predict_with_ids(hid, aid, home, away, '', '')
            fixtures.append(f"*{date}*\n{home} vs {away}\n{pred}")
    
    result = f"*{league_name.title()} Upcoming*\n\n" + '\n\n'.join(fixtures) if fixtures else "No fixtures."
    
    stop_event.set()
    bot.edit_message_text(
        chat_id=m.chat.id,
        message_id=loading.message_id,
        text=result,
        parse_mode='Markdown'
    )

# === /today ANIMATED ===
def run_today(chat_id, reply_to_id=None):
    uid = chat_id
    if uid in LOADING_MSGS: return

    loading = bot.send_message(chat_id, "Loading fixtures...", reply_to_message_id=reply_to_id, parse_mode='Markdown')
    LOADING_MSGS[uid] = loading.message_id

    stop_event = threading.Event()
    th = threading.Thread(
        target=animate_smooth,
        args=(chat_id, loading.message_id, LEAGUE_ANIM["premier league"], stop_event),
        daemon=True
    )
    th.start()

    try:
        today = date.today().isoformat()
        all_fixtures = []
        def fetch_league(lid, name):
            data = safe_get(f"{API_BASE}/competitions/{lid}/matches", {'dateFrom': today, 'dateTo': today})
            if not data or not data.get('matches'): return []
            return [(m['homeTeam']['name'], m['awayTeam']['name'], m['utcDate'][11:16], m['homeTeam']['id'], m['awayTeam']['id']) for m in data['matches']]
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_league, lid, name): name for name, lid in LEAGUE_MAP.items() if ' ' in name}
            for future in as_completed(futures):
                league_name = futures[future]
                try:
                    matches = future.result()
                    if matches:
                        all_fixtures.append(f"**{league_name.title()}**")
                        for h, a, t, hid, aid in matches[:3]:
                            pred = predict_with_ids(hid, aid, h, a, '', '')
                            all_fixtures.append(f"`{t}` {h} vs {a}\n{pred}")
                        if len(matches) > 3:
                            all_fixtures.append(f"_+{len(matches)-3} more..._")
                        all_fixtures.append("")
                except: pass

        result = "*Today's Fixtures & Predictions*\n\n" + "\n".join(all_fixtures).strip() if all_fixtures else "No fixtures today."
        
        stop_event.set()
        bot.edit_message_text(chat_id=chat_id, message_id=loading.message_id, text=result, parse_mode='Markdown')
    except:
        stop_event.set()
        bot.edit_message_text(chat_id=chat_id, message_id=loading.message_id, text="Error.", parse_mode='Markdown')
    finally:
        LOADING_MSGS.pop(uid, None)

# === /users ANIMATED ===
def run_users(chat_id, reply_to_id=None):
    loading = bot.send_message(chat_id, "Counting users.", reply_to_message_id=reply_to_id, parse_mode='Markdown')
    stop_event = threading.Event()
    th = threading.Thread(
        target=animate_smooth,
        args=(chat_id, loading.message_id, USERS_FRAMES, stop_event, 0.6),
        daemon=True
    )
    th.start()
    time.sleep(1.5)
    active = len(USER_SESSIONS)
    stop_event.set()
    bot.edit_message_text(chat_id=chat_id, message_id=loading.message_id, text=f"**Active users:** `{active}`", parse_mode='Markdown')

# === MENU & CALLBACKS ===
@bot.message_handler(commands=['start'])
def start(m):
    show_menu_page(m, 1)

def show_menu_page(m, page=1):
    markup = types.InlineKeyboardMarkup(row_width=2)
    if page == 1:
        text = "*Welcome to KickVision v1.3.0*\n\n*Page 1: Major Leagues*"
        row1 = [types.InlineKeyboardButton("Premier League", callback_data="cmd_/premierleague"), types.InlineKeyboardButton("La Liga", callback_data="cmd_/laliga")]
        row2 = [types.InlineKeyboardButton("Bundesliga", callback_data="cmd_/bundesliga"), types.InlineKeyboardButton("Serie A", callback_data="cmd_/seriea")]
        row3 = [types.InlineKeyboardButton("Ligue 1", callback_data="cmd_/ligue1"), types.InlineKeyboardButton("Champions", callback_data="cmd_/champions")]
        nav = [types.InlineKeyboardButton("Next", callback_data="menu_2")]
        markup.add(*row1, *row2, *row3, *nav)
    elif page == 2:
        text = "*Quick Actions*"
        row1 = [types.InlineKeyboardButton("Today", callback_data="cmd_/today"), types.InlineKeyboardButton("Users", callback_data="cmd_/users")]
        row2 = [types.InlineKeyboardButton("Help", callback_data="help_1")]
        nav = [types.InlineKeyboardButton("Prev", callback_data="menu_1")]
        markup.add(*row1, *row2, *nav)
    bot.send_message(m.chat.id, text, reply_markup=markup, parse_mode='Markdown')

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    chat_id = call.message.chat.id
    reply_to_id = call.message.message_id
    if call.data.startswith("cmd_/"):
        cmd = call.data[5:]
        bot.answer_callback_query(call.id)
        if cmd == "/today":
            run_today(chat_id, reply_to_id)
        elif cmd == "/users":
            run_users(chat_id, reply_to_id)
        else:
            fake_msg = types.Message(message_id=0, from_user=call.from_user, date=datetime.now(), chat=call.message.chat, content_type='text', options=[], json_string=None)
            fake_msg.text = cmd
            dynamic_league_handler(fake_msg)
    elif call.data.startswith("menu_"):
        page = int(call.data.split("_")[1])
        bot.answer_callback_query(call.id)
        show_menu_page(call.message, page)
    elif call.data.startswith("help_"):
        page = int(call.data.split("_")[1])
        show_help_page(call.message, page)
        bot.answer_callback_query(call.id)

# === LEAGUE HANDLER ===
@bot.message_handler(func=lambda m: any(m.text and (m.text.lower().startswith(f"/{k.replace(' ', '')}") or m.text.lower() == k) for k in LEAGUE_MAP))
def dynamic_league_handler(m):
    if not m.text: return
    txt = m.text.strip().lower()
    if txt.startswith('/'): txt = txt[1:]
    matched = next((k for k in LEAGUE_MAP if txt == k.replace(' ', '') or txt == k), None)
    if not matched: return
    get_league_fixtures_with_anim(m, matched)

# === RATE LIMIT ===
def is_allowed(uid):
    now = time.time()
    user_rate[uid] = [t for t in user_rate[uid] if now - t < 5]
    if len(user_rate[uid]) >= 3: return False
    user_rate[uid].append(now)
    return True

# === MAIN HANDLER ===
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
                predict_with_animation(m, h, a)
                del PENDING_MATCH[uid]
            else:
                bot.reply_to(m, "Invalid. Try `1 2` or /cancel")
        else:
            bot.reply_to(m, "Reply with two numbers: `1 3`")
        return

    if not is_allowed(uid):
        bot.reply_to(m, "Wait 5s...")
        return

    txt = re.sub(r'[|\[\](){}]', ' ', txt)
    if not re.search(r'\s+vs\s+|\s+[-–—]\s+', txt, re.IGNORECASE):
        return

    parts = re.split(r'\s+vs\s+|\s+[-–—]\s+', txt, re.IGNORECASE)
    home = parts[0].strip()
    away = ' '.join(parts[1:]).strip()

    home_cands = find_team_candidates(home)
    away_cands = find_team_candidates(away)

    if not home_cands or not away_cands:
        bot.reply_to(m, f"*{home} vs {away}*\n\n_Not found._", parse_mode='Markdown')
        return

    if home_cands[0][0] > 0.9 and away_cands[0][0] > 0.9:
        predict_with_animation(m, home_cands[0], away_cands[0])
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

# === HELP ===
HELP_PAGES = [ ... ]  # (same as before)

def show_help_page(message, page=1):
    page = max(1, min(len(HELP_PAGES), page))
    markup = types.InlineKeyboardMarkup(row_width=2)
    if page > 1:
        markup.add(types.InlineKeyboardButton("Prev", callback_data=f"help_{page-1}"))
    if page < len(HELP_PAGES):
        markup.add(types.InlineKeyboardButton("Next", callback_data=f"help_{page+1}"))
    bot.send_message(message.chat.id, HELP_PAGES[page-1], reply_markup=markup, parse_mode='Markdown')

# === FLASK ===
app = Flask(__name__)
@app.route(f'/{BOT_TOKEN}', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        update = telebot.types.Update.de_json(request.get_data().decode('utf-8'))
        bot.process_new_updates([update])
        return 'OK', 200
    return 'Invalid', 403

if __name__ == '__main__':
    log.info("KickVision v1.3.0 — SMOOTH & UNIVERSAL")
    bot.remove_webhook()
    time.sleep(1)
    bot.set_webhook(url=f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/{BOT_TOKEN}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
