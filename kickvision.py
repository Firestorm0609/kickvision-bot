#!/usr/bin/env python3
"""
KickVision v1.0.0 — Official Release
100-model ensemble | Typo-proof | /cancel | vs Only | Full League Support
"""

import os
import re
import time
import zipfile
import logging
import json
import random
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, mode

import numpy as np
import requests
import difflib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import telebot
from flask import Flask, request

# === CONFIG ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("API_KEY")
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

# === GLOBAL STATE ===
user_rate = defaultdict(list)
TEAM_ALIASES = {}
TEAM_CACHE = {}
LEAGUES_CACHE = {}
PENDING_MATCH = {}

LEAGUE_PRIORITY = {
    "UEFA Champions League": 2001,
    "Premier League": 2021,
    "La Liga": 2014,
    "Bundesliga": 2002,
    "Serie A": 2019,
    "Ligue 1": 2015,
    "Europa League": 2018
}

# === LOAD ALIASES ===
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
                            fixed_teams = []
                            for team in v['data']:
                                if len(team) == 4:
                                    fixed_teams.append(team + (lid,))
                                else:
                                    fixed_teams.append(team)
                            new_cache[k] = {'time': v['time'], 'data': fixed_teams}
                        else:
                            new_cache[k] = v
                TEAM_CACHE = new_cache
            log.info(f"Loaded cache: {len(TEAM_CACHE)} entries")
        except Exception as e:
            log.exception("Cache error")

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
                wait = 60 * (attempt + 1)
                log.warning(f"429 → wait {wait}s")
                time.sleep(wait)
            else:
                log.warning(f"API {r.status_code}: {url}")
                return None
        except Exception as e:
            log.exception(f"Request error: {e}")
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
                    log.info(f"Loaded leagues cache: {len(LEAGUES_CACHE)} competitions")
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
        log.info(f"Fetched {len(LEAGUES_CACHE)} leagues from API")
        return True
    log.warning("Failed to fetch leagues—using priority only")
    return False

if not load_leagues_cache():
    fetch_all_leagues()

if not LEAGUES_CACHE:
    LEAGUES_CACHE = {v: k for k, v in LEAGUE_PRIORITY.items()}

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
    leagues = list(LEAGUES_CACHE.keys())
    candidates = []
    
    for lid in leagues:
        teams = get_league_teams(lid)
        for team in teams:
            if len(team) == 5:
                tid, tname, tshort, tla, _ = team
            else:
                tid, tname, tshort, tla = team
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
    if not common and h_leagues:
        lid = next(iter(h_leagues))
        return lid, LEAGUES_CACHE.get(lid, "League")
    
    priority_order = list(LEAGUE_PRIORITY.values())
    best_lid = max(common, key=lambda x: priority_order.index(x) if x in priority_order else len(priority_order))
    return best_lid, LEAGUES_CACHE.get(best_lid, "League")

# === GET STATS ===
def get_team_stats(team_id, is_home):
    cache_key = f"stats_{team_id}_{is_home}"
    if cache_key in TEAM_CACHE:
        return TEAM_CACHE[cache_key]['data']
    
    data = safe_get(f"{API_BASE}/teams/{team_id}/matches", {'status': 'FINISHED', 'limit': 10})
    if not data or not data.get('matches'):
        stats = (1.6, 1.2) if is_home else (1.1, 1.4)
    else:
        gf, ga = [], []
        for m in data['matches']:
            try:
                home_id = m['homeTeam']['id']
                sh = m['score']['fullTime']['home'] or 0
                sa = m['score']['fullTime']['away'] or 0
                if home_id == team_id:
                    gf.append(sh); ga.append(sa)
                else:
                    gf.append(sa); ga.append(sh)
            except: pass
        stats = (round(np.mean(gf), 2), round(np.mean(ga), 2)) if gf else ((1.6, 1.2) if is_home else (1.1, 1.4))
    
    TEAM_CACHE[cache_key] = {'time': time.time(), 'data': stats}
    save_cache()
    return stats

# === 100 MODEL VARIANTS ===
def run_single_model(seed, h_gf, h_ga, a_gf, a_ga):
    random.seed(seed)
    np.random.seed(seed)
    
    ah = h_gf * random.uniform(0.7, 1.3)
    dh = h_ga * random.uniform(0.7, 1.3)
    aa = a_gf * random.uniform(0.7, 1.3)
    da = a_ga * random.uniform(0.7, 1.3)
    
    rho = random.uniform(-0.1, 0.15)
    home_xg = (ah / 1.4) * (da / 1.4) * 1.4 * random.uniform(1.0, 1.2)
    away_xg = (aa / 1.4) * (dh / 1.4) * 1.4 * random.uniform(0.8, 1.0)
    
    if home_xg < 1.5 and away_xg < 1.5:
        home_xg *= (1 - rho * home_xg * away_xg)
        away_xg *= (1 - rho * home_xg * away_xg)
    
    hg = np.random.poisson(home_xg, SIMS_PER_MODEL)
    ag = np.random.poisson(away_xg, SIMS_PER_MODEL)
    p_home = (hg > ag).mean()
    p_draw = (hg == ag).mean()
    p_away = (hg < ag).mean()
    
    scores = [f"{int(h)}-{int(a)}" for h, a in zip(hg, ag)]
    most_likely = Counter(scores).most_common(1)[0][0]
    
    return {
        'xg_home': home_xg,
        'xg_away': away_xg,
        'home_win': p_home,
        'draw': p_draw,
        'away_win': p_away,
        'score': most_likely
    }

def ensemble_100_models(h_gf, h_ga, a_gf, a_ga):
    seeds = list(range(TOTAL_MODELS))
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda s: run_single_model(s, h_gf, h_ga, a_gf, a_ga), seeds))
    
    final = {
        'xg_home': round(mean([r['xg_home'] for r in results]), 2),
        'xg_away': round(mean([r['xg_away'] for r in results]), 2),
        'home_win': round(mean([r['home_win'] for r in results]) * 100),
        'draw': round(mean([r['draw'] for r in results]) * 100),
        'away_win': round(mean([r['away_win'] for r in results]) * 100),
        'score': mode([r['score'] for r in results])
    }
    return final

# === PREDICT ===
def predict_with_ids(hid, aid, hname, aname, h_tla, a_tla):
    lid, league_name = auto_detect_league(hid, aid)
    h_gf, h_ga = get_team_stats(hid, True)
    a_gf, a_ga = get_team_stats(aid, False)
    
    result = ensemble_100_models(h_gf, h_ga, a_gf, a_ga)
    
    verdict = "Home Win" if result['home_win'] > max(result['away_win'], result['draw']) else \
              "Away Win" if result['away_win'] > max(result['home_win'], result['draw']) else "Draw"
    
    out = [
        f"**{hname} vs {aname} — {league_name}**",
        f"",
        f"**xG: {result['xg_home']:.2f} — {result['xg_away']:.2f}**",
        f"**Home Win: {result['home_win']}%**",
        f"**Draw: {result['draw']}%**",
        f"**Away Win: {result['away_win']}%**",
        f"",
        f"**Most Likely Score: {result['score']}**",
        f"**VERDICT: {verdict}**"
    ]
    return '\n'.join(out)

# === RATE LIMIT ===
def is_allowed(uid):
    now = time.time()
    user_rate[uid] = [t for t in user_rate[uid] if now - t < 5]
    if len(user_rate[uid]) >= 3: return False
    user_rate[uid].append(now)
    return True

# === HELP / HOW ===
def send_help(m):
    help_text = (
        "**How KickVision Works**\n\n"
        "I use **100 AI models** to simulate each match **1000 times per model** — that's **100,000 simulations**!\n\n"
        "From real stats (last 10 games), I predict:\n"
        "• **xG** (expected goals)\n"
        "• **Win %** for Home, Draw, Away\n"
        "• **Most likely score**\n"
        "• **Final verdict**\n\n"
        "Just type: `Team A vs Team B`\n"
        "Example: `Lincoln vs York` (works for minor leagues too!)\n\n"
        "Use **/cancel** to stop selection\n"
        "Use **/start** to begin"
    )
    bot.reply_to(m, help_text, parse_mode='Markdown')

@bot.message_handler(commands=['start', 'help', 'how'])
def start(m):
    send_help(m)

# === MAIN HANDLER ===
@bot.message_handler(func=lambda m: True)
def handle(m):
    uid = m.from_user.id
    txt = m.text.strip()

    if txt.strip().lower() == '/cancel':
        if uid in PENDING_MATCH:
            del PENDING_MATCH[uid]
            bot.reply_to(m, "Match selection cancelled.")
        else:
            bot.reply_to(m, "Nothing to cancel. Try a match: `Team A vs Team B`")
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
                result = predict_with_ids(h[2], a[2], h[1], a[1], h[3], a[3])
                bot.reply_to(m, result, parse_mode='Markdown')
                del PENDING_MATCH[uid]
            else:
                bot.reply_to(m, "Invalid numbers. Try again or **/cancel**")
        else:
            bot.reply_to(m, "Reply with **two numbers**: `1 3` ← picks 1st home, 3rd away\nOr type **/cancel**")
        return

    if not is_allowed(uid):
        bot.reply_to(m, "Wait 5s...")
        return

    txt = re.sub(r'[|\[\](){}]', ' ', txt)
    
    if not re.search(r'\s+vs\s+|\s+[-–—]\s+', txt, re.IGNORECASE):
        bot.reply_to(m, "Use **Team A vs Team B** format\nExample: `Lincoln vs York`\nType **/how** for details")
        return

    parts = re.split(r'\s+vs\s+|\s+[-–—]\s+', txt, re.IGNORECASE)
    home = parts[0].strip()
    away = ' '.join(parts[1:]).strip()

    home_cands = find_team_candidates(home)
    away_cands = find_team_candidates(away)

    if not home_cands or not away_cands:
        bot.reply_to(m, f"Couldn't find: `{home}` or `{away}` in 144+ leagues.\nTry a different spelling or example: `Man City vs Liverpool`\nType **/how** for tips")
        return

    if home_cands[0][0] > 0.9 and away_cands[0][0] > 0.9:
        h = home_cands[0]
        a = away_cands[0]
        result = predict_with_ids(h[2], a[2], h[1], a[1], h[3], a[3])
        bot.reply_to(m, result, parse_mode='Markdown')
        return

    msg = ["**Did you mean?**"]
    msg.append(f"**Home:** {home}")
    for i, (_, name, _, tla, lid, lname) in enumerate(home_cands, 1):
        msg.append(f"{i}. {name} ({tla}) — {lname}")
    msg.append(f"**Away:** {away}")
    for i, (_, name, _, tla, lid, lname) in enumerate(away_cands, 1):
        msg.append(f"{i}. {name} ({tla}) — {lname}")
    msg.append("\n**Reply with two numbers**: `1 3` ← picks 1st home, 3rd away\nOr type **/cancel**")
    bot.reply_to(m, '\n'.join(msg), parse_mode='Markdown')
    PENDING_MATCH[uid] = (home, away, home_cands, away_cands)

# === FLASK WEBHOOK ===
app = Flask(__name__)

@app.route(f'/{BOT_TOKEN}', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return 'OK', 200
    return 'Invalid', 403

# === STARTUP ===
if __name__ == '__main__':
    log.info("KickVision v1.0.0 STARTED — Official Release")
    
    bot.remove_webhook()
    time.sleep(1)
    webhook_url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/{BOT_TOKEN}"
    bot.set_webhook(url=webhook_url)
    log.info(f"Webhook set: {webhook_url}")

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
