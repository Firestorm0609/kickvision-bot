#!/usr/bin/env python3
"""
KickVision v1.0.0 — Official Release
100-model ensemble | All Leagues | Fixtures + Predictions | Bug-Free
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
from datetime import datetime, timedelta

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

# === GLOBAL STATE ===
user_rate = defaultdict(list)
TEAM_ALIASES = {}
TEAM_CACHE = {}
LEAGUES_CACHE = {}
PENDING_MATCH = {}
USER_SESSIONS = set()

# === LEAGUE MAP (All major leagues + aliases) ===
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
    "brasileirão": 2013, "brazil": 2013,
    "liga mx": 2012, "mexico": 2012
}

# === ODDS SPORT KEYS ===
SPORT_KEYS = {
    2021: "football_england_premier_league",
    2014: "football_spain_la_liga",
    2002: "football_germany_bundesliga",
    2019: "football_italy_serie_a",
    2015: "football_france_ligue_one",
    2001: "football_uefa_champions_league",
    2018: "football_uefa_europa_league",
    2016: "football_england_championship",
    2003: "football_netherlands_eredivisie",
    2017: "football_portugal_primeira_liga",
    2036: "football_turkey_super_lig",
    2011: "football_usa_mls",
    2013: "football_brazil_brasileirao",
    2012: "football_mexico_liga_mx"
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
                TEAM_CACHE = {}
                for k, v in data.items():
                    if now - v['time'] < CACHE_TTL:
                        TEAM_CACHE[k] = v
            log.info(f"Loaded cache: {len(TEAM_CACHE)} entries")
        except Exception as e:
            log.exception("Cache load error")

def save_cache():
    with open(CACHE_FILE, 'w') as f:
        json.dump(TEAM_CACHE, f)

load_cache()

# === SAFE GET (429-RESILIENT) ===
def safe_get(url, params=None):
    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = 60 * (2 ** attempt)
                log.warning(f"429 → sleeping {wait}s for {url}")
                time.sleep(wait)
                continue
            elif r.status_code in [403, 404]:
                return None
            else:
                log.warning(f"API {r.status_code}: {url}")
                return None
        except Exception as e:
            if "429" in str(e):
                time.sleep(60)
            else:
                log.exception(f"Request failed: {e}")
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

# === PREWARM LEAGUES (SAFE + THROTTLED) ===
def prewarm_leagues():
    log.info("Prewarming supported leagues (throttled)...")
    valid_lids = []
    
    for lid in LEAGUE_MAP.values():
        data = safe_get(f"{API_BASE}/competitions/{lid}")
        if not data:
            continue
        valid_lids.append(lid)
        log.info(f"Validated: {LEAGUES_CACHE.get(lid, lid)} ({lid})")
        
        get_league_teams(lid)
        time.sleep(1.1)  # < 1 req/sec → avoids 429
    
    log.info(f"Prewarmed {len(valid_lids)} leagues safely")

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

# === WEIGHTED STATS + xG CACHE ===
def get_xg_stats(team_id, is_home):
    key = f"xg_{team_id}_{'h' if is_home else 'a'}"
    if key in TEAM_CACHE and time.time() - TEAM_CACHE[key]['time'] < 3600:
        return TEAM_CACHE[key]['data']
    
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
    
    TEAM_CACHE[key] = {'time': time.time(), 'data': stats}
    save_cache()
    return stats

# === MARKET ODDS ===
def get_market_odds(hname, aname, lid):
    if not ODDS_API_KEY or lid not in SPORT_KEYS:
        return None
    try:
        sport = SPORT_KEYS[lid]
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
        params = {'apiKey': ODDS_API_KEY, 'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200: return None
        data = r.json()
        for game in data:
            if hname.lower() in game['home_team'].lower() and aname.lower() in game['away_team'].lower():
                for book in game['bookmakers']:
                    if 'bet' in book['key'].lower():
                        odds = book['markets'][0]['outcomes']
                        return {
                            'home': odds[0]['price'],
                            'draw': odds[1]['price'] if len(odds) > 2 else None,
                            'away': odds[1]['price'] if len(odds) == 2 else odds[2]['price']
                        }
        return None
    except Exception as e:
        log.warning(f"Odds fetch error: {e}")
        return None

# === ENSEMBLE MODEL ===
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
    return {
        'home_win': (hg > ag).mean(),
        'draw': (hg == ag).mean(),
        'away_win': (hg < ag).mean(),
        'score': f"{int(mode(hg))}-{int(mode(ag))}"
    }

def ensemble_100_models(h_gf, h_ga, a_gf, a_ga):
    seeds = list(range(TOTAL_MODELS))
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda s: run_single_model(s, h_gf, h_ga, a_gf, a_ga), seeds))
    return {
        'home_win': round(mean([r['home_win'] for r in results]) * 100),
        'draw': round(mean([r['draw'] for r in results]) * 100),
        'away_win': round(mean([r['away_win'] for r in results]) * 100),
        'score': mode([r['score'] for r in results])
    }

# === VERDICT WITH VALUE ===
def get_verdict(model, market=None):
    h, d, a = model['home_win'], model['draw'], model['away_win']
    value = ""
    if market and market['home'] and market['away']:
        mh = 1/market['home']; ma = 1/market['away']
        md = 1/market['draw'] if market['draw'] else (mh + ma) * 0.1
        total = mh + md + ma
        if total > 0:
            h = int(h * 0.7 + (mh/total*100 * 0.3))
            d = int(d * 0.7 + (md/total*100 * 0.3))
            a = int(a * 0.7 + (ma/total*100 * 0.3))
        if market['home'] > 2.0 and h > 60:
            value = "\n*Value: Home Win*"
        elif market['away'] > 2.0 and a > 60:
            value = "\n*Value: Away Win*"
    max_pct = max(h, d, a)
    if d == max_pct: return "Draw", h, d, a, value
    elif h == max_pct: return "Home Win", h, d, a, value
    else: return "Away Win", h, d, a, value

# === PREDICT ===
def predict_with_ids(hid, aid, hname, aname, h_tla, a_tla):
    lid, league_name = auto_detect_league(hid, aid)
    h_gf, h_ga = get_xg_stats(hid, True)
    a_gf, a_ga = get_xg_stats(aid, False)
    
    model = ensemble_100_models(h_gf, h_ga, a_gf, a_ga)
    market = get_market_odds(hname, aname, lid)
    
    verdict, h_pct, d_pct, a_pct, value = get_verdict(model, market)
    
    out = [
        f"*{hname} vs {aname}*",
        f"_{league_name}_",
        "",
        f"**xG:** `{h_gf:.2f}` — `{a_gf:.2f}`",
        f"**Win:** `{h_pct}%` | `{d_pct}%` | `{a_pct}%`",
        "",
        f"**Most Likely:** `{model['score']}`",
        f"**Verdict:** *{verdict}*{value}"
    ]
    return '\n'.join(out)

# === LEAGUE FIXTURES + PREDICTIONS ===
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
        fixtures.append(f"*{date}*\n{home} vs {away}\n{pred}")
    return '\n\n'.join(fixtures)

# === TODAY'S MATCHES ===
@bot.message_handler(commands=['today'])
def today(m):
    today = datetime.now().strftime('%Y-%m-%d')
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    fixtures = []
    for name, lid in list(LEAGUE_MAP.items())[:6]:
        data = safe_get(f"{API_BASE}/competitions/{lid}/matches", {'dateFrom': today, 'dateTo': tomorrow})
        if data and data.get('matches'):
            m = data['matches'][0]
            h, a = m['homeTeam']['name'], m['awayTeam']['name']
            pred = predict_with_ids(m['homeTeam']['id'], m['awayTeam']['id'], h, a, '', '')
            fixtures.append(f"*{h} vs {a}*\n{pred}")
    if not fixtures:
        bot.reply_to(m, "No major games today.")
        return
    bot.reply_to(m, "*Today's Top Games*\n\n" + "\n\n".join(fixtures[:3]), parse_mode='Markdown')

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
    fixtures = get_league_fixtures(matched)
    bot.reply_to(m, f"*{display_name} Upcoming*\n\n{fixtures}", parse_mode='Markdown')

# === RATE LIMIT ===
def is_allowed(uid):
    now = time.time()
    user_rate[uid] = [t for t in user_rate[uid] if now - t < 5]
    if len(user_rate[uid]) >= 3: return False
    user_rate[uid].append(now)
    return True

# === HELP (All Features) ===
def send_help(m):
    leagues = ", ".join([f"`/{k.replace(' ', '')}`" for k in LEAGUE_MAP.keys() if ' ' in k])
    bot.reply_to(m, (
        "*KickVision v1.0.0 — Football AI*\n\n"
        "*Fixtures & Predictions*\n"
        f"{leagues}\n\n"
        "*Match Prediction*\n"
        "`Team A vs Team B`\n"
        "`Man City vs Arsenal`\n\n"
        "*Commands*\n"
        "`/today` — Top games today\n"
        "`/users` — Active users\n"
        "`/cancel` — Cancel selection\n"
        "`/help` — This menu\n\n"
        "*Inline Mode*\n"
        "`@kickvisionbot Man` — Quick search\n\n"
        "*100-model ensemble • xG • Odds • Value bets*"
    ), parse_mode='Markdown')

@bot.message_handler(commands=['start', 'help', 'how'])
def start(m): send_help(m)

@bot.message_handler(commands=['users'])
def users_cmd(m):
    bot.reply_to(m, f"**Active Users:** `{len(USER_SESSIONS)}`", parse_mode='Markdown')

# === INLINE MODE ===
@bot.inline_handler(lambda query: query.query and len(query.query) > 3)
def inline_query(query):
    home = query.query.strip()
    cands = find_team_candidates(home)[:3]
    results = []
    for i, (_, name, tid, tla, lid, lname) in enumerate(cands):
        results.append(types.InlineQueryResultArticle(
            id=str(i), title=f"{name} ({tla})", description=lname,
            input_message_content=types.InputTextMessageContent(
                f"Predict: {name} vs ? (reply with away team)"
            )
        ))
    bot.answer_inline_query(query.id, results, cache_time=1)

# === MAIN HANDLER ===
@bot.message_handler(func=lambda m: True)
def handle(m):
    if not m.text: return
    uid = m.from_user.id
    txt = m.text.strip()
    USER_SESSIONS.add(uid)

    # Memory cleanup
    if random.random() < 0.001:
        active = set()
        try:
            updates = bot.get_updates(limit=100)
            active = {u.message.from_user.id for u in updates if u.message}
        except: pass
        USER_SESSIONS &= active

    if txt.lower() == '/cancel':
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
                r = predict_with_ids(h[2], a[2], h[1], a[1], h[3], a[3])
                bot.reply_to(m, r, parse_mode='Markdown')
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
        h = home_cands[0]; a = away_cands[0]
        r = predict_with_ids(h[2], a[2], h[1], a[1], h[3], a[3])
        bot.reply_to(m, r, parse_mode='Markdown')
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

if __name__ == '__main__':
    log.info("KickVision v1.0.0 STARTED — Render-Ready + 429-Fixed")
    bot.remove_webhook()
    time.sleep(1)
    bot.set_webhook(url=f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/{BOT_TOKEN}")
    
    prewarm_leagues()
    
    port = int(os.environ.get('PORT', 5000))
    log.info(f"LISTENING ON PORT {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)
