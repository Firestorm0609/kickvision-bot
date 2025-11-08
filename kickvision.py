#!/usr/bin/env python3
"""
KickVision v1.3.0 — Enhanced Free Edition
Live scores | FPL | Results comparison | Top scorers | Standings | Better UX
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
from datetime import datetime, date

import numpy as np
import requests
import difflib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import telebot
from telebot import types
from flask import Flask, request

# ============================= CONFIG =============================
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

# ============================= LOGGING =============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger('kickvision')
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# ============================= STATE =============================
user_rate = defaultdict(list)
TEAM_ALIASES = {}
TEAM_CACHE = {}
LEAGUES_CACHE = {}
PENDING_MATCH = {}
USER_SESSIONS = set()
ODDS_CACHE = {}
LOADING_MSGS = {}
HELP_STATE = {}
CANCEL_ALL = set()

# Caches
PREDICTION_CACHE = {}
TEAM_RESOLVE_CACHE = {}
USER_HISTORY = defaultdict(list)
LEAGUES_LOADED = {}
STANDINGS_CACHE = {}
SCORERS_CACHE = {}
MATCHDAY_CACHE = {}
LIVE_SCORES_CACHE = {}
FPL_CACHE = {}
RESULTS_CACHE = {}

CACHE_DURATIONS = {
    'standings': 7200, 'scorers': 43200, 'matchday': 3600,
    'predictions': 3600, 'live': 300, 'fpl': 1800, 'results': 3600
}

# Educational & previews
EDUCATIONAL_TIPS = [...]
MATCH_PREVIEWS = [...]

# ============================= LEAGUE MAP =============================
LEAGUE_MAP = { ... }  # same as yours
LEAGUE_DISPLAY_NAMES = { ... }

# ============================= ALIASES =============================
# (same as before – unchanged)

# ============================= HTTP & TELEBOT =============================
session = requests.Session()
session.headers.update({'X-Auth-Token': API_KEY})
retries = Retry(total=5, backoff_factor=2, status_forcelist=[429,500,502,503,504])
session.mount('https://', HTTPAdapter(max_retries=retries))

bot = telebot.TeleBot(BOT_TOKEN)
time.sleep(2)

# ============================= CACHE =============================
# (load_cache, save_cache – unchanged)

# ============================= FAST RESOLVE =============================
def fast_resolve_alias(name): ...  # same

# ============================= SAFE GET =============================
def safe_get(url, params=None): ...  # same

# ============================= LEAGUES CACHE =============================
# (load_leagues_cache, fetch_all_leagues – unchanged)

# ============================= TEAM LOADING =============================
def get_league_teams_lazy(lid): ...  # same
def get_league_teams(lid): ...  # same

# ============================= CANDIDATES =============================
def find_team_candidates(name): ...  # same

# ============================= LEAGUE DETECT =============================
def auto_detect_league(hid, aid): ...  # same

# ============================= STATS =============================
def get_weighted_stats(tid, is_home): ...  # same

# ============================= ODDS =============================
def get_market_odds(hn, an): ...  # same

# ============================= SIMULATION =============================
def run_single_model(seed, h_gf, h_ga, a_gf, a_ga):
    random.seed(seed); np.random.seed(seed)
    hx = (h_gf * a_ga * 1.1) ** 0.5 * random.uniform(0.9, 1.1)
    ax = (a_gf * h_ga * 0.9) ** 0.5 * random.uniform(0.9, 1.1)
    if hx < 2.0 and ax < 2.0:
        tau = 1 - 0.05 * hx * ax
        hx *= tau; ax *= tau
    return np.random.poisson(hx, SIMS_PER_MODEL), np.random.poisson(ax, SIMS_PER_MODEL)

def ensemble_models(h_gf, h_ga, a_gf, a_ga):
    seeds = range(TOTAL_MODELS)
    all_h, all_a = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for hg, ag in ex.map(lambda s: run_single_model(s, h_gf, h_ga, a_gf, a_ga), seeds):
            all_h.extend(hg); all_a.extend(ag)
    total = len(all_h)
    hw = sum(1 for h,a in zip(all_h, all_a) if h > a) / total
    draw = sum(1 for h,a in zip(all_h, all_a) if h == a) / total
    aw = 1 - hw - draw
    counts = Counter(zip(all_h, all_a))
    most = counts.most_common(1)[0][0]
    return {
        'home_win': round(hw * 100),
        'draw': round(draw * 100),
        'away_win': round(aw * 100),
        'score': f"{most[0]}-{most[1]}"
    }

# ============================= VERDICT =============================
def get_verdict(model, market=None): ...  # same

# ============================= CACHED PREDICTION =============================
def cached_prediction(hid, aid, hname, aname, h_tla, a_tla):
    key = f"pred_{hid}_{aid}"
    now = time.time()
    if key in PREDICTION_CACHE and now - PREDICTION_CACHE[key]['time'] < CACHE_DURATIONS['predictions']:
        return PREDICTION_CACHE[key]['data']
    lid, lname = auto_detect_league(hid, aid)
    h_gf, h_ga = get_weighted_stats(hid, True)
    a_gf, a_ga = get_weighted_stats(aid, False)
    model = ensemble_models(h_gf, h_ga, a_gf, a_ga)
    market = get_market_odds(hname, aname)
    verdict, hp, dp, ap = get_verdict(model, market)
    result = '\n'.join([
        f"*{hname} vs {aname}*",
        f"_{lname}_",
        "",
        f"**xG:** `{h_gf:.2f}` — `{a_gf:.2f}`",
        f"**Win:** `{hp}%` | `{dp}%` | `{ap}%`",
        "",
        f"**Most Likely:** `{model['score']}`",
        f"**Verdict:** *{verdict}*"
    ])
    PREDICTION_CACHE[key] = {'time': now, 'data': result}
    return result

def predict_with_ids(hid, aid, hname, aname, h_tla, a_tla):
    return cached_prediction(hid, aid, hname, aname, h_tla, a_tla)

# ============================= HISTORY =============================
def add_to_history(uid, match, pred): ...  # same
def get_user_history(uid): ...  # same

# ============================= LOADING ANIMATION =============================
def fun_loading(chat_id, base_text="Loading", reply_to_id=None, stages_count=3):
    stages = [...]  # your list
    random.shuffle(stages)
    msg = bot.send_message(chat_id, f"{base_text}...", reply_to_message_id=reply_to_id, parse_mode='Markdown')
    for stage in stages[:stages_count]:
        time.sleep(random.uniform(0.9, 1.4))
        try: bot.edit_message_text(stage, chat_id, msg.message_id, parse_mode='Markdown')
        except: pass
    return msg

# ============================= NEW FEATURES =============================
def get_live_scores():
    key = "live_scores"
    now = time.time()
    if key in LIVE_SCORES_CACHE and now - LIVE_SCORES_CACHE[key]['time'] < CACHE_DURATIONS['live']:
        return LIVE_SCORES_CACHE[key]['data']
    data = safe_get(f"{API_BASE}/matches", {'status': 'LIVE'})
    if not data or not data.get('matches'): return "No live matches."
    lines = ["*Live Scores*"]
    for m in data['matches'][:10]:
        h = m['homeTeam']['name']; a = m['awayTeam']['name']
        s = m['score']['fullTime']
        lines.append(f"`{s['home']}-{s['away']}` *{h} vs {a}*")
    result = '\n'.join(lines)
    LIVE_SCORES_CACHE[key] = {'time': now, 'data': result}
    return result

def get_fpl_data():
    key = "fpl_data"
    now = time.time()
    if key in FPL_CACHE and now - FPL_CACHE[key]['time'] < CACHE_DURATIONS['fpl']:
        return FPL_CACHE[key]['data']
    try:
        r = requests.get(f"{FPL_BASE}/bootstrap-static/", timeout=10)
        data = r.json()
        top = data['elements'][0]
        name = f"{top['first_name']} {top['second_name']}"
        pts = top['total_points']
        result = f"*FPL Top Player*\n\n{name}\n**Points:** {pts}"
        FPL_CACHE[key] = {'time': now, 'data': result}
        return result
    except: return "FPL data unavailable."

def get_todays_results_with_comparison():
    key = "results"
    now = time.time()
    if key in RESULTS_CACHE and now - RESULTS_CACHE[key]['time'] < CACHE_DURATIONS['results']:
        return RESULTS_CACHE[key]['data']
    today = date.today().isoformat()
    lines = ["*Today's Results vs KickVision*"]
    for lid in [2021, 2014, 2002, 2019, 2015]:
        data = safe_get(f"{API_BASE}/competitions/{lid}/matches", {'dateFrom': today, 'dateTo': today, 'status': 'FINISHED'})
        if not data or not data.get('matches'): continue
        for m in data['matches']:
            h = m['homeTeam']['name']; a = m['awayTeam']['name']
            hid = m['homeTeam']['id']; aid = m['awayTeam']['id']
            s = m['score']['fullTime']
            if not s['home'] or not s['away']: continue
            pred = predict_with_ids(hid, aid, h, a, '', '')
            verdict = [l for l in pred.split('\n') if "Verdict:" in l][0].split("*")[1]
            actual = "Home Win" if s['home'] > s['away'] else "Draw" if s['home'] == s['away'] else "Away Win"
            icon = "Correct" if verdict == actual else "Incorrect"
            lines.append(f"{icon} `{h} {s['home']}-{s['away']} {a}` — *{verdict}*")
    result = '\n'.join(lines[:15])
    RESULTS_CACHE[key] = {'time': now, 'data': result}
    return result

# ============================= COMMANDS =============================
@bot.message_handler(commands=['live'])
def live_cmd(m):
    loading = fun_loading(m.chat.id, "Live scores...", m.message_id)
    bot.edit_message_text(get_live_scores(), m.chat.id, loading.message_id, parse_mode='Markdown')

@bot.message_handler(commands=['fpl'])
def fpl_cmd(m):
    loading = fun_loading(m.chat.id, "FPL data...", m.message_id)
    bot.edit_message_text(get_fpl_data(), m.chat.id, loading.message_id, parse_mode='Markdown')

@bot.message_handler(commands=['results'])
def results_cmd(m):
    loading = fun_loading(m.chat.id, "Results...", m.message_id)
    bot.edit_message_text(get_todays_results_with_comparison(), m.chat.id, loading.message_id, parse_mode='Markdown')

# ============================= /start MENU =============================
@bot.message_handler(commands=['start'])
def start(m):
    USER_SESSIONS.add(m.from_user.id)
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("Today", callback_data="cmd_/today"),
        types.InlineKeyboardButton("Live", callback_data="cmd_/live"),
        types.InlineKeyboardButton("Results", callback_data="cmd_/results"),
        types.InlineKeyboardButton("FPL", callback_data="cmd_/fpl"),
        types.InlineKeyboardButton("Premier League", callback_data="cmd_/premierleague"),
        types.InlineKeyboardButton("La Liga", callback_data="cmd_/laliga"),
        types.InlineKeyboardButton("Help", callback_data="help_1")
    )
    bot.send_message(m.chat.id, "*KickVision v1.3.0*\nEnhanced Free Edition", reply_markup=markup, parse_mode='Markdown')

# ============================= CALLBACK HANDLER =============================
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    if not call.from_user: return
    uid = call.from_user.id
    if uid in CANCEL_ALL: return
    bot.answer_callback_query(call.id)
    data = call.data

    fake_msg = types.Message()
    fake_msg.message_id = call.message.message_id
    fake_msg.from_user = call.from_user
    fake_msg.chat = call.message.chat
    fake_msg.text = data[5:] if data.startswith("cmd_/") else None
    fake_msg.date = call.message.date

    if data.startswith("cmd_/"):
        cmd = data[5:]
        if cmd == "/today": today_handler(fake_msg)
        elif cmd == "/live": live_cmd(fake_msg)
        elif cmd == "/results": results_cmd(fake_msg)
        elif cmd == "/fpl": fpl_cmd(fake_msg)
        else: dynamic_league_handler(fake_msg)
    elif data.startswith("help_"):
        show_help_page(call.message, int(data.split("_")[1]))

# ============================= WEBHOOK =============================
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
    return 'KickVision v1.3.0 Running'

if __name__ == '__main__':
    log.info("KickVision v1.3.0 STARTED")
    bot.remove_webhook()
    time.sleep(1)
    bot.set_webhook(url=f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/{BOT_TOKEN}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
