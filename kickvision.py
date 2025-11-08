#!/usr/bin/env python3
# KickVision v1.0.1 - Optimized Telebot + Flask for Render
# Features: /start with league selection, today’s fixtures + predictions, top scorers, standings with pagination, daily results broadcast

import os, time, json, logging, random
from datetime import datetime, date, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import requests
import telebot
from telebot import types
from flask import Flask, request

# ===== CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("API_KEY")
API_BASE = 'https://api.football-data.org/v4'
CACHE_FILE = 'team_cache.json'
CACHE_TTL = 3600
TOTAL_MODELS = 50
PREDICTION_TIME = "00:00"  # UTC daily broadcast

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('kickvision')
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# ===== GLOBAL STATE =====
TEAM_CACHE = {}
USER_SESSIONS = set()
PREDICTION_CACHE = {}
USER_HISTORY = defaultdict(list)

# ===== TELEBOT =====
bot = telebot.TeleBot(BOT_TOKEN)
time.sleep(1)

# ===== FLASK APP =====
app = Flask(__name__)

# ===== UTILITIES =====
def safe_get(url, params=None):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers={'X-Auth-Token': API_KEY}, timeout=10)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                log.warning(f"API {r.status_code}: {url}")
        except Exception as e:
            log.warning(f"Request failed: {e}")
            time.sleep(2)
    return None

# ===== LEAGUES =====
LEAGUE_MAP = {
    "Premier League": 2021,
    "La Liga": 2014,
    "Serie A": 2019,
    "Bundesliga": 2002,
    "Ligue 1": 2015,
    "Champions League": 2001
}
LEAGUE_DISPLAY = {v: k for k, v in LEAGUE_MAP.items()}

# ===== CACHE =====
def load_cache():
    global TEAM_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                TEAM_CACHE = json.load(f)
                log.info(f"Loaded cache with {len(TEAM_CACHE)} entries")
        except:
            TEAM_CACHE = {}

def save_cache():
    with open(CACHE_FILE, 'w') as f:
        json.dump(TEAM_CACHE, f)

load_cache()

# ===== ENSEMBLE PREDICTIONS =====
def run_single_model(seed, h_gf, h_ga, a_gf, a_ga):
    random.seed(seed)
    hg = [max(0, int(random.gauss(h_gf,1))) for _ in range(1)]
    ag = [max(0, int(random.gauss(a_gf,1))) for _ in range(1)]
    return hg, ag

def ensemble_prediction(h_gf,h_ga,a_gf,a_ga):
    all_h, all_a = [], []
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = executor.map(lambda s: run_single_model(s,h_gf,h_ga,a_gf,a_ga), range(TOTAL_MODELS))
        for hg,ag in results:
            all_h.extend(hg)
            all_a.extend(ag)
    home_win = sum(1 for h,a in zip(all_h,all_a) if h>a)/len(all_h)*100
    draw = sum(1 for h,a in zip(all_h,all_a) if h==a)/len(all_h)*100
    away_win = sum(1 for h,a in zip(all_h,all_a) if h<a)/len(all_h)*100
    outcome = "Home" if home_win >= max(draw,away_win) else "Away" if away_win >= max(draw,home_win) else "Draw"
    return round(home_win,1), round(draw,1), round(away_win,1), outcome

# ===== TEAM STATS =====
def get_team_stats(team_id,is_home=True):
    data = safe_get(f"{API_BASE}/teams/{team_id}/matches",{'status':'FINISHED','limit':8})
    if not data or len(data.get('matches',[]))<3:
        return (1.5,1.2) if is_home else (1.2,1.5)
    gf = sum(m['score']['fullTime']['home' if is_home else 'away'] for m in data['matches'])/len(data['matches'])
    ga = sum(m['score']['fullTime']['away' if is_home else 'home'] for m in data['matches'])/len(data['matches'])
    return gf,ga

# ===== TODAY'S FIXTURES =====
def get_today_fixtures(league_id):
    today = date.today().isoformat()
    data = safe_get(f"{API_BASE}/competitions/{league_id}/matches",{'dateFrom':today,'dateTo':today})
    if not data or 'matches' not in data:
        return []
    fixtures = []
    for m in data['matches']:
        hid = m['homeTeam']['id']
        aid = m['awayTeam']['id']
        hname = m['homeTeam']['name']
        aname = m['awayTeam']['name']
        t = m['utcDate'][11:16]
        h_gf,h_ga = get_team_stats(hid,True)
        a_gf,a_ga = get_team_stats(aid,False)
        home,draw,away,outcome = ensemble_prediction(h_gf,h_ga,a_gf,a_ga)
        fixtures.append(f"{hname} vs {aname} @ {t} UTC\nHome {home}% | Draw {draw}% | Away {away}%\nPossible Outcome: {outcome}\nLeague: {LEAGUE_DISPLAY[league_id]}")
    return fixtures

# ===== TOP SCORERS =====
def get_top_scorers(league_id,page=1,per_page=10):
    data = safe_get(f"{API_BASE}/competitions/{league_id}/scorers")
    if not data or 'scorers' not in data:
        return ["No data"]
    start = (page-1)*per_page
    end = start+per_page
    text = f"🏆 Top Scorers - {LEAGUE_DISPLAY[league_id]}\n"
    for idx, s in enumerate(data['scorers'][start:end],start=start+1):
        text += f"{idx}. {s['player']['name']} ({s['team']['name']}) - {s['goals']} goals\n"
    if end < len(data['scorers']):
        text += f"\nUse /topscorers_{league_id}_{page+1} for next page"
    return [text]

# ===== STANDINGS =====
def get_standings(league_id,page=1,per_page=10):
    data = safe_get(f"{API_BASE}/competitions/{league_id}/standings")
    if not data or 'standings' not in data:
        return ["No data"]
    table = data['standings'][0]['table']
    start = (page-1)*per_page
    end = start+per_page
    text = f"🏆 Standings - {LEAGUE_DISPLAY[league_id]}\n"
    for pos, team in enumerate(table[start:end],start=start+1):
        text += f"{pos}. {team['team']['name']} - P:{team['playedGames']} W:{team['won']} D:{team['draw']} L:{team['lost']} GF:{team['goalsFor']} GA:{team['goalsAgainst']} Pts:{team['points']}\n"
    if end < len(table):
        text += f"\nUse /standings_{league_id}_{page+1} for next page"
    return [text]

# ===== INLINE MENUS =====
def main_menu():
    markup = types.InlineKeyboardMarkup(row_width=2)
    buttons = [types.InlineKeyboardButton(text=k, callback_data=f"league_{v}") for k,v in LEAGUE_MAP.items()]
    markup.add(*buttons)
    markup.add(types.InlineKeyboardButton("Top Scorers ⚽️", callback_data="topscorers"))
    markup.add(types.InlineKeyboardButton("Standings 📊", callback_data="standings"))
    return markup

# ===== CALLBACK HANDLER =====
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    if call.data.startswith("league_"):
        lid = int(call.data.split("_")[1])
        fixtures = get_today_fixtures(lid)
        text = "\n\n".join(fixtures) if fixtures else "No matches today in this league."
        bot.send_message(call.message.chat.id,text)
    elif call.data == "topscorers":
        for lid in LEAGUE_MAP.values():
            for text in get_top_scorers(lid):
                bot.send_message(call.message.chat.id,text)
    elif call.data == "standings":
        for lid in LEAGUE_MAP.values():
            for text in get_standings(lid):
                bot.send_message(call.message.chat.id,text)

# ===== START =====
@bot.message_handler(commands=['start'])
def start(m):
    USER_SESSIONS.add(m.from_user.id)
    bot.send_message(m.chat.id,"Welcome! Select a league to view today's fixtures and predictions:",reply_markup=main_menu())

# ===== DAILY RESULTS BROADCAST =====
def broadcast_results():
    for uid in USER_SESSIONS:
        bot.send_message(uid,"📊 Daily Results & Prediction Accuracy will appear here.")
    # schedule next call using external cron/render scheduler

# ===== FLASK WEBHOOK =====
@app.route(f'/{BOT_TOKEN}', methods=['POST'])
def webhook():
    if request.headers.get('content-type')=='application/json':
        update = telebot.types.Update.de_json(request.get_data().decode('utf-8'))
        bot.process_new_updates([update])
        return 'OK',200
    return 'Invalid',403

@app.route('/')
def index():
    return "KickVision Bot v1.0.1 is running!"

# ===== RUN =====
if __name__=="__main__":
    log.info("KickVision Bot v1.0.1 ready")
    bot.remove_webhook()
    time.sleep(1)
    bot.set_webhook(url=f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/{BOT_TOKEN}")
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT',5000)))
