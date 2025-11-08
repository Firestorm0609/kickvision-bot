#!/usr/bin/env python3
"""
KickVision v1.0.0 - Telebot + Flask build (webhook)
- Requires: pip install pyTelegramBotAPI flask requests
- Env vars:
    BOT_TOKEN
    API_KEY            (football-data.org)
    RENDER_EXTERNAL_HOSTNAME (optional, used to set webhook)
    ADMIN_CHAT_ID      (optional, comma-separated chat ids to receive end-of-day results)
"""

import os, time, math, random, logging, json
from datetime import date, timedelta, datetime
from collections import defaultdict
from flask import Flask, request
import requests

# Telebot import
try:
    import telebot
    from telebot import types
except Exception as e:
    raise RuntimeError("Install pyTelegramBotAPI (pip install pyTelegramBotAPI)")

# Config
VERSION = "1.0.0"
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("API_KEY")
RENDER_HOST = os.getenv("RENDER_EXTERNAL_HOSTNAME")
PORT = int(os.getenv("PORT", 5000))
ADMIN_CHAT_IDS = [int(x) for x in os.getenv("ADMIN_CHAT_ID","").split(",") if x.strip()]

if not BOT_TOKEN or not API_KEY:
    raise RuntimeError("BOT_TOKEN and API_KEY env vars are required")

API_BASE = "https://api.football-data.org/v4"
FPL_BASE = "https://fantasy.premierleague.com/api"

# Focus leagues
LEAGUES = {
    "premier league": 2021,
    "champions league": 2001,
    "la liga": 2014,
    "bundesliga": 2002,
    "serie a": 2019,
    "ligue 1": 2015
}
LEAGUE_DISPLAY = {v: k.title() for k, v in LEAGUES.items()}

# Caching
CACHE = {}
TTL = {
    'fixtures': 60*5,
    'teams': 60*60*12,
    'standings': 60*30,
    'scorers': 60*30,
    'predictions': 60*60,
    'fpl': 60*30,
    'live': 60*1
}

# Prediction ensemble params
TOTAL_MODELS = 100
SIMS_PER_MODEL = 80   # small for render speed

# Logging & session
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("kickvision")
session = requests.Session()
session.headers.update({"X-Auth-Token": API_KEY, "User-Agent": "KickVision/1.0.0"})

# helpers
def cache_get(k):
    e = CACHE.get(k)
    if not e: return None
    val, t, ttl = e
    if time.time() - t > ttl:
        del CACHE[k]; return None
    return val

def cache_set(k, v, ttl):
    CACHE[k] = (v, time.time(), ttl)

def safe_get(url, params=None, timeout=10):
    try:
        r = session.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            log.warning("Rate limited by API")
            return None
        else:
            log.debug(f"API {r.status_code} {r.text[:200]}")
            return None
    except Exception as e:
        log.exception("Request failed")
        return None

# football-data wrappers
def get_upcoming_fixtures(league_id, days=7, limit=6):
    key = f"fixtures_{league_id}_{days}"
    cached = cache_get(key)
    if cached: return cached
    start = date.today().isoformat()
    end = (date.today()+timedelta(days=days)).isoformat()
    data = safe_get(f"{API_BASE}/competitions/{league_id}/matches", params={"status":"SCHEDULED","dateFrom":start,"dateTo":end})
    matches = data.get("matches", [])[:limit] if data else []
    cache_set(key, matches, TTL['fixtures'])
    return matches

def get_team_recent_stats(team_id, last=6):
    # return (gf_avg, ga_avg) falling back to league average
    key = f"team_stats_{team_id}_{last}"
    cached = cache_get(key)
    if cached: return cached
    data = safe_get(f"{API_BASE}/teams/{team_id}/matches", params={"status":"FINISHED","limit":last})
    if not data or "matches" not in data:
        res = (1.3, 1.3)
        cache_set(key, res, TTL['teams'])
        return res
    gf = []; ga = []
    for m in data["matches"]:
        score = m.get("score", {}).get("fullTime", {})
        if m["homeTeam"]["id"] == team_id:
            gf.append(score.get("home",0)); ga.append(score.get("away",0))
        else:
            gf.append(score.get("away",0)); ga.append(score.get("home",0))
    if not gf:
        res=(1.3,1.3)
    else:
        res=(sum(gf)/len(gf), sum(ga)/len(ga))
    cache_set(key, res, TTL['teams'])
    return res

# Lightweight ensemble
def _poisson_sample(lam):
    # Knuth
    L = math.exp(-lam)
    k=0; p=1.0
    while p> L:
        k+=1; p *= random.random()
    return k-1

def simulate_model(h_lambda,a_lambda,sims=100,seed=None):
    if seed is not None: random.seed(seed)
    hw=0; dw=0; aw=0
    for _ in range(sims):
        hg = _poisson_sample(h_lambda); ag = _poisson_sample(a_lambda)
        if hg>ag: hw+=1
        elif hg==ag: dw+=1
        else: aw+=1
    tot=float(sims)
    return {'home': hw/tot, 'draw': dw/tot, 'away': aw/tot}

def estimate_lambdas(h_stats,a_stats):
    # h_stats=(gf,ga), a_stats=(gf,ga)
    h_attack = max(0.1, h_stats[0]); h_def = max(0.1, h_stats[1])
    a_attack = max(0.1, a_stats[0]); a_def = max(0.1, a_stats[1])
    home_lambda = (h_attack * a_def) / 2.5 * 1.08
    away_lambda = (a_attack * h_def) / 2.5 * 0.95
    return max(0.05,home_lambda), max(0.05,away_lambda)

def ensemble_predict(h_stats,a_stats):
    key = f"pred_{h_stats}_{a_stats}"
    cached = cache_get(key)
    if cached: return cached
    homes=[]; draws=[]; aways=[]
    base_h, base_a = estimate_lambdas(h_stats,a_stats)
    for m in range(TOTAL_MODELS):
        jitter_h = random.uniform(0.85,1.15)
        jitter_a = random.uniform(0.85,1.15)
        scale = random.uniform(0.85,1.25)
        h_l = base_h * jitter_h * scale
        a_l = base_a * jitter_a * scale
        res = simulate_model(h_l,a_l,sims=SIMS_PER_MODEL, seed=(m*17 + int(time.time()%1000)))
        homes.append(res['home']); draws.append(res['draw']); aways.append(res['away'])
    final = {'home': sum(homes)/len(homes), 'draw': sum(draws)/len(draws), 'away': sum(aways)/len(aways)}
    cache_set(key, final, TTL['predictions'])
    return final

# store predictions for later comparison
PRED_RECORD = {}  # match_id => prediction

def store_prediction(match_id, pred):
    PRED_RECORD[str(match_id)] = {'pred': pred, 'time': time.time()}

def compare_today_results():
    # gather finished matches today across our leagues and compare
    today = date.today().isoformat()
    total=0; correct=0; details=[]
    for lid in LEAGUES.values():
        data = safe_get(f"{API_BASE}/competitions/{lid}/matches", params={"status":"FINISHED","dateFrom":today,"dateTo":today})
        matches = data.get("matches",[]) if data else []
        for m in matches:
            mid=str(m['id'])
            full=m.get('score',{}).get('fullTime',{})
            hg=full.get('home'); ag=full.get('away')
            if hg is None or ag is None: continue
            actual = 'home' if hg>ag else ('draw' if hg==ag else 'away')
            pred = PRED_RECORD.get(mid)
            if pred:
                p_choice = max(('home','draw','away'), key=lambda k: pred['pred'].get(k,0))
                if p_choice == actual: correct+=1
                total+=1
                details.append({'match': f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}", 'predicted': p_choice, 'actual': actual, 'confidence': pred['pred'].get(p_choice,0)})
    acc = (correct/total*100) if total else None
    return {'total':total,'correct':correct,'accuracy':acc,'details':details}

# Telebot & Flask setup
bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

def start_menu_markup(page=1):
    markup = types.InlineKeyboardMarkup(row_width=2)
    if page==1:
        # first page: leagues
        markup.add(types.InlineKeyboardButton("Premier League", callback_data="league_2021"),
                   types.InlineKeyboardButton("Champions League", callback_data="league_2001"))
        markup.add(types.InlineKeyboardButton("La Liga", callback_data="league_2014"),
                   types.InlineKeyboardButton("Serie A", callback_data="league_2019"))
        markup.add(types.InlineKeyboardButton("Bundesliga", callback_data="league_2002"),
                   types.InlineKeyboardButton("Next ▶", callback_data="menu_page_2"))
    else:
        # page 2
        markup.add(types.InlineKeyboardButton("Today", callback_data="today"),
                   types.InlineKeyboardButton("Live", callback_data="live"))
        markup.add(types.InlineKeyboardButton("Results", callback_data="results"),
                   types.InlineKeyboardButton("Standings", callback_data="standings"))
        markup.add(types.InlineKeyboardButton("FPL", callback_data="fpl"),
                   types.InlineKeyboardButton("Top Scorers", callback_data="scorers"))
        markup.add(types.InlineKeyboardButton("◀ Back", callback_data="menu_page_1"))
    return markup

@bot.message_handler(commands=['start'])
def start_handler(m):
    txt = f"KickVision v{VERSION} — pick a league (page 1) or navigate to tools (page 2)"
    bot.send_message(m.chat.id, txt, reply_markup=start_menu_markup(page=1))

@bot.callback_query_handler(func=lambda c: True)
def callback_handler(call):
    try:
        data = call.data
        if data == "menu_page_2":
            bot.edit_message_text("KickVision — Tools", call.message.chat.id, call.message.message_id, reply_markup=start_menu_markup(page=2))
            return
        if data == "menu_page_1":
            bot.edit_message_text("KickVision — Leagues", call.message.chat.id, call.message.message_id, reply_markup=start_menu_markup(page=1))
            return

        if data.startswith("league_"):
            lid = int(data.split("_",1)[1])
            fixtures = get_upcoming_fixtures(lid, days=7, limit=5)
            if not fixtures:
                bot.answer_callback_query(call.id, "No upcoming fixtures found.")
                return
            out=[]
            for m in fixtures:
                h = m['homeTeam']['name']; a = m['awayTeam']['name']; t = m['utcDate'][11:16]
                h_stats = get_team_recent_stats(m['homeTeam']['id'])
                a_stats = get_team_recent_stats(m['awayTeam']['id'])
                pred = ensemble_predict(h_stats, a_stats)
                store_prediction(m['id'], pred)
                out.append(f"{h} vs {a} @ {t} UTC\nHome {pred['home']*100:.1f}% | Draw {pred['draw']*100:.1f}% | Away {pred['away']*100:.1f}%\n")
            bot.edit_message_text("\n".join(out)[:3900], call.message.chat.id, call.message.message_id)
            return

        if data == "today":
            today = date.today().isoformat(); texts=[]
            for name, lid in LEAGUES.items():
                fixtures = get_upcoming_fixtures(lid, days=0, limit=4)
                if not fixtures: continue
                texts.append(f"🏆 {name.title()} 🏆")
                for m in fixtures:
                    h = m['homeTeam']['name']; a = m['awayTeam']['name']; t = m['utcDate'][11:16]
                    h_stats = get_team_recent_stats(m['homeTeam']['id'])
                    a_stats = get_team_recent_stats(m['awayTeam']['id'])
                    pred = ensemble_predict(h_stats, a_stats)
                    store_prediction(m['id'], pred)
                    texts.append(f"{t} UTC — {h} vs {a}\nHome {pred['home']*100:.1f}% | Draw {pred['draw']*100:.1f}% | Away {pred['away']*100:.1f}%")
            bot.edit_message_text("\n\n".join(texts)[:3900], call.message.chat.id, call.message.message_id)
            return

        if data == "live":
            # simple live fetch
            collected=[]
            for lid in LEAGUES.values():
                data = safe_get(f"{API_BASE}/competitions/{lid}/matches", params={"status":"LIVE"})
                matches = data.get("matches",[]) if data else []
                for m in matches:
                    score = m.get("score",{}).get("fullTime",{})
                    collected.append(f"{m['homeTeam']['name']} {score.get('home','-')} - {score.get('away','-')} {m['awayTeam']['name']}")
            bot.edit_message_text("\n".join(collected) if collected else "No live matches right now.", call.message.chat.id, call.message.message_id)
            return

        if data == "results":
            comp = compare_today_results()
            if comp['total']==0:
                bot.edit_message_text("No finished matches with stored predictions today.", call.message.chat.id, call.message.message_id)
                return
            out=[f"Results comparison — {comp['total']} matches"]
            if comp['accuracy'] is not None:
                out.append(f"Accuracy: {comp['accuracy']:.2f}% ({comp['correct']}/{comp['total']})")
            for d in comp['details'][:10]:
                out.append(f"{d['match']}: predicted {d['predicted']} — actual {d['actual']} (conf {d['confidence']*100:.1f}%)")
            bot.edit_message_text("\n".join(out)[:3900], call.message.chat.id, call.message.message_id)
            return

        if data == "standings":
            # present inline league selection for standings
            markup = types.InlineKeyboardMarkup(row_width=2)
            for name,lid in LEAGUES.items():
                markup.add(types.InlineKeyboardButton(name.title(), callback_data=f"stand_{lid}"))
            markup.add(types.InlineKeyboardButton("Back", callback_data="menu_page_2"))
            bot.edit_message_text("Choose a league for standings:", call.message.chat.id, call.message.message_id, reply_markup=markup)
            return

        if data.startswith("stand_"):
            lid = int(data.split("_",1)[1])
            data = safe_get(f"{API_BASE}/competitions/{lid}/standings")
            if not data:
                bot.answer_callback_query(call.id, "Standings unavailable")
                return
            text = f"{LEAGUE_DISPLAY.get(lid,'League')} Standings\n"
            for s in data.get("standings",[]):
                if s.get("type")=="TOTAL":
                    for row in s.get("table",[])[:10]:
                        text += f"{row['position']}. {row['team']['name']} — {row['points']} pts, GD {row['goalDifference']}\n"
                    break
            bot.edit_message_text(text[:3900], call.message.chat.id, call.message.message_id)
            return

        if data == "fpl":
            b = safe_get(f"{FPL_BASE}/bootstrap-static/")
            if not b:
                bot.edit_message_text("FPL data unavailable", call.message.chat.id, call.message.message_id)
                return
            top = sorted(b.get("elements",[]), key=lambda x: float(x.get("form") or 0), reverse=True)[:8]
            text = "Top FPL form players (sample):\n"
            for p in top:
                text += f"{p.get('web_name')} — form {p.get('form')} — selected {p.get('selected_by_percent')}\n"
            bot.edit_message_text(text[:3900], call.message.chat.id, call.message.message_id)
            return

        if data == "scorers":
            # select league scorers
            markup = types.InlineKeyboardMarkup(row_width=2)
            for name,lid in LEAGUES.items():
                markup.add(types.InlineKeyboardButton(name.title(), callback_data=f"scor_{lid}"))
            markup.add(types.InlineKeyboardButton("Back", callback_data="menu_page_2"))
            bot.edit_message_text("Choose a league for top scorers:", call.message.chat.id, call.message.message_id, reply_markup=markup)
            return

        if data.startswith("scor_"):
            lid = int(data.split("_",1)[1])
            sdata = safe_get(f"{API_BASE}/competitions/{lid}/scorers")
            if not sdata:
                bot.answer_callback_query(call.id,"Scorers unavailable")
                return
            lines=[]
            for sc in sdata.get("scorers",[])[:10]:
                lines.append(f"{sc['player']['name']} — {sc['team']['name']} — {sc['goals']} goals")
            bot.edit_message_text("\n".join(lines)[:3900], call.message.chat.id, call.message.message_id)
            return

        bot.answer_callback_query(call.id, "Unknown action")
    except Exception as e:
        log.exception("Callback failed")
        try:
            bot.answer_callback_query(call.id, "Error processing action")
        except:
            pass

# Flask webhook endpoints
@app.route("/", methods=["GET"])
def index():
    return f"KickVision Telebot v{VERSION} running."

@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    if request.headers.get("content-type") == "application/json":
        update = request.get_data().decode("utf-8")
        telebot.types.Update.de_json(update)  # just ensure formatting
        bot.process_new_updates([telebot.types.Update.de_json(update)])
        return "OK", 200
    return "Invalid", 403

def set_webhook():
    if RENDER_HOST:
        url = f"https://{RENDER_HOST}/{BOT_TOKEN}"
        try:
            bot.remove_webhook()
        except:
            pass
        bot.set_webhook(url=url)
        log.info(f"Webhook set to {url}")

# Optional: scheduled end-of-day broadcaster (simple loop run in background thread)
def scheduled_results_broadcaster(interval_minutes=20):
    # This function when started will check for finished matches and broadcast comparison
    # to ADMIN_CHAT_IDS once per day after 23:30 UTC local - basic heuristic.
    last_broadcast_date = None
    while True:
        try:
            now_utc = datetime.utcnow()
            today = now_utc.date()
            # broadcasting window: after 23:30 UTC
            if now_utc.hour >= 23 and (last_broadcast_date != today):
                comp = compare_today_results()
                if comp['total'] > 0:
                    text = f"Daily results comparison — {comp['total']} matches\n"
                    if comp['accuracy'] is not None:
                        text += f"Accuracy: {comp['accuracy']:.2f}% ({comp['correct']}/{comp['total']})\n\n"
                    for d in comp['details'][:10]:
                        text += f"{d['match']}: predicted {d['predicted']} — actual {d['actual']}\n"
                    for cid in ADMIN_CHAT_IDS:
                        try:
                            bot.send_message(cid, text[:3900])
                        except Exception as e:
                            log.exception("Failed to notify admin")
                    last_broadcast_date = today
        except Exception:
            log.exception("broadcaster error")
        time.sleep(interval_minutes*60)

if __name__ == "__main__":
    # set webhook
    set_webhook()
    # optionally start broadcaster thread if admin ids present
    if ADMIN_CHAT_IDS:
        import threading
        t = threading.Thread(target=scheduled_results_broadcaster, daemon=True)
        t.start()
    log.info("Starting Flask app (Telebot)")
    app.run(host="0.0.0.0", port=PORT)
