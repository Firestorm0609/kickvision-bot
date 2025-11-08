#!/usr/bin/env python3
"""
KickVision v1.0.0 - Telebot + Flask (Render-ready)
Features:
 - Auto delete webhook at startup (prevents 409 when polling)
 - Telebot polling in background thread + Flask app.run() for Render
 - Predictions: lightweight 100-model Poisson ensemble using recent match stats
 - /start inline menu (2 pages): leagues (fixtures+predictions) + tools (today, live, results, standings, Top Stats ⚽️, FPL)
 - Top Stats combines Top Scorers & Top Assists (league header)
 - Today & League fixtures paginated
 - Standings paginated (10 per page)
 - FPL bootstrap usage (public API)
 - Daily automatic results summary at 00:00 UTC to ALL users who used /start (persisted in users.json)
 - Caching for API calls
Requirements:
 pip install pyTelegramBotAPI flask requests
Env:
 BOT_TOKEN (required)
 API_KEY   (football-data.org token) (required)
 PORT (optional, Render provides)
"""

import os
import time
import math
import random
import logging
import threading
import json
from datetime import date, datetime, timedelta
from collections import defaultdict
from flask import Flask, request
import requests

# Telebot imports
try:
    import telebot
    from telebot import types
except Exception as e:
    raise RuntimeError("Install pyTelegramBotAPI: pip install pyTelegramBotAPI") from e

# ---------------- Config ----------------
VERSION = "1.0.0"
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("API_KEY")
PORT = int(os.getenv("PORT", 5000))

if not BOT_TOKEN or not API_KEY:
    raise RuntimeError("Please set BOT_TOKEN and API_KEY environment variables")

API_BASE = "https://api.football-data.org/v4"
FPL_BASE = "https://fantasy.premierleague.com/api"

# Focused leagues (ids from football-data.org)
LEAGUES = {
    "premier league": 2021,
    "la liga": 2014,
    "bundesliga": 2002,
    "serie a": 2019,
    "ligue 1": 2015,
    "champions league": 2001
}
LEAGUE_DISPLAY = {v: k.title() for k, v in LEAGUES.items()}

# TTLs for caches (seconds)
TTL = {
    "teams": 24 * 3600,
    "fixtures": 5 * 60,
    "standings": 30 * 60,
    "scorers": 30 * 60,
    "predictions": 60 * 60,
    "live": 60 * 1,
    "fpl": 15 * 60,
    "team_matches": 60 * 60
}

# Ensemble params (kept conservative for Render)
TOTAL_MODELS = 100
SIMS_PER_MODEL = 80

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("kickvision")

# requests session
session = requests.Session()
session.headers.update({"X-Auth-Token": API_KEY, "User-Agent": "KickVision/1.0.0"})

# in-memory caches and stores
CACHE = {}
PREDICTION_STORE = {}  # match_id -> {'pred': {...}, 'time': ts}
USERS_FILE = "users.json"  # persist list of user chat ids who used /start
USER_SESSIONS = set()

# ---------- Helpers: cache, safe_get ----------
def cache_get(key):
    e = CACHE.get(key)
    if not e:
        return None
    val, ts, ttl = e
    if time.time() - ts > ttl:
        del CACHE[key]
        return None
    return val

def cache_set(key, val, ttl):
    CACHE[key] = (val, time.time(), ttl)

def safe_get(url, params=None, timeout=12, tries=2):
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = 2 ** attempt
                log.warning(f"Rate limited by API for {url}. sleeping {wait}s")
                time.sleep(wait)
            else:
                log.debug(f"HTTP {r.status_code} from {url}: {r.text[:200]}")
                return None
        except Exception as e:
            log.debug(f"Request error {e} for {url}")
            time.sleep(1)
    return None

# ---------- Football-data wrappers ----------
def get_upcoming_fixtures_for_league(league_id, days=7, limit=12):
    key = f"fixtures_{league_id}_{days}"
    cached = cache_get(key)
    if cached:
        return cached
    start = date.today().isoformat()
    end = (date.today() + timedelta(days=days)).isoformat()
    data = safe_get(f"{API_BASE}/competitions/{league_id}/matches", params={"status": "SCHEDULED", "dateFrom": start, "dateTo": end})
    matches = data.get("matches", []) if data else []
    matches = matches[:limit]
    cache_set(key, matches, TTL["fixtures"])
    return matches

def get_team_recent_matches(team_id, limit=6):
    key = f"team_matches_{team_id}_{limit}"
    cached = cache_get(key)
    if cached:
        return cached
    data = safe_get(f"{API_BASE}/teams/{team_id}/matches", params={"status": "FINISHED", "limit": limit})
    matches = data.get("matches", []) if data else []
    cache_set(key, matches, TTL["team_matches"])
    return matches

def get_team_recent_stats(team_id, n=6):
    matches = get_team_recent_matches(team_id, limit=n)
    gf = []
    ga = []
    for m in matches[:n]:
        full = m.get("score", {}).get("fullTime", {})
        if m["homeTeam"]["id"] == team_id:
            gf.append(full.get("home", 0))
            ga.append(full.get("away", 0))
        else:
            gf.append(full.get("away", 0))
            ga.append(full.get("home", 0))
    if not gf:
        return (1.3, 1.3)
    return (sum(gf) / len(gf), sum(ga) / len(ga))

def get_standings_for_league(league_id):
    key = f"standings_{league_id}"
    cached = cache_get(key)
    if cached:
        return cached
    data = safe_get(f"{API_BASE}/competitions/{league_id}/standings")
    standings = []
    if data and "standings" in data:
        for st in data["standings"]:
            if st.get("type") == "TOTAL":
                for row in st.get("table", []):
                    standings.append({
                        "position": row["position"],
                        "team": row["team"]["name"],
                        "points": row["points"],
                        "played": row["playedGames"],
                        "gd": row["goalDifference"]
                    })
                break
    cache_set(key, standings, TTL["standings"])
    return standings

def get_top_scorers(league_id, limit=10):
    key = f"scorers_{league_id}"
    cached = cache_get(key)
    if cached:
        return cached
    data = safe_get(f"{API_BASE}/competitions/{league_id}/scorers")
    scorers = []
    if data and "scorers" in data:
        for s in data["scorers"][:limit]:
            scorers.append({"player": s["player"]["name"], "team": s["team"]["name"], "goals": s["goals"]})
    cache_set(key, scorers, TTL["scorers"])
    return scorers

def get_top_assists(league_id, limit=10):
    key = f"assists_{league_id}"
    cached = cache_get(key)
    if cached:
        return cached
    assists = []
    # Try football-data scorers object for assist-like fields (rare)
    data = safe_get(f"{API_BASE}/competitions/{league_id}/scorers")
    if data and "scorers" in data:
        for s in data["scorers"]:
            # defensive checks; many APIs don't include assists
            if isinstance(s.get("player"), dict) and s.get("player").get("assists") is not None:
                assists.append({"player": s["player"]["name"], "team": s["team"]["name"], "assists": s["player"]["assists"]})
    # Fallback for Premier League: use FPL bootstrap (has 'assists')
    if not assists and league_id == LEAGUES["premier league"]:
        b = safe_get(f"{FPL_BASE}/bootstrap-static/")
        if b:
            teams_map = {t["id"]: t["name"] for t in b.get("teams", [])}
            players = b.get("elements", [])
            players_sorted = sorted(players, key=lambda x: x.get("assists", 0), reverse=True)
            for p in players_sorted[:limit]:
                assists.append({"player": p.get("web_name"), "team": teams_map.get(p.get("team")), "assists": p.get("assists", 0)})
    cache_set(key, assists, TTL["scorers"])
    return assists

def get_fpl_summary(limit=8):
    key = "fpl_summary"
    cached = cache_get(key)
    if cached:
        return cached
    data = safe_get(f"{FPL_BASE}/bootstrap-static/")
    out = []
    if data:
        teams = {t["id"]: t["name"] for t in data.get("teams", [])}
        elements = data.get("elements", [])
        players_sorted = sorted(elements, key=lambda x: float(x.get("form") or 0), reverse=True)[:limit]
        for p in players_sorted:
            out.append({"name": p.get("web_name"), "team": teams.get(p.get("team")), "form": p.get("form"), "selected": p.get("selected_by_percent"), "cost": p.get("now_cost")})
    cache_set(key, out, TTL["fpl"])
    return out

def get_live_matches():
    key = "live_matches"
    cached = cache_get(key)
    if cached:
        return cached
    items = []
    for lid in LEAGUES.values():
        data = safe_get(f"{API_BASE}/competitions/{lid}/matches", params={"status": "LIVE"})
        if data and "matches" in data:
            items.extend(data["matches"])
    cache_set(key, items, TTL["live"])
    return items

# ---------- Prediction Engine ----------
def _poisson_sample(lam):
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1

def simulate_model_once(h_lambda, a_lambda, sims=100, seed=None):
    if seed is not None:
        random.seed(seed)
    hw = dw = aw = 0
    for _ in range(sims):
        hg = _poisson_sample(h_lambda)
        ag = _poisson_sample(a_lambda)
        if hg > ag:
            hw += 1
        elif hg == ag:
            dw += 1
        else:
            aw += 1
    tot = float(sims)
    return {"home": hw / tot, "draw": dw / tot, "away": aw / tot}

def estimate_expected_goals(h_stats, a_stats):
    h_attack = max(0.1, h_stats[0])
    h_def = max(0.1, h_stats[1])
    a_attack = max(0.1, a_stats[0])
    a_def = max(0.1, a_stats[1])
    home_adv = 1.08
    home_lambda = (h_attack * a_def) / 2.5 * home_adv
    away_lambda = (a_attack * h_def) / 2.5 * 0.95
    return max(0.05, home_lambda), max(0.05, away_lambda)

def ensemble_predict_for_match(home_team_id, away_team_id):
    h_stats = get_team_recent_stats(home_team_id, n=6)
    a_stats = get_team_recent_stats(away_team_id, n=6)
    base_h, base_a = estimate_expected_goals(h_stats, a_stats)
    homes = []
    draws = []
    aways = []
    for m in range(TOTAL_MODELS):
        jitter_h = random.uniform(0.86, 1.14)
        jitter_a = random.uniform(0.86, 1.14)
        scale = random.uniform(0.85, 1.25)
        h_l = base_h * jitter_h * scale
        a_l = base_a * jitter_a * scale
        res = simulate_model_once(h_l, a_l, sims=SIMS_PER_MODEL, seed=(m * 97 + int(time.time() % 10000)))
        homes.append(res["home"])
        draws.append(res["draw"])
        aways.append(res["away"])
    final = {"home": sum(homes) / len(homes), "draw": sum(draws) / len(draws), "away": sum(aways) / len(aways)}
    return final

def store_prediction(match_id, prediction):
    PREDICTION_STORE[str(match_id)] = {"pred": prediction, "time": time.time()}

# ---------- Formatting ----------
def format_fixture_block(league_name, match, pred):
    h = match["homeTeam"]["name"]
    a = match["awayTeam"]["name"]
    t = match.get("utcDate", "")[11:16] if match.get("utcDate") else ""
    lines = []
    lines.append(f"🏆 {league_name} 🏆")
    lines.append(f"{h} vs {a} @ {t} UTC")
    lines.append(f"Home {pred['home']*100:.1f}% | Draw {pred['draw']*100:.1f}% | Away {pred['away']*100:.1f}%")
    top = max(("home", "draw", "away"), key=lambda k: pred[k])
    outcome_map = {"home": "Home", "draw": "Draw", "away": "Away"}
    lines.append(f"Possible Outcome – {outcome_map[top]}")
    return "\n".join(lines)

def format_top_scorers_block(league_name, scorers):
    out = [f"⚽ {league_name} Top Scorers"]
    if not scorers:
        out.append("No scorer data available")
        return "\n".join(out)
    for i, s in enumerate(scorers, 1):
        out.append(f"{i}. {s['player']} ({s['team']}) – {s['goals']}")
    return "\n".join(out)

def format_top_assists_block(league_name, assists):
    out = [f"🅰️ {league_name} Top Assists"]
    if not assists:
        out.append("No assists data available")
        return "\n".join(out)
    for i, s in enumerate(assists, 1):
        out.append(f"{i}. {s['player']} ({s['team']}) – {s.get('assists', 0)}")
    return "\n".join(out)

def format_fpl_block(entries):
    out = ["🌟 FPL — Top Form (sample)"]
    if not entries:
        out.append("FPL data unavailable")
        return "\n".join(out)
    for i, p in enumerate(entries, 1):
        out.append(f"{i}. {p['name']} ({p['team']}) — Form {p['form']} — Selected {p['selected']} — Cost {p['cost']}")
    return "\n".join(out)

# ---------- Pagination helpers ----------
def paginate_list(items, page, per_page=5):
    total = len(items)
    pages = max(1, math.ceil(total / per_page))
    page = max(1, min(page, pages))
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end], page, pages

# ---------- Telebot + Flask app ----------
bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

def start_menu_markup(page=1):
    markup = types.InlineKeyboardMarkup(row_width=2)
    if page == 1:
        markup.add(types.InlineKeyboardButton("Premier League", callback_data="league_2021"),
                   types.InlineKeyboardButton("Champions League", callback_data="league_2001"))
        markup.add(types.InlineKeyboardButton("La Liga", callback_data="league_2014"),
                   types.InlineKeyboardButton("Serie A", callback_data="league_2019"))
        markup.add(types.InlineKeyboardButton("Bundesliga", callback_data="league_2002"),
                   types.InlineKeyboardButton("Next ▶", callback_data="menu_page_2"))
    else:
        markup.add(types.InlineKeyboardButton("Today", callback_data="today_1"),
                   types.InlineKeyboardButton("Live", callback_data="live"))
        markup.add(types.InlineKeyboardButton("Results (manual)", callback_data="results_manual"),
                   types.InlineKeyboardButton("Standings", callback_data="standings"))
        markup.add(types.InlineKeyboardButton("Top Stats ⚽️", callback_data="topstats"),
                   types.InlineKeyboardButton("FPL", callback_data="fpl"))
        markup.add(types.InlineKeyboardButton("◀ Back", callback_data="menu_page_1"))
    return markup

# persist users who used /start (for daily broadcast)
def load_users():
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r") as f:
                ids = json.load(f)
                return set(ids)
    except Exception:
        log.exception("load_users failed")
    return set()

def save_users():
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(list(USER_SESSIONS), f)
    except Exception:
        log.exception("save_users failed")

USER_SESSIONS = load_users()

@bot.message_handler(commands=["start"])
def handle_start(m):
    USER_SESSIONS.add(m.chat.id)
    save_users()
    txt = f"KickVision v{VERSION} — pick a league (page 1) or go to tools (page 2)"
    bot.send_message(m.chat.id, txt, reply_markup=start_menu_markup(page=1))

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        data = call.data
        # navigation
        if data == "menu_page_2":
            bot.edit_message_text("KickVision — Tools", chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=start_menu_markup(page=2))
            return
        if data == "menu_page_1":
            bot.edit_message_text("KickVision — Leagues", chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=start_menu_markup(page=1))
            return

        # league fixtures (first page)
        if data.startswith("league_") and "_p" not in data:
            lid = int(data.split("_", 1)[1])
            matches = get_upcoming_fixtures_for_league(lid, days=7, limit=30)
            if not matches:
                bot.answer_callback_query(call.id, "No upcoming fixtures found for that league")
                return
            page = 1
            per_page = 5
            paged, page, pages = paginate_list(matches, page, per_page)
            blocks = []
            for m in paged:
                pred = ensemble_predict_for_match(m["homeTeam"]["id"], m["awayTeam"]["id"])
                store_prediction(m["id"], pred)
                blocks.append(format_fixture_block(LEAGUE_DISPLAY.get(lid, "League"), m, pred))
            text = "\n\n".join(blocks)
            markup = types.InlineKeyboardMarkup(row_width=2)
            if pages > 1:
                markup.add(types.InlineKeyboardButton("Next ▶", callback_data=f"league_{lid}_p2"))
            markup.add(types.InlineKeyboardButton("Back", callback_data="menu_page_1"))
            bot.edit_message_text(text[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=markup)
            return

        # league with page e.g. league_2021_p2
        if data.startswith("league_") and "_p" in data:
            try:
                parts = data.split("_")
                lid = int(parts[1])
                ppart = parts[2]
                page = int(ppart.lstrip("p")) if ppart.startswith("p") else 1
            except Exception:
                page = 1
            matches = get_upcoming_fixtures_for_league(lid, days=7, limit=60)
            paged, page, pages = paginate_list(matches, page, per_page=5)
            blocks = []
            for m in paged:
                pred = ensemble_predict_for_match(m["homeTeam"]["id"], m["awayTeam"]["id"])
                store_prediction(m["id"], pred)
                blocks.append(format_fixture_block(LEAGUE_DISPLAY.get(lid, "League"), m, pred))
            text = "\n\n".join(blocks) or "No fixtures"
            markup = types.InlineKeyboardMarkup(row_width=2)
            if page > 1:
                markup.add(types.InlineKeyboardButton("◀ Prev", callback_data=f"league_{lid}_p{page-1}"))
            if page < pages:
                markup.add(types.InlineKeyboardButton("Next ▶", callback_data=f"league_{lid}_p{page+1}"))
            markup.add(types.InlineKeyboardButton("Back", callback_data="menu_page_1"))
            bot.edit_message_text(text[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=markup)
            return

        # today_{page}
        if data.startswith("today"):
            page = 1
            try:
                if "_" in data:
                    page = int(data.split("_")[1])
            except Exception:
                page = 1
            today_str = date.today().isoformat()
            all_fixtures = []
            for name, lid in LEAGUES.items():
                resp = safe_get(f"{API_BASE}/competitions/{lid}/matches", params={"dateFrom": today_str, "dateTo": today_str, "status": "SCHEDULED"})
                matches = resp.get("matches", []) if resp else []
                for m in matches:
                    pred = ensemble_predict_for_match(m["homeTeam"]["id"], m["awayTeam"]["id"])
                    store_prediction(m["id"], pred)
                    all_fixtures.append((LEAGUE_DISPLAY.get(lid, name.title()), m, pred))
            if not all_fixtures:
                bot.edit_message_text("No fixtures today in the selected leagues.", chat_id=call.message.chat.id, message_id=call.message.message_id)
                return
            per_page = 5
            total = len(all_fixtures)
            pages = max(1, math.ceil(total / per_page))
            page = max(1, min(page, pages))
            start = (page - 1) * per_page
            slice_ = all_fixtures[start:start + per_page]
            blocks = []
            for league_name, m, pred in slice_:
                blocks.append(format_fixture_block(league_name, m, pred))
            text = "\n\n".join(blocks)
            markup = types.InlineKeyboardMarkup(row_width=2)
            if page > 1:
                markup.add(types.InlineKeyboardButton("◀ Prev", callback_data=f"today_{page-1}"))
            if page < pages:
                markup.add(types.InlineKeyboardButton("Next ▶", callback_data=f"today_{page+1}"))
            markup.add(types.InlineKeyboardButton("Back", callback_data="menu_page_2"))
            bot.edit_message_text(text[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=markup)
            return

        # Live matches
        if data == "live":
            live = get_live_matches()
            if not live:
                bot.edit_message_text("No live matches right now.", chat_id=call.message.chat.id, message_id=call.message.message_id)
                return
            lines = []
            for m in live:
                score = m.get("score", {}).get("fullTime", {})
                lines.append(f"{m['homeTeam']['name']} {score.get('home','-')} - {score.get('away','-')} {m['awayTeam']['name']}")
            bot.edit_message_text("\n".join(lines)[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id)
            return

        # Results manual trigger
        if data == "results_manual":
            comp = compare_today_results()
            if comp["total"] == 0:
                bot.edit_message_text("No finished matches with stored predictions today.", chat_id=call.message.chat.id, message_id=call.message.message_id)
                return
            out = [f"Results comparison — {comp['total']} matches"]
            if comp["accuracy"] is not None:
                out.append(f"Accuracy: {comp['accuracy']:.2f}% ({comp['correct']}/{comp['total']})")
            out.append("")
            for d in comp["details"][:20]:
                out.append(f"{d['match']}: predicted {d['predicted']} — actual {d['actual']} (conf {d['confidence']*100:.1f}%)")
            bot.edit_message_text("\n".join(out)[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id)
            return

        # Standings -> choose league
        if data == "standings":
            markup = types.InlineKeyboardMarkup(row_width=2)
            for name, lid in LEAGUES.items():
                markup.add(types.InlineKeyboardButton(name.title(), callback_data=f"stand_{lid}_p1"))
            markup.add(types.InlineKeyboardButton("Back", callback_data="menu_page_2"))
            bot.edit_message_text("Choose a league for standings:", chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=markup)
            return

        # Standings with page
        if data.startswith("stand_"):
            try:
                _, lid_str, pstr = data.split("_")
                lid = int(lid_str)
                page = int(pstr.lstrip("p"))
            except Exception:
                bot.answer_callback_query(call.id, "Invalid standings request")
                return
            standings = get_standings_for_league(lid)
            if not standings:
                bot.edit_message_text("Standings unavailable.", chat_id=call.message.chat.id, message_id=call.message.message_id)
                return
            per_page = 10
            slice_, page, pages = paginate_list(standings, page, per_page=per_page)
            lines = [f"{LEAGUE_DISPLAY.get(lid,'League')} Standings — Page {page}/{pages}"]
            for row in slice_:
                lines.append(f"{row['position']}. {row['team']} — {row['points']} pts, GD {row['gd']}")
            markup = types.InlineKeyboardMarkup(row_width=2)
            if page > 1:
                markup.add(types.InlineKeyboardButton("◀ Prev", callback_data=f"stand_{lid}_p{page-1}"))
            if page < pages:
                markup.add(types.InlineKeyboardButton("Next ▶", callback_data=f"stand_{lid}_p{page+1}"))
            markup.add(types.InlineKeyboardButton("Back", callback_data="menu_page_2"))
            bot.edit_message_text("\n".join(lines)[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=markup)
            return

        # Top Stats combined menu
        if data == "topstats":
            markup = types.InlineKeyboardMarkup(row_width=2)
            for name, lid in LEAGUES.items():
                markup.add(types.InlineKeyboardButton(name.title() + " — Scorers", callback_data=f"tops_scorers_{lid}"),
                           types.InlineKeyboardButton(name.title() + " — Assists", callback_data=f"tops_assists_{lid}"))
            markup.add(types.InlineKeyboardButton("Back", callback_data="menu_page_2"))
            bot.edit_message_text("Top Stats — choose league and stat:", chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=markup)
            return

        # Top scorers display
        if data.startswith("tops_scorers_"):
            lid = int(data.split("_")[-1])
            sc = get_top_scorers(lid, limit=15)
            text = format_top_scorers_block(LEAGUE_DISPLAY.get(lid, "League"), sc)
            markup = types.InlineKeyboardMarkup().add(types.InlineKeyboardButton("Back", callback_data="topstats"))
            bot.edit_message_text(text[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=markup)
            return

        # Top assists display
        if data.startswith("tops_assists_"):
            lid = int(data.split("_")[-1])
            asst = get_top_assists(lid, limit=15)
            text = format_top_assists_block(LEAGUE_DISPLAY.get(lid, "League"), asst)
            markup = types.InlineKeyboardMarkup().add(types.InlineKeyboardButton("Back", callback_data="topstats"))
            bot.edit_message_text(text[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=markup)
            return

        # FPL
        if data == "fpl":
            fpl = get_fpl_summary(limit=8)
            text = format_fpl_block(fpl)
            markup = types.InlineKeyboardMarkup().add(types.InlineKeyboardButton("Back", callback_data="menu_page_2"))
            bot.edit_message_text(text[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=markup)
            return

        bot.answer_callback_query(call.id, "Unknown action")
    except Exception:
        log.exception("callback_handler error")
        try:
            bot.answer_callback_query(call.id, "Error processing request")
        except:
            pass

# ---------- Results comparison ----------
def compare_today_results(league_id=None):
    today = date.today().isoformat()
    total = 0
    correct = 0
    details = []
    lids = [league_id] if league_id else list(LEAGUES.values())
    for lid in lids:
        data = safe_get(f"{API_BASE}/competitions/{lid}/matches", params={"dateFrom": today, "dateTo": today, "status": "FINISHED"})
        matches = data.get("matches", []) if data else []
        for m in matches:
            mid = str(m["id"])
            full = m.get("score", {}).get("fullTime", {})
            hg = full.get("home"); ag = full.get("away")
            if hg is None or ag is None:
                continue
            actual = "home" if hg > ag else ("draw" if hg == ag else "away")
            pred_record = PREDICTION_STORE.get(mid)
            if pred_record:
                pred = pred_record["pred"]
                pred_choice = max(("home", "draw", "away"), key=lambda k: pred.get(k, 0))
                if pred_choice == actual:
                    correct += 1
                total += 1
                details.append({"match": f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}", "predicted": pred_choice, "actual": actual, "confidence": pred.get(pred_choice, 0)})
    accuracy = (correct / total * 100) if total else None
    return {"accuracy": accuracy, "total": total, "correct": correct, "details": details}

# ---------- Daily broadcaster to ALL users ----------
def daily_broadcast_to_users():
    """
    Runs in background thread. Waits for 00:00 UTC then sends results comparison to all users who used /start.
    Repeats daily.
    """
    log.info("Daily broadcaster thread started (sends to all users at 00:00 UTC).")
    while True:
        try:
            now = datetime.utcnow()
            # seconds until next 00:00 UTC of next day
            next_midnight = (datetime.combine(now.date(), datetime.min.time()) + timedelta(days=1))
            seconds = (next_midnight - now).total_seconds()
            # sleep until next midnight
            if seconds > 0:
                time.sleep(seconds + 2)  # small buffer
            # It's 00:00 UTC now (or shortly after)
            comp = compare_today_results()
            if comp["total"] > 0:
                lines = [f"Daily Results Comparison — {comp['total']} matches"]
                if comp["accuracy"] is not None:
                    lines.append(f"Accuracy: {comp['accuracy']:.2f}% ({comp['correct']}/{comp['total']})")
                lines.append("")
                for d in comp["details"][:30]:
                    lines.append(f"{d['match']} — Pred: {d['predicted']} (conf {d['confidence']*100:.1f}%) — Actual: {d['actual']}")
                text = "\n".join(lines)[:3900]
                # send to all stored users
                users = list(USER_SESSIONS)
                log.info(f"Broadcasting daily results to {len(users)} users.")
                for uid in users:
                    try:
                        bot.send_message(uid, text)
                    except Exception:
                        log.exception(f"Failed to send daily results to {uid}")
            else:
                log.info("Daily broadcaster: no finished matches with predictions today.")
            # small sleep to avoid tight loop if time calc off
            time.sleep(5)
        except Exception:
            log.exception("daily broadcaster error")
            time.sleep(60)

# ---------- Startup utilities ----------
def delete_existing_webhook():
    try:
        # delete existing webhook and drop pending updates to avoid conflict
        bot.delete_webhook(drop_pending_updates=True)
        log.info("Existing webhook deleted (if any).")
    except Exception:
        log.exception("delete_webhook failed (likely no webhook or network issue)")

def run_bot_polling():
    try:
        log.info("Starting telebot polling (background thread)...")
        # Infinity polling will handle reconnections
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
    except Exception:
        log.exception("bot polling stopped unexpectedly")

# ---------- Flask endpoints (health) ----------
@app.route("/", methods=["GET"])
def index():
    return f"KickVision Telebot v{VERSION} running."

# ---------- Main ----------
if __name__ == "__main__":
    # ensure webhook removed so polling works
    delete_existing_webhook()

    # start polling in background thread
    poll_thread = threading.Thread(target=run_bot_polling, daemon=True)
    poll_thread.start()

    # start daily broadcaster thread to send results to all users at 00:00 UTC
    broadcaster_thread = threading.Thread(target=daily_broadcast_to_users, daemon=True)
    broadcaster_thread.start()

    log.info("Starting Flask app (KickVision Telebot).")
    # run Flask to keep Render process alive
    app.run(host="0.0.0.0", port=PORT)
