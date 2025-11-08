#!/usr/bin/env python3
"""
KickVision v1.0.0 - Telebot + Flask (Render-friendly)

Usage:
  - pip install pyTelegramBotAPI flask requests
  - Set env vars:
      BOT_TOKEN (required)
      API_KEY (required - football-data.org)
      PORT (Render provides)
      ADMIN_CHAT_ID (optional) comma-separated chat ids to receive daily results summary
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

# Telebot
try:
    import telebot
    from telebot import types
except Exception as e:
    raise RuntimeError("Install pyTelegramBotAPI: pip install pyTelegramBotAPI") from e

# ----------------- Config & Globals -----------------
VERSION = "1.0.0"
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("API_KEY")  # football-data.org API token
PORT = int(os.getenv("PORT", 5000))
ADMIN_CHAT_IDS = [int(x) for x in os.getenv("ADMIN_CHAT_ID", "").split(",") if x.strip()]

if not BOT_TOKEN or not API_KEY:
    raise RuntimeError("Please set BOT_TOKEN and API_KEY environment variables")

API_BASE = "https://api.football-data.org/v4"
FPL_BASE = "https://fantasy.premierleague.com/api"

# Focused leagues
LEAGUES = {
    "premier league": 2021,
    "la liga": 2014,
    "bundesliga": 2002,
    "serie a": 2019,
    "ligue 1": 2015,
    "champions league": 2001
}
LEAGUE_DISPLAY = {v: k.title() for k, v in LEAGUES.items()}

# Caching and TTLs (seconds)
CACHE = {}
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

# Prediction ensemble params (kept modest for Render)
TOTAL_MODELS = 100
SIMS_PER_MODEL = 80

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("kickvision")

# requests session with API key header
session = requests.Session()
session.headers.update({"X-Auth-Token": API_KEY, "User-Agent": "KickVision/1.0.0"})

# store predictions in-memory (persist if you want to disk)
PREDICTION_STORE = {}  # key: match_id -> {'pred': {'home':..,'draw':..,'away':..}, 'time': ts}

# ----------------- Utilities -----------------
def cache_get(key):
    ent = CACHE.get(key)
    if not ent:
        return None
    val, ts, ttl = ent
    if time.time() - ts > ttl:
        del CACHE[key]
        return None
    return val

def cache_set(key, value, ttl):
    CACHE[key] = (value, time.time(), ttl)

def safe_get(url, params=None, timeout=12, tries=2):
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = 2 ** attempt
                log.warning(f"Rate limited: sleeping {wait}s")
                time.sleep(wait)
            else:
                log.debug(f"HTTP {r.status_code} from {url}: {r.text[:200]}")
                return None
        except Exception as e:
            log.debug(f"Request error: {e} ({url})")
            time.sleep(1)
    return None

# ----------------- Football-data wrappers -----------------
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
    """
    Return (gf_avg, ga_avg) computed from last n finished matches.
    Falls back to league average if not enough data.
    """
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
    """
    Football-Data may not expose assists. Try /competitions/{id}/scorers contains only goals.
    Fallback: For Premier League, use FPL assists info if present in bootstrap (approx).
    Otherwise return empty list.
    """
    # try football-data endpoint (some providers include assists)
    key = f"assists_{league_id}"
    cached = cache_get(key)
    if cached:
        return cached
    # No direct endpoint — attempt to fetch scorers and look for 'assists' field
    data = safe_get(f"{API_BASE}/competitions/{league_id}/scorers")
    assists = []
    if data and "scorers" in data:
        for s in data["scorers"]:
            # football-data v4 sometimes includes 'assists' in the scorer object (rare). Try it.
            # Example: s may have s['assist'] or similar — check defensively
            assist_count = None
            # defensive checks:
            if isinstance(s.get("player"), dict) and s.get("player").get("assists") is not None:
                assist_count = s["player"]["assists"]
            # else skip; (most APIs don't include assists)
            if assist_count is not None:
                assists.append({"player": s["player"]["name"], "team": s["team"]["name"], "assists": assist_count})
    # fallback: for Premier League, try FPL
    if not assists and league_id == LEAGUES["premier league"]:
        try:
            b = safe_get(f"{FPL_BASE}/bootstrap-static/")
            if b:
                elements = b.get("elements", [])
                # FPL includes 'assists' property for Premier League players
                players = []
                for p in elements:
                    assists_count = p.get("assists", 0)
                    players.append({"player": p.get("web_name"), "team": p.get("team"), "assists": assists_count, "now_cost": p.get("now_cost")})
                # To map team IDs -> names, use bootstrap teams
                teams = {t["id"]: t["name"] for t in b.get("teams", [])}
                # convert and sort by assists
                players_conv = []
                for p in players:
                    players_conv.append({"player": p["player"], "team": teams.get(p["team"], "Unknown"), "assists": p["assists"]})
                players_conv_sorted = sorted(players_conv, key=lambda x: x["assists"], reverse=True)[:limit]
                assists = players_conv_sorted
        except Exception:
            pass
    cache_set(key, assists, TTL["scorers"])
    return assists

def get_fpl_summary(limit=8):
    key = "fpl_bootstrap_summary"
    cached = cache_get(key)
    if cached:
        return cached
    data = safe_get(f"{FPL_BASE}/bootstrap-static/")
    out = []
    if data:
        elements = data.get("elements", [])
        players_sorted = sorted(elements, key=lambda x: float(x.get("form") or 0), reverse=True)[:limit]
        teams = {t["id"]: t["name"] for t in data.get("teams", [])}
        for p in players_sorted:
            out.append({"name": p.get("web_name"), "team": teams.get(p.get("team")), "form": p.get("form"), "selected": p.get("selected_by_percent"), "cost": p.get("now_cost")})
    cache_set(key, out, TTL["fpl"])
    return out

def get_live_matches():
    key = "live_matches"
    cached = cache_get(key)
    if cached:
        return cached
    matches = []
    # loop our six leagues
    for lid in LEAGUES.values():
        data = safe_get(f"{API_BASE}/competitions/{lid}/matches", params={"status": "LIVE"})
        if data and "matches" in data:
            matches.extend(data["matches"])
    cache_set(key, matches, TTL["live"])
    return matches

# ----------------- Prediction engine (lightweight 100-model ensemble) -----------------
def _poisson_sample(lam):
    # Knuth's algorithm
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
    total = float(sims)
    return {"home": hw / total, "draw": dw / total, "away": aw / total}

def estimate_expected_goals(h_stats, a_stats):
    # h_stats and a_stats are (gf_avg, ga_avg)
    # a simple multiplicative expected goals estimate with home advantage
    h_attack = max(0.1, h_stats[0])
    h_def = max(0.1, h_stats[1])
    a_attack = max(0.1, a_stats[0])
    a_def = max(0.1, a_stats[1])
    home_adv = 1.08
    home_lambda = (h_attack * a_def) / 2.5 * home_adv
    away_lambda = (a_attack * h_def) / 2.5 * 0.95
    return max(0.05, home_lambda), max(0.05, away_lambda)

def ensemble_predict_for_match(home_team_id, away_team_id):
    # get recent stats
    h_stats = get_team_recent_stats(home_team_id, n=6)
    a_stats = get_team_recent_stats(away_team_id, n=6)
    base_h, base_a = estimate_expected_goals(h_stats, a_stats)
    homes = []
    draws = []
    aways = []
    # produce TOTAL_MODELS variants (jittering)
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

# ----------------- Formatting helpers -----------------
def format_fixture_block(league_name, match, pred):
    h = match["homeTeam"]["name"]
    a = match["awayTeam"]["name"]
    t = match.get("utcDate", "")[11:16] if match.get("utcDate") else ""
    lines = []
    lines.append(f"🏆 {league_name} 🏆")
    lines.append(f"{h} vs {a} @ {t} UTC")
    lines.append(f"Home {pred['home']*100:.1f}% | Draw {pred['draw']*100:.1f}% | Away {pred['away']*100:.1f}%")
    # possible outcome:
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

# ----------------- Pagination helpers -----------------
def paginate_list(items, page, per_page=5):
    total = len(items)
    pages = max(1, math.ceil(total / per_page))
    page = max(1, min(page, pages))
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end], page, pages

# ----------------- Telebot + Flask app -----------------
bot = telebot.TeleBot(BOT_TOKEN)
app = Flask(__name__)

def start_menu_markup(page=1):
    markup = types.InlineKeyboardMarkup(row_width=2)
    if page == 1:
        # first page: leagues (show first 5)
        markup.add(types.InlineKeyboardButton("Premier League", callback_data="league_2021"),
                   types.InlineKeyboardButton("Champions League", callback_data="league_2001"))
        markup.add(types.InlineKeyboardButton("La Liga", callback_data="league_2014"),
                   types.InlineKeyboardButton("Serie A", callback_data="league_2019"))
        markup.add(types.InlineKeyboardButton("Bundesliga", callback_data="league_2002"),
                   types.InlineKeyboardButton("Next ▶", callback_data="menu_page_2"))
    else:
        # page 2
        markup.add(types.InlineKeyboardButton("Today", callback_data="today_1"),
                   types.InlineKeyboardButton("Live", callback_data="live"))
        markup.add(types.InlineKeyboardButton("Results", callback_data="results_manual"),
                   types.InlineKeyboardButton("Standings", callback_data="standings"))
        markup.add(types.InlineKeyboardButton("Top Stats ⚽️", callback_data="topstats"),
                   types.InlineKeyboardButton("FPL", callback_data="fpl"))
        markup.add(types.InlineKeyboardButton("◀ Back", callback_data="menu_page_1"))
    return markup

# /start handler
@bot.message_handler(commands=["start"])
def handle_start(m):
    txt = f"KickVision v{VERSION} — pick a league (page 1) or go to tools (page 2)"
    bot.send_message(m.chat.id, txt, reply_markup=start_menu_markup(page=1))

# callback handler (central)
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        data = call.data
        # Menu navigation
        if data == "menu_page_2":
            bot.edit_message_text("KickVision — Tools", chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=start_menu_markup(page=2))
            return
        if data == "menu_page_1":
            bot.edit_message_text("KickVision — Leagues", chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=start_menu_markup(page=1))
            return

        # League fixtures (show upcoming fixtures + prediction)
        if data.startswith("league_"):
            lid = int(data.split("_", 1)[1])
            matches = get_upcoming_fixtures_for_league(lid, days=7, limit=8)
            if not matches:
                bot.answer_callback_query(call.id, "No upcoming fixtures found for that league")
                return
            # paginate fixtures if many: page param can be encoded like league_{id}_p{page}
            # For first implementation show first 5, and include Next button if more
            page = 1
            per_page = 5
            paged, page, pages = paginate_list(matches, page, per_page)
            blocks = []
            for m in paged:
                pred = ensemble_predict_for_match(m["homeTeam"]["id"], m["awayTeam"]["id"])
                store_prediction(m["id"], pred)
                blocks.append(format_fixture_block(LEAGUE_DISPLAY.get(lid, "League"), m, pred))
            text = "\n\n".join(blocks)
            # build nav markup
            markup = types.InlineKeyboardMarkup(row_width=2)
            if pages > 1:
                # include next callback with encoded page
                markup.add(types.InlineKeyboardButton("Next ▶", callback_data=f"league_{lid}_p2"))
            markup.add(types.InlineKeyboardButton("Back", callback_data="menu_page_1"))
            bot.edit_message_text(text[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=markup)
            return

        # League with explicit page (league_{id}_p{page})
        if data.startswith("league_") and "_p" in data:
            # format league_2021_p2
            try:
                parts = data.split("_")
                lid = int(parts[1])
                page_part = parts[2]
                if page_part.startswith("p"):
                    page = int(page_part[1:])
                else:
                    page = 1
            except:
                page = 1
            matches = get_upcoming_fixtures_for_league(lid, days=7, limit=30)
            paged, page, pages = paginate_list(matches, page, per_page=5)
            blocks = []
            for m in paged:
                pred = ensemble_predict_for_match(m["homeTeam"]["id"], m["awayTeam"]["id"])
                store_prediction(m["id"], pred)
                blocks.append(format_fixture_block(LEAGUE_DISPLAY.get(lid, "League"), m, pred))
            text = "\n\n".join(blocks) or "No fixtures"
            markup = types.InlineKeyboardMarkup(row_width=2)
            if page > 1:
                prev_page = page - 1
                markup.add(types.InlineKeyboardButton("◀ Prev", callback_data=f"league_{lid}_p{prev_page}"))
            if page < pages:
                next_page = page + 1
                markup.add(types.InlineKeyboardButton("Next ▶", callback_data=f"league_{lid}_p{next_page}"))
            markup.add(types.InlineKeyboardButton("Back", callback_data="menu_page_1"))
            bot.edit_message_text(text[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=markup)
            return

        # Today: show fixtures & predictions across all leagues; paginate across many fixtures
        if data.startswith("today"):
            # data format: today_{page}
            page = 1
            if "_" in data:
                parts = data.split("_")
                if len(parts) > 1:
                    try:
                        page = int(parts[1])
                    except:
                        page = 1
            # gather today's fixtures across all leagues
            today_str = date.today().isoformat()
            all_fixtures = []
            for name, lid in LEAGUES.items():
                matches = safe_get(f"{API_BASE}/competitions/{lid}/matches", params={"dateFrom": today_str, "dateTo": today_str, "status": "SCHEDULED"})
                mlist = matches.get("matches", []) if matches else []
                for m in mlist:
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
            page_slice = all_fixtures[start:start + per_page]
            blocks = []
            for league_name, m, pred in page_slice:
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

        # Live
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
            for d in comp["details"][:10]:
                out.append(f"{d['match']}: predicted {d['predicted']} — actual {d['actual']}")
            bot.edit_message_text("\n".join(out)[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id)
            return

        # Standings: show league selection first
        if data == "standings":
            markup = types.InlineKeyboardMarkup(row_width=2)
            for name, lid in LEAGUES.items():
                markup.add(types.InlineKeyboardButton(name.title(), callback_data=f"stand_{lid}_p1"))
            markup.add(types.InlineKeyboardButton("Back", callback_data="menu_page_2"))
            bot.edit_message_text("Choose a league for standings:", chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=markup)
            return

        # Standings with page: stand_{lid}_p{page}
        if data.startswith("stand_"):
            try:
                _, lid_str, page_str = data.split("_")
                lid = int(lid_str)
                page = int(page_str.lstrip("p"))
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
            bot.edit_message_text(text[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=types.InlineKeyboardMarkup().add(types.InlineKeyboardButton("Back", callback_data="topstats")))
            return

        # Top assists display
        if data.startswith("tops_assists_"):
            lid = int(data.split("_")[-1])
            asst = get_top_assists(lid, limit=15)
            text = format_top_assists_block(LEAGUE_DISPLAY.get(lid, "League"), asst)
            bot.edit_message_text(text[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=types.InlineKeyboardMarkup().add(types.InlineKeyboardButton("Back", callback_data="topstats")))
            return

        # FPL
        if data == "fpl":
            fpl = get_fpl_summary(limit=8)
            text = format_fpl_block(fpl)
            bot.edit_message_text(text[:3900], chat_id=call.message.chat.id, message_id=call.message.message_id, reply_markup=types.InlineKeyboardMarkup().add(types.InlineKeyboardButton("Back", callback_data="menu_page_2")))
            return

        # Catchall
        bot.answer_callback_query(call.id, "Unknown action")
    except Exception as e:
        log.exception("Callback handler error")
        try:
            bot.answer_callback_query(call.id, "Error processing request")
        except:
            pass

# ----------------- Results comparison -----------------
def compare_today_results(league_id=None):
    # returns dict: {'accuracy': float or None, 'total': int, 'correct': int, 'details': [...]}
    today = date.today().isoformat()
    total = 0
    correct = 0
    details = []
    # get finished matches across either selected league or all six
    lids = [league_id] if league_id else list(LEAGUES.values())
    for lid in lids:
        data = safe_get(f"{API_BASE}/competitions/{lid}/matches", params={"dateFrom": today, "dateTo": today, "status": "FINISHED"})
        matches = data.get("matches", []) if data else []
        for m in matches:
            mid = str(m["id"])
            full = m.get("score", {}).get("fullTime", {})
            hg = full.get("home")
            ag = full.get("away")
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

# ----------------- Scheduled daily results broadcaster -----------------
def daily_results_broadcaster(check_hour_utc=23, check_minute_utc=30):
    """
    Runs in background thread. Sends daily results comparison to ADMIN_CHAT_IDS
    once per day after the check_hour/check_minute (UTC).
    """
    last_sent_date = None
    while True:
        try:
            now = datetime.utcnow()
            today = now.date()
            # only broadcast once per day after the scheduled time
            if (now.hour > check_hour_utc or (now.hour == check_hour_utc and now.minute >= check_minute_utc)) and last_sent_date != today:
                # compile results
                comp = compare_today_results()
                if comp["total"] > 0 and ADMIN_CHAT_IDS:
                    text_lines = [f"Daily Results Comparison — {comp['total']} matches"]
                    if comp["accuracy"] is not None:
                        text_lines.append(f"Accuracy: {comp['accuracy']:.2f}% ({comp['correct']}/{comp['total']})")
                    text_lines.append("")
                    for d in comp["details"][:30]:
                        text_lines.append(f"{d['match']} — Pred: {d['predicted']} (conf {d['confidence']*100:.1f}%) — Actual: {d['actual']}")
                    text = "\n".join(text_lines)[:3900]
                    for cid in ADMIN_CHAT_IDS:
                        try:
                            bot.send_message(cid, text)
                        except Exception:
                            log.exception(f"Failed to send daily results to {cid}")
                    last_sent_date = today
                else:
                    # nothing to broadcast, but mark as sent to avoid retries
                    last_sent_date = today
        except Exception:
            log.exception("daily broadcaster error")
        # sleep a short while before next check
        time.sleep(60 * 5)

# ----------------- Run Telebot polling and Flask -----------------
@app.route("/", methods=["GET"])
def index():
    return f"KickVision Telebot v{VERSION} running."

# Keep Flask endpoint for health but telebot will use long polling in background
def run_bot_polling():
    """
    Run telebot long-polling in a daemon thread.
    We use non-blocking polling so Flask app.run continues.
    """
    try:
        log.info("Starting telebot polling (background thread)...")
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
    except Exception:
        log.exception("bot polling stopped")

if __name__ == "__main__":
    # start telebot polling in background thread
    t = threading.Thread(target=run_bot_polling, daemon=True)
    t.start()

    # start daily broadcaster if admin ids specified
    if ADMIN_CHAT_IDS:
        bthread = threading.Thread(target=daily_results_broadcaster, daemon=True)
        bthread.start()
        log.info(f"Daily broadcaster started — admin ids: {ADMIN_CHAT_IDS}")

    log.info("Starting Flask app (KickVision Telebot)")
    # User said Render works with app.run trick — run Flask to keep process alive
    app.run(host="0.0.0.0", port=PORT)
