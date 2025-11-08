#!/usr/bin/env python3
"""
KickVision v1.0.0 - Aiogram async build (webhook)
- Requires: pip install aiogram aiohttp requests
- Env vars:
    BOT_TOKEN
    API_KEY
    RENDER_EXTERNAL_HOSTNAME
    ADMIN_CHAT_ID (optional comma-separated)
"""

import os, time, math, random, logging, json, asyncio
from datetime import date, timedelta, datetime
import requests

# Aiogram imports
try:
    from aiogram import Bot, Dispatcher, types
    from aiogram.utils.executor import start_webhook
    from aiohttp import web
except Exception as e:
    raise RuntimeError("Install aiogram and aiohttp: pip install aiogram aiohttp")

# Config
VERSION = "1.0.0"
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("API_KEY")
RENDER_HOST = os.getenv("RENDER_EXTERNAL_HOSTNAME")
PORT = int(os.getenv("PORT", 5000))
WEBHOOK_PATH = f"/{BOT_TOKEN}"
WEBHOOK_URL = f"https://{RENDER_HOST}{WEBHOOK_PATH}" if RENDER_HOST else None
ADMIN_CHAT_IDS = [int(x) for x in os.getenv("ADMIN_CHAT_ID","").split(",") if x.strip()]

if not BOT_TOKEN or not API_KEY:
    raise RuntimeError("BOT_TOKEN and API_KEY env vars are required")

API_BASE = "https://api.football-data.org/v4"
FPL_BASE = "https://fantasy.premierleague.com/api"

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
TTL = {'fixtures':300,'teams':3600*12,'standings':1800,'scorers':1800,'predictions':3600,'fpl':1800,'live':60}

# prediction params
TOTAL_MODELS=100
SIMS_PER_MODEL=80

# Logging and session
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log=logging.getLogger("kickvision")
session = requests.Session()
session.headers.update({"X-Auth-Token": API_KEY})

# helpers
def cache_get(k):
    e=CACHE.get(k)
    if not e: return None
    val,ts,ttl=e
    if time.time()-ts>ttl:
        del CACHE[k]; return None
    return val

def cache_set(k,v,ttl):
    CACHE[k]=(v,time.time(),ttl)

def safe_get(url, params=None, timeout=10):
    try:
        r = session.get(url, params=params, timeout=timeout)
        if r.status_code==200:
            return r.json()
        else:
            log.debug(f"API {r.status_code}")
            return None
    except Exception:
        log.exception("safe_get error")
        return None

# data functions (same logic as telebot script)
def get_upcoming_fixtures(league_id, days=7, limit=6):
    key=f"fixtures_{league_id}_{days}"
    cached=cache_get(key)
    if cached: return cached
    start=date.today().isoformat(); end=(date.today()+timedelta(days=days)).isoformat()
    data=safe_get(f"{API_BASE}/competitions/{league_id}/matches", params={"status":"SCHEDULED","dateFrom":start,"dateTo":end})
    matches = data.get("matches",[])[:limit] if data else []
    cache_set(key,matches,TTL['fixtures']); return matches

def get_team_recent_stats(team_id,last=6):
    key=f"team_stats_{team_id}_{last}"; cached=cache_get(key)
    if cached: return cached
    data=safe_get(f"{API_BASE}/teams/{team_id}/matches", params={"status":"FINISHED","limit":last})
    if not data or "matches" not in data: res=(1.3,1.3); cache_set(key,res,TTL['teams']); return res
    gf=[]; ga=[]
    for m in data["matches"]:
        ft = m.get("score",{}).get("fullTime",{})
        if m["homeTeam"]["id"]==team_id:
            gf.append(ft.get("home",0)); ga.append(ft.get("away",0))
        else:
            gf.append(ft.get("away",0)); ga.append(ft.get("home",0))
    res=(sum(gf)/len(gf), sum(ga)/len(ga)) if gf else (1.3,1.3)
    cache_set(key,res,TTL['teams']); return res

def _poisson_sample(lam):
    L=math.exp(-lam); k=0; p=1.0
    while p> L:
        k+=1; p*=random.random()
    return k-1

def simulate_model(h_lambda,a_lambda,sims=100, seed=None):
    if seed is not None: random.seed(seed)
    hw=dw=aw=0
    for _ in range(sims):
        hg=_poisson_sample(h_lambda); ag=_poisson_sample(a_lambda)
        if hg>ag: hw+=1
        elif hg==ag: dw+=1
        else: aw+=1
    tot=float(sims)
    return {'home': hw/tot, 'draw': dw/tot, 'away': aw/tot}

def estimate_lambdas(h_stats,a_stats):
    h_attack=max(0.1,h_stats[0]); h_def=max(0.1,h_stats[1])
    a_attack=max(0.1,a_stats[0]); a_def=max(0.1,a_stats[1])
    home_lambda=(h_attack*a_def)/2.5*1.08
    away_lambda=(a_attack*h_def)/2.5*0.95
    return max(0.05,home_lambda), max(0.05,away_lambda)

def ensemble_predict(h_stats,a_stats):
    key=f"ens_{h_stats}_{a_stats}"
    cached=cache_get(key)
    if cached: return cached
    homes=[]; draws=[]; aways=[]
    base_h, base_a = estimate_lambdas(h_stats,a_stats)
    for m in range(TOTAL_MODELS):
        jitter_h=random.uniform(0.85,1.15); jitter_a=random.uniform(0.85,1.15); scale=random.uniform(0.85,1.25)
        h_l = base_h * jitter_h * scale; a_l = base_a * jitter_a * scale
        res = simulate_model(h_l,a_l,sims=SIMS_PER_MODEL, seed=(m*13+int(time.time()%1000)))
        homes.append(res['home']); draws.append(res['draw']); aways.append(res['away'])
    final={'home': sum(homes)/len(homes), 'draw': sum(draws)/len(draws), 'away': sum(aways)/len(aways)}
    cache_set(key,final,TTL['predictions']); return final

# store predictions
PRED = {}

def store_prediction(match_id, pred):
    PRED[str(match_id)] = {'pred': pred, 'time': time.time()}

def compare_today_results():
    today=date.today().isoformat(); total=0; correct=0; details=[]
    for lid in LEAGUES.values():
        data=safe_get(f"{API_BASE}/competitions/{lid}/matches", params={"status":"FINISHED","dateFrom":today,"dateTo":today})
        matches=data.get("matches",[]) if data else []
        for m in matches:
            mid=str(m['id']); ft=m.get('score',{}).get('fullTime',{}); hg=ft.get('home'); ag=ft.get('away')
            if hg is None or ag is None: continue
            actual = 'home' if hg>ag else ('draw' if hg==ag else 'away')
            pred = PRED.get(mid)
            if pred:
                p_choice=max(('home','draw','away'), key=lambda k: pred['pred'].get(k,0))
                if p_choice==actual: correct+=1
                total+=1
                details.append({'match': f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}", 'predicted': p_choice, 'actual': actual, 'confidence': pred['pred'].get(p_choice,0)})
    acc = (correct/total*100) if total else None
    return {'accuracy':acc,'total':total,'correct':correct,'details':details}

# Aiogram bot
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

# helpers to build inline menus
def main_menu(page=1):
    kb = types.InlineKeyboardMarkup(row_width=2)
    if page==1:
        kb.add(types.InlineKeyboardButton("Premier League", callback_data="league_2021"),
               types.InlineKeyboardButton("Champions League", callback_data="league_2001"))
        kb.add(types.InlineKeyboardButton("La Liga", callback_data="league_2014"),
               types.InlineKeyboardButton("Serie A", callback_data="league_2019"))
        kb.add(types.InlineKeyboardButton("Bundesliga", callback_data="league_2002"),
               types.InlineKeyboardButton("Next ▶", callback_data="menu_page_2"))
    else:
        kb.add(types.InlineKeyboardButton("Today", callback_data="today"),
               types.InlineKeyboardButton("Live", callback_data="live"))
        kb.add(types.InlineKeyboardButton("Results", callback_data="results"),
               types.InlineKeyboardButton("Standings", callback_data="standings"))
        kb.add(types.InlineKeyboardButton("FPL", callback_data="fpl"),
               types.InlineKeyboardButton("Top Scorers", callback_data="scorers"))
        kb.add(types.InlineKeyboardButton("◀ Back", callback_data="menu_page_1"))
    return kb

@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    await message.answer(f"KickVision v{VERSION} — choose", reply_markup=main_menu(page=1))

@dp.callback_query_handler(lambda c: True)
async def cb_handler(c: types.CallbackQuery):
    data = c.data
    try:
        if data == "menu_page_2":
            await c.message.edit_text("KickVision — Tools", reply_markup=main_menu(page=2)); return
        if data == "menu_page_1":
            await c.message.edit_text("KickVision — Leagues", reply_markup=main_menu(page=1)); return

        if data.startswith("league_"):
            lid = int(data.split("_",1)[1])
            fixtures = get_upcoming_fixtures(lid, days=7, limit=5)
            if not fixtures:
                await c.answer("No upcoming fixtures"); return
            lines=[]
            for m in fixtures:
                h = m['homeTeam']['name']; a = m['awayTeam']['name']; t = m['utcDate'][11:16]
                h_stats = get_team_recent_stats(m['homeTeam']['id']); a_stats = get_team_recent_stats(m['awayTeam']['id'])
                pred = ensemble_predict(h_stats, a_stats)
                store_prediction(m['id'], pred)
                lines.append(f"{h} vs {a} @ {t} UTC\nHome {pred['home']*100:.1f}% | Draw {pred['draw']*100:.1f}% | Away {pred['away']*100:.1f}%")
            await c.message.edit_text("\n\n".join(lines)[:3900]); return

        if data == "today":
            texts=[]
            for name,lid in LEAGUES.items():
                fixtures=get_upcoming_fixtures(lid, days=0, limit=4)
                if not fixtures: continue
                texts.append(f"🏆 {name.title()} 🏆")
                for m in fixtures:
                    h=m['homeTeam']['name']; a=m['awayTeam']['name']; t=m['utcDate'][11:16]
                    h_stats=get_team_recent_stats(m['homeTeam']['id']); a_stats=get_team_recent_stats(m['awayTeam']['id'])
                    pred=ensemble_predict(h_stats,a_stats); store_prediction(m['id'],pred)
                    texts.append(f"{t} UTC — {h} vs {a}\nHome {pred['home']*100:.1f}% | Draw {pred['draw']*100:.1f}% | Away {pred['away']*100:.1f}%")
            await c.message.edit_text("\n\n".join(texts)[:3900]); return

        if data == "live":
            collected=[]
            for lid in LEAGUES.values():
                data_resp = safe_get(f"{API_BASE}/competitions/{lid}/matches", params={"status":"LIVE"})
                matches=data_resp.get("matches",[]) if data_resp else []
                for m in matches:
                    score = m.get("score",{}).get("fullTime",{})
                    collected.append(f"{m['homeTeam']['name']} {score.get('home','-')} - {score.get('away','-')} {m['awayTeam']['name']}")
            await c.message.edit_text("\n".join(collected) if collected else "No live matches right now."); return

        if data == "results":
            comp = compare_today_results()
            if comp['total']==0:
                await c.message.edit_text("No finished matches with stored predictions today."); return
            out=[f"Results comparison — {comp['total']} matches"]
            if comp['accuracy'] is not None:
                out.append(f"Accuracy: {comp['accuracy']:.2f}% ({comp['correct']}/{comp['total']})")
            for d in comp['details'][:10]:
                out.append(f"{d['match']}: predicted {d['predicted']} — actual {d['actual']}")
            await c.message.edit_text("\n".join(out)[:3900]); return

        if data == "standings":
            kb = types.InlineKeyboardMarkup(row_width=2)
            for name,lid in LEAGUES.items():
                kb.add(types.InlineKeyboardButton(name.title(), callback_data=f"stand_{lid}"))
            kb.add(types.InlineKeyboardButton("Back", callback_data="menu_page_2"))
            await c.message.edit_text("Choose league for standings:", reply_markup=kb); return

        if data.startswith("stand_"):
            lid=int(data.split("_",1)[1])
            resp = safe_get(f"{API_BASE}/competitions/{lid}/standings")
            if not resp:
                await c.answer("Standings unavailable"); return
            txt=f"{LEAGUE_DISPLAY.get(lid,'League')} Standings\n"
            for s in resp.get("standings",[]):
                if s.get("type")=="TOTAL":
                    for row in s.get("table",[])[:10]:
                        txt+=f"{row['position']}. {row['team']['name']} — {row['points']} pts\n"
                    break
            await c.message.edit_text(txt[:3900]); return

        if data == "fpl":
            b = safe_get(f"{FPL_BASE}/bootstrap-static/")
            if not b:
                await c.message.edit_text("FPL data unavailable"); return
            top = sorted(b.get("elements",[]), key=lambda x: float(x.get("form") or 0), reverse=True)[:8]
            text = "Top FPL form players (sample):\n"
            for p in top:
                text += f"{p.get('web_name')} — form {p.get('form')} — selected {p.get('selected_by_percent')}\n"
            await c.message.edit_text(text[:3900]); return

        if data == "scorers":
            kb = types.InlineKeyboardMarkup(row_width=2)
            for name,lid in LEAGUES.items():
                kb.add(types.InlineKeyboardButton(name.title(), callback_data=f"scor_{lid}"))
            kb.add(types.InlineKeyboardButton("Back", callback_data="menu_page_2"))
            await c.message.edit_text("Choose a league for top scorers:", reply_markup=kb); return

        if data.startswith("scor_"):
            lid=int(data.split("_",1)[1])
            sdata = safe_get(f"{API_BASE}/competitions/{lid}/scorers")
            if not sdata:
                await c.answer("Scorers unavailable"); return
            lines=[f"{sc['player']['name']} — {sc['team']['name']} — {sc['goals']} goals" for sc in sdata.get("scorers",[])[:10]]
            await c.message.edit_text("\n".join(lines)[:3900]); return

        await c.answer("Unknown action")
    except Exception:
        log.exception("Callback error")
        try:
            await c.answer("Error processing request")
        except:
            pass

# Webhook handlers: aiohttp app
async def on_startup(app):
    if WEBHOOK_URL:
        await bot.set_webhook(WEBHOOK_URL)
        log.info(f"Aiogram webhook set to {WEBHOOK_URL}")

async def on_shutdown(app):
    await bot.delete_webhook()
    await bot.close()

async def handle(request):
    body = await request.text()
    update = types.Update.to_object(json.loads(body))
    await dp.process_update(update)
    return web.Response(text="OK")

# optional daily broadcaster coroutine
async def daily_broadcaster():
    last_date=None
    while True:
        try:
            now_utc = datetime.utcnow()
            today = now_utc.date()
            if now_utc.hour >= 23 and last_date != today and ADMIN_CHAT_IDS:
                comp = compare_today_results()
                if comp['total'] > 0:
                    text = f"Daily results comparison — {comp['total']} matches\n"
                    if comp['accuracy'] is not None:
                        text += f"Accuracy: {comp['accuracy']:.2f}% ({comp['correct']}/{comp['total']})\n\n"
                    for d in comp['details'][:10]:
                        text += f"{d['match']}: predicted {d['predicted']} — actual {d['actual']}\n"
                    for cid in ADMIN_CHAT_IDS:
                        try:
                            await bot.send_message(cid, text[:3900])
                        except Exception:
                            log.exception("notify admin failed")
                last_date = today
        except Exception:
            log.exception("broadcaster error")
        await asyncio.sleep(60*30)

if __name__ == "__main__":
    # run aiohttp webhook
    app = web.Application()
    app.router.add_post(f"/{BOT_TOKEN}", handle)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    # optionally start broadcaster task
    if ADMIN_CHAT_IDS:
        loop = asyncio.get_event_loop()
        loop.create_task(daily_broadcaster())
    log.info("Starting Aiogram webhook app")
    web.run_app(app, host="0.0.0.0", port=PORT)
