#!/usr/bin/env python3
"""
KickVision v1.0.0 — Official Release
100-model ensemble | Typo-proof | /cancel | vs Only | Complete Global Team Support - Expanded Leagues
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
    log.info(f"Loaded {len(TEAM_ALIASES)} aliases from ZIP")
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

# === COMPLETE GLOBAL NON-MAJOR TEAM ALIASES ===
# Full teams from expanded leagues with common aliases
GLOBAL_NON_MAJOR_TEAMS = [
    # SCOTTISH PREMIERSHIP 2024/25 - All 12 teams
    ["Aberdeen FC", "aberdeen", "dons", "the dons"],
    ["Celtic FC", "celtic", "glasgow celtic", "bhoys", "hoops", "the bhoys", "celts"],
    ["Dundee FC", "dundee", "dundee fc", "the dee"],
    ["Dundee United FC", "dundee united", "dundee utd", "tangerines", "the terrors"],
    ["Heart of Midlothian FC", "hearts", "heart of midlothian", "jambos", "hmfc"],
    ["Hibernian FC", "hibernian", "hibs", "hibees"],
    ["Kilmarnock FC", "kilmarnock", "killie"],
    ["Motherwell FC", "motherwell", "well", "steelmen"],
    ["Rangers FC", "rangers", "glasgow rangers", "gers", "teddy bears", "bears", "light blues", "the gers"],
    ["Ross County FC", "ross county", "staggies"],
    ["St Johnstone FC", "st johnstone", "saints"],
    ["St Mirren FC", "st mirren", "buddies", "saints"],

    # SWEDISH ALLSVENSKAN 2024 - All 16 teams
    ["AIK", "aik", "aik stockholm", "gnaget"],
    ["BK Häcken", "hacken", "häcken", "hacken gothenburg", "goteborg hacken", "wasps"],
    ["Djurgårdens IF", "djurgarden", "djurgårdens", "dif", "järnkaminerna"],
    ["GAIS", "gais", "gais gothenburg"],
    ["Hammarby IF", "hammarby", "bajen"],
    ["IF Elfsborg", "elfsborg", "elfsborg boras", "guliganerna"],
    ["IF Brommapojkarna", "brommapojkarna", "bp", "if bp"],
    ["IFK Göteborg", "goteborg", "ifk goteborg", "blåvitt", "änglarna"],
    ["IFK Norrköping", "norrkoping", "ifk norrköping", "peking"],
    ["IK Sirius", "sirius", "ik sirius", "sirius uppsala"],
    ["Kalmar FF", "kalmar", "kalmar ff", "röda bröder"],
    ["Malmö FF", "malmo", "malmö", "malmo ff", "di blåe"],
    ["Västerås SK", "vasteras sk", "vsk", "västerås", "grönvitt"],
    ["Värnamo IF", "varnamo", "ifk varnamo"],
    ["IFK Värnamo", "ifk varnamo", "varnamo"],
    ["Halmstads BK", "halmstad", "halmstads bk", "hbk"],

    # ENGLISH NATIONAL LEAGUE 2024/25 - All 24 teams
    ["Aldershot Town FC", "aldershot", "shots"],
    ["Altrincham FC", "altrincham"],
    ["Barnet FC", "barnet", "bees"],
    ["Boreham Wood FC", "boreham wood", "wood"],
    ["Boston United FC", "boston united"],
    ["Braintree Town FC", "braintree town"],
    ["Chesterfield FC", "chesterfield", "spireites"],
    ["Dagenham & Redbridge FC", "dagenham", "daggers"],
    ["Eastleigh FC", "eastleigh"],
    ["Ebbsfleet United FC", "ebbsfleet", "fleet"],
    ["FC Halifax Town", "halifax town"],
    ["Forest Green Rovers FC", "forest green", "rovers"],
    ["Gateshead FC", "gateshead"],
    ["Hartlepool United FC", "hartlepool", "pools"],
    ["Oldham Athletic AFC", "oldham", "latics"],
    ["Rochdale AFC", "rochdale"],
    ["Solihull Moors FC", "solihull moors", "moors"],
    ["Southend United FC", "southend", "shrimpers"],
    ["York City FC", "york city", "minstermen"],
    ["Woking FC", "woking"],
    ["Wealdstone FC", "wealdstone"],
    ["Fylde FC", "fylde"],
    ["Sutton United FC", "sutton united"],
    ["Tamworth FC", "tamworth"],

    # EGYPTIAN PREMIER LEAGUE 2024/25 - All 18 teams
    ["Al Ahly SC", "al ahly", "ahly", "red devils", "club of the century"],
    ["Al Ittihad Alexandria Club", "al ittihad", "ittihad alexandria"],
    ["Al Masry SC", "al masry", "masry", "green eagles"],
    ["Ceramica Cleopatra FC", "ceramica cleopatra", "ceramica"],
    ["El Geish SC", "el geish", "talaea el gaish", "military"],
    ["El Gouna FC", "el gouna", "gouna"],
    ["ENPPI Club", "enppi", "enppi club"],
    ["Future FC", "future fc", "future"],
    ["Ghazl El Mahalla SC", "ghazl el mahalla"],
    ["Ismaily SC", "ismaily", "daraweesh", "brazil of egypt"],
    ["Modern Future FC", "modern future", "modern sport"],
    ["National Bank of Egypt SC", "national bank", "nbe"],
    ["Pharco FC", "pharco", "pharco fc"],
    ["Pyramids FC", "pyramids fc", "pyramids"],
    ["Smouha SC", "smouha"],
    ["Zamalek SC", "zamalek", "white knights", "royal club"],
    ["ZED FC", "zed fc"],
    ["Haras El Hodood SC", "haras el hodood", "hodood"],

    # BRAZILIAN SERIE A 2024 - All 20 teams
    ["Athletico Paranaense", "athletico pr", "furacão", "rubro-negro"],
    ["Atlético Goianiense", "atletico goianiense", "dragão", "atletico-go"],
    ["Atlético Mineiro", "atletico mineiro", "galo", "atletico-mg"],
    ["Bahia", "bahia", "tricolor", "esquadrao"],
    ["Botafogo FR", "botafogo", "glorioso", "botafogo fr"],
    ["Bragantino", "bragantino", "red bull bragantino", "braga"],
    ["Ceará SC", "ceara", "vozão"],
    ["Corinthians", "corinthians", "timao", "parque são jorge"],
    ["Criciúma EC", "criciuma", "tigre"],
    ["Cruzeiro EC", "cruzeiro", "raposa", "celeste"],
    ["Cuiabá EC", "cuiaba", "dourado"],
    ["Flamengo", "flamengo", "fla", "mengao", "rubro-negro"],
    ["Fluminense FC", "fluminense", "tricolor", "flu"],
    ["Fortaleza EC", "fortaleza", "leão do pici"],
    ["Grêmio FBPA", "gremio", "imortal", "tricolor gaucho"],
    ["Internacional", "internacional", "colorado", "inter"],
    ["Juventude", "juventude", "juvenude"],
    ["Palmeiras", "palmeiras", "verdão", "porco"],
    ["São Paulo FC", "sao paulo", "tricolor", "soberano"],
    ["Vasco da Gama", "vasco", "gigante da colina", "cruzmaltino"],

    # MLS 2024 - All 29 teams
    ["Atlanta United FC", "atlanta united", "atl", "five stripes"],
    ["Austin FC", "austin fc"],
    ["Charlotte FC", "charlotte fc"],
    ["Chicago Fire FC", "chicago fire"],
    ["FC Cincinnati", "fc cincinnati", "cincinnati"],
    ["Columbus Crew", "columbus crew"],
    ["D.C. United", "dc united"],
    ["FC Dallas", "fc dallas", "dallas"],
    ["Houston Dynamo FC", "houston dynamo"],
    ["Sporting Kansas City", "sporting kc", "kansas city"],
    ["LA Galaxy", "la galaxy"],
    ["Los Angeles FC", "lafc"],
    ["Inter Miami CF", "inter miami", "miami"],
    ["Minnesota United FC", "minnesota united"],
    ["CF Montréal", "cf montreal"],
    ["New England Revolution", "new england"],
    ["New York City FC", "nyc fc"],
    ["New York Red Bulls", "new york red bulls"],
    ["Orlando City SC", "orlando city"],
    ["Philadelphia Union", "philadelphia union"],
    ["Portland Timbers", "portland timbers"],
    ["Real Salt Lake", "real salt lake"],
    ["San Jose Earthquakes", "san jose earthquakes"],
    ["Seattle Sounders FC", "seattle sounders"],
    ["St. Louis City SC", "st louis city"],
    ["Toronto FC", "toronto fc"],
    ["Vancouver Whitecaps FC", "vancouver whitecaps"],
    ["Nashville SC", "nashville sc"],
    ["Wrexham AFC", "wrexham", "red dragons", "the dragons"],

    # J-LEAGUE 2024 - All 20 teams
    ["Albirex Niigata", "albirex niigata"],
    ["Avispa Fukuoka", "avispa fukuoka"],
    ["Cerezo Osaka", "cerezo osaka"],
    ["Consadole Sapporo", "consadole sapporo", "hokkaido consadole"],
    ["FC Tokyo", "fc tokyo"],
    ["Gamba Osaka", "gamba osaka"],
    ["Kashima Antlers", "kashima antlers"],
    ["Kawasaki Frontale", "kawasaki frontale"],
    ["Kyoto Sanga FC", "kyoto sanga"],
    ["Machida Zelvia", "machida zelvia"],
    ["Nagoya Grampus", "nagoya grampus"],
    ["Sanfrecce Hiroshima", "sanfrecce hiroshima"],
    ["Shonan Bellmare", "shonan bellmare"],
    ["Urawa Red Diamonds", "urawa reds", "red diamonds"],
    ["Vissel Kobe", "vissel kobe"],
    ["Yokohama F. Marinos", "yokohama marinos"],
    ["Yokohama FC", "yokohama fc"],
    ["FC Gifu", "fc gifu"],
    ["Jubilo Iwata", "jubilo iwata"],
    ["Omiya Ardija", "omiya ardija"],

    # PRIMEIRA LIGA 2024/25 - All 18 teams
    ["AVS Futebol SAD", "avs", "avs futebol"],
    ["Boavista FC", "boavista", "panteras"],
    ["Casa Pia AC", "casa pia"],
    ["CD Nacional", "nacional", "madeira"],
    ["FC Arouca", "arouca"],
    ["FC Famalicão", "famalicao"],
    ["FC Porto", "porto", "dragões"],
    ["Gil Vicente FC", "gil vicente"],
    ["Moreirense FC", "moreirense"],
    ["Os Belenenses SAD", "belenenses"],
    ["Rio Ave FC", "rio ave"],
    ["SC Braga", "braga", "arsenalistas"],
    ["SL Benfica", "benfica", "águias"],
    ["Sporting CP", "sporting", "leões"],
    ["FC Vizela", "vizela"],
    ["Vitória SC", "vitoria", "concelheiros"],
    ["Estoril Praia", "estoril praia"],
    ["Santa Clara", "santa clara"],

    # EREDIVISIE 2024/25 - All 18 teams
    ["AFC Ajax", "ajax", "godenzonen"],
    ["AZ Alkmaar", "az alkmaar"],
    ["FC Groningen", "fc groningen"],
    ["FC Twente", "fc twente"],
    ["FC Utrecht", "fc utrecht"],
    ["Fortuna Sittard", "fortuna sittard"],
    ["Go Ahead Eagles", "go ahead eagles"],
    ["NAC Breda", "nac breda"],
    ["NEC Nijmegen", "nec nijmegen"],
    ["PEC Zwolle", "pec zwolle"],
    ["PSV Eindhoven", "psv", "boeren"],
    ["RKC Waalwijk", "rkc waalwijk"],
    ["SC Heerenveen", "sc heerenveen"],
    ["Sparta Rotterdam", "sparta rotterdam"],
    ["FC Volendam", "fc volendam"],
    ["Vitesse Arnhem", "vitesse"],
    ["Willem II", "willem ii"],
    ["Heracles Almelo", "heracles almelo"],

    # SAUDI PRO LEAGUE 2024/25 - All 18 teams
    ["Al Ahli SFC", "al ahli", "ahli"],
    ["Al Akhdood Club", "al akhdood"],
    ["Al Akhdoud", "al akhdoud"],
    ["Al Ettifaq FC", "al ettifaq"],
    ["Al Fateh SC", "al fateh"],
    ["Al Hilal SFC", "al hilal", "hilal"],
    ["Al Ittihad Club", "al ittihad", "ittihad"],
    ["Al Khaleej FC", "al khaleej"],
    ["Al Nassr FC", "al nassr", "nassr"],
    ["Al Orobah FC", "al orobah"],
    ["Al Raed SFC", "al raed"],
    ["Al Riyadh SC", "al riyadh"],
    ["Al Shabab FC", "al shabab"],
    ["Al Taawoun FC", "al taawoun"],
    ["Al Wehda Mecca", "al wehda"],
    ["Al Fayha FC", "al fayha"],
    ["Damac FC", "damac"],
    ["Al Kholood", "al kholood"],

    # ARGENTINE PRIMERA DIVISIÓN 2024 - All 28 teams
    ["Argentinos Juniors", "argentinos juniors"],
    ["Arsenal de Sarandí", "arsenal sarandi"],
    ["Atlético Tucumán", "atletico tucuman"],
    ["Banfield", "banfield"],
    ["Boca Juniors", "boca juniors", "xeneizes"],
    ["Central Córdoba SdE", "central cordoba"],
    ["Deportivo Riestra", "deportivo riestra"],
    ["Defensa y Justicia", "defensa justicia"],
    ["Estudiantes LP", "estudiantes lp"],
    ["Gimnasia LP", "gimnasia lp"],
    ["Godoy Cruz", "godoy cruz"],
    ["Independiente", "independiente"],
    ["Independiente Rivadavia", "independiente rivadavia"],
    ["Instituto AC Córdoba", "instituto cordoba"],
    ["Lanús", "lanus"],
    ["Newell's Old Boys", "newells"],
    ["Platense", "platense"],
    ["Racing Club", "racing club"],
    ["River Plate", "river plate", "millonarios"],
    ["Rosario Central", "rosario central"],
    ["Sarmiento", "sarmiento"],
    ["Talleres Córdoba", "talleres cordoba"],
    ["Tigre", "tigre"],
    ["Unión Santa Fe", "union santa fe"],
    ["Vélez Sarsfield", "velez sarsfield"],
    ["Barracas Central", "barracas central"],
    ["Huracán", "huracan"],
    ["CA Independiente", "independiente", "rey de copas", "rojo"],

    # LIGA MX 2024/25 - All 18 teams
    ["América", "america", "águilas"],
    ["Atlas", "atlas"],
    ["Atlético San Luis", "atletico san luis"],
    ["Cruz Azul", "cruz azul", "cementeros"],
    ["FC Juárez", "fc juarez"],
    ["León", "leon"],
    ["Mazatlán FC", "mazatlan"],
    ["Monterrey", "monterrey"],
    ["Morelia", "morelia"],
    ["Necaxa", "necaxa"],
    ["Pachuca", "pachuca"],
    ["Puebla", "puebla"],
    ["Pumas UNAM", "pumas unam", "pumas"],
    ["Querétaro", "queretaro"],
    ["Santos Laguna", "santos laguna"],
    ["Tijuana", "tijuana", "xolos"],
    ["Tigres UANL", "tigres", "felinos"],
    ["Toluca", "toluca"],

    # 2. BUNDESLIGA 2024/25 - All 18 teams
    ["1. FC Kaiserslautern", "kaiserslautern", "roten teufel"],
    ["1. FC Nürnberg", "nurnberg", "der club"],
    ["1. FC Saarbrücken", "saarbrucken"],
    ["1. FC Union Berlin", "union berlin", "eisern union"],
    ["1. FC Magdeburg", "magdeburg"],
    ["Eintracht Braunschweig", "braunschweig"],
    ["Elversberg", "elversberg"],
    ["Fortuna Düsseldorf", "duesseldorf", "f95"],
    ["Hansa Rostock", "hansa rostock", "kogge"],
    ["Hertha BSC", "hertha bsc", "die alte dame"],
    ["H Hamburger SV", "hsv", "der dinos"],
    ["Preußen Münster", "preussen munster"],
    ["SC Paderborn 07", "paderborn"],
    ["Schalke 04", "schalke", "die knappen"],
    ["SpVgg Greuther Fürth", "fuerth", "kleeblätter"],
    ["SV Darmstadt 98", "darmstadt", "lilien"],
    ["SV Elversberg", "elversberg"],
    ["1. FC Köln", "koeln", "geissboecke"],

    # SERIE B 2024/25 - All 20 teams
    ["Ascoli FC", "ascoli"],
    ["Brescia Calcio", "brescia"],
    ["Catanzaro", "catanzaro"],
    ["Cesena FC", "cesena"],
    ["Cittadella", "cittadella"],
    ["Cosenza Calcio", "cosenza"],
    ["Cremonese", "cremonese"],
    ["Feralpisalo", "feralpisalo"],
    ["Mantova 1911", "mantova"],
    ["Modena FC", "modena"],
    ["Novara FC", "novara"],
    ["Palermo FC", "palermo"],
    ["Parma Calcio 1913", "parma", "ducale"],
    ["Pisa SC", "pisa"],
    ["Reggiana 1919", "reggiana"],
    ["Sampdoria", "sampdoria", "blucerchiati"],
    ["Salernitana", "salernitana"],
    ["Sassuolo", "sassuolo"],
    ["Südtirol", "sudtirol"],
    ["Venezia FC", "venezia"],

    # LIGUE 2 2024/25 - All 20 teams
    ["Amiens SC", "amiens"],
    ["AJ Auxerre", "auxerre"],
    ["Annecy FC", "annecy"],
    ["AS Nancy Lorraine", "nancy", "lorrains"],
    ["Bastia", "bastia"],
    ["Bordeaux", "bordeaux", "girondins"],
    ["Caen", "caen"],
    ["Châteauroux", "chateauroux"],
    ["Dijon FCO", "dijon"],
    ["EN Aveyron", "en avevron"],
    ["Estac Troyes", "troyes"],
    ["Guingamp", "guingamp"],
    ["Laval", "laval"],
    ["Martigues", "martigues"],
    ["Paris FC", "paris fc"],
    ["Pau FC", "pau"],
    ["Quevilly Rouen", "quevilly rouen"],
    ["Red Star FC", "red star"],
    ["Rodez AF", "rodez"],
    ["US Concarneau", "concarneau"],

    # CHAMPIONSHIP 2024/25 - All 24 teams
    ["Blackburn Rovers FC", "blackburn", "rovers"],
    ["Burnley FC", "burnley", "clarets"],
    ["Cardiff City FC", "cardiff", "bluebirds"],
    ["Coventry City FC", "coventry", "sky blues"],
    ["Derby County FC", "derby", "rams"],
    ["Hull City AFC", "hull", "tigers"],
    ["Leeds United FC", "leeds", "whites"],
    ["Luton Town FC", "luton", "hatters"],
    ["Middlesbrough FC", "middlesbrough", "boro"],
    ["Millwall FC", "millwall", "lions"],
    ["Norwich City FC", "norwich", "canaries"],
    ["Oxford United FC", "oxford", "u’s"],
    ["Plymouth Argyle FC", "plymouth", "pilgrims"],
    ["Portsmouth FC", "portsmouth", "pompey"],
    ["Preston North End FC", "preston", "lilywhites"],
    ["Queens Park Rangers FC", "qpr", "hoops"],
    ["Sheffield United FC", "sheffield united", "blades"],
    ["Sheffield Wednesday FC", "sheffield wednesday", "owls"],
    ["Stoke City FC", "stoke", "potters"],
    ["Sunderland AFC", "sunderland", "black cats"],
    ["Swansea City AFC", "swansea", "swans"],
    ["Watford FC", "watford", "hornets"],
    ["West Bromwich Albion FC", "west brom", "baggies"],
    ["Bristol City FC", "bristol city", "robins"],

    # LA LIGA 2 2024/25 - All 22 teams
    ["Albacete Balompié", "albacete"],
    ["CD Castellón", "castellon"],
    ["CD Eldense", "eldense"],
    ["CD Leganés", "leganes"],
    ["CD Mirandés", "mirandes"],
    ["CD Tenerife", "tenerife"],
    ["CF Reus Deportiu", "reus"],
    ["Deportivo Alavés", "alaves"],
    [ "Deportivo de La Coruña", "deportivo la coruna", "depor"],
    ["Elche CF", "elche"],
    ["FC Andorra", "andorra"],
    ["FC Cartagena", "cartagena"],
    ["Girona FC", "girona"],
    ["Levante UD", "levante"],
    ["Racing Ferrol", "racing ferrol"],
    ["Racing Santander", "racing santander", "montañeses"],
    ["Real Oviedo", "oviedo"],
    ["SD Amorebieta", "amorebieta"],
    ["SD Eibar", "eibar"],
    ["SD Huesca", "huesca"],
    ["SD Logroñés", "logrones"],
    ["Sporting Gijón", "sporting gijon"]
]

# Inject all non-major aliases
for team in GLOBAL_NON_MAJOR_TEAMS:
    official = team[0]
    for alias in team[1:]:
        TEAM_ALIASES[alias.lower()] = official
log.info(f"Injected {len(GLOBAL_NON_MAJOR_TEAMS)} non-major teams with aliases")

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
        "Example: `Gers vs Bhoys` or `Hacken vs Malmo`\n\n"
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
        bot.reply_to(m, "Use **Team A vs Team B** format\nExample: `Gers vs Bhoys`\nType **/how** for details")
        return

    parts = re.split(r'\s+vs\s+|\s+[-–—]\s+', txt, re.IGNORECASE)
    home = parts[0].strip()
    away = ' '.join(parts[1:]).strip()

    home_cands = find_team_candidates(home)
    away_cands = find_team_candidates(away)

    if not home_cands or not away_cands:
        bot.reply_to(m, f"Couldn't find: `{home}` or `{away}`.\nTry a different spelling or major league match: `Chelsea vs Man U`\nType **/how** for tips")
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
    log.info("KickVision v1.0.0 STARTED — Complete Global Team Support Active")
    
    bot.remove_webhook()
    time.sleep(1)
    webhook_url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/{BOT_TOKEN}"
    bot.set_webhook(url=webhook_url)
    log.info(f"Webhook set: {webhook_url}")

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
