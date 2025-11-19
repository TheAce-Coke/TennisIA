import os
import json
import pandas as pd
import glob
import sys

# --- DICCIONARIO OFICIAL ID -> NOMBRE ---
# Esto garantiza que salgan nombres reales aunque el CSV solo traiga números
NBA_TEAMS = {
    1610612737: "Atlanta Hawks", 1610612738: "Boston Celtics", 1610612739: "Cleveland Cavaliers",
    1610612740: "New Orleans Pelicans", 1610612741: "Chicago Bulls", 1610612742: "Dallas Mavericks",
    1610612743: "Denver Nuggets", 1610612744: "Golden State Warriors", 1610612745: "Houston Rockets",
    1610612746: "LA Clippers", 1610612747: "Los Angeles Lakers", 1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks", 1610612750: "Minnesota Timberwolves", 1610612751: "Brooklyn Nets",
    1610612752: "New York Knicks", 1610612753: "Orlando Magic", 1610612754: "Indiana Pacers",
    1610612755: "Philadelphia 76ers", 1610612756: "Phoenix Suns", 1610612757: "Portland Trail Blazers",
    1610612758: "Sacramento Kings", 1610612759: "San Antonio Spurs", 1610612760: "Oklahoma City Thunder",
    1610612761: "Toronto Raptors", 1610612762: "Utah Jazz", 1610612763: "Memphis Grizzlies",
    1610612764: "Washington Wizards", 1610612765: "Detroit Pistons", 1610612766: "Charlotte Hornets"
}

# --- 1. AUTENTICACIÓN PREVIA ---
if "KAGGLE_JSON" in os.environ:
    try:
        creds = json.loads(os.environ["KAGGLE_JSON"])
        os.environ['KAGGLE_USERNAME'] = creds['username']
        os.environ['KAGGLE_KEY'] = creds['key']
        print("✅ Credenciales de Kaggle inyectadas.")
    except Exception as e:
        print(f"⚠️ Error leyendo KAGGLE_JSON: {e}")

# --- 2. IMPORTAR KAGGLE ---
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("❌ Error: Instala kaggle con 'pip install kaggle'")
    sys.exit()
except OSError:
    # Si falla en local sin variables, intenta seguir si ya hay archivo
    pass

# --- CONFIGURACIÓN ---
DATASET_KAGGLE = "nathanlauga/nba-games"
ARCHIVO_SALIDA = "nba_games.csv"

print("==========================================================")
print("   🏀 ACTUALIZADOR NBA (CON NOMBRES REALES) 🏀")
print("==========================================================")

try:
    print(f"⬇️  Descargando {DATASET_KAGGLE}...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET_KAGGLE, path=".", unzip=True)
    print("✅ Descarga completada.")

    # --- SELECCIÓN DE ARCHIVO ---
    csv_files = glob.glob("*.csv")
    target_file = None
    candidates = [f for f in csv_files if 'games' in f.lower() and 'player' not in f.lower()]
    if candidates: target_file = sorted(candidates, key=len)[0]
    
    if not target_file:
        # Fallback
        non_players = [f for f in csv_files if 'player' not in f.lower()]
        if non_players: target_file = max(non_players, key=os.path.getsize)
        elif csv_files: target_file = max(csv_files, key=os.path.getsize)

    if not target_file: raise FileNotFoundError("No CSV found.")
    print(f"📂 Procesando: {target_file}")
    
    df = pd.read_csv(target_file, low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    
    # --- RENOMBRADO ---
    mapa_cols = {
        'GAME_DATE': ['GAME_DATE_EST', 'DATE', 'GAMEDATE'],
        'MATCHUP': ['MATCHUP', 'MATCH_UP'],
        'WL': ['HOME_TEAM_WINS', 'W_L', 'WL'],
        'PTS': ['PTS_home', 'PTS', 'POINTS'],
        'TEAM_ID': ['HOME_TEAM_ID', 'TEAM_ID'],
        'GAME_ID': ['GAME_ID']
    }
    
    rename_dict = {}
    for std_col, candidates in mapa_cols.items():
        for cand in candidates:
            if cand in df.columns:
                rename_dict[cand] = std_col
                break
    df.rename(columns=rename_dict, inplace=True)

    # --- TRANSFORMACIÓN ANCHO -> LARGO ---
    if 'HOME_TEAM_ID' in [c.upper() for c in pd.read_csv(target_file, nrows=0).columns]:
        print("🔄 Normalizando estructura...")
        df_raw = pd.read_csv(target_file, low_memory=False)
        
        # Local
        df_h = df_raw.copy()
        df_h['TEAM_ID'] = df_h['HOME_TEAM_ID']
        df_h['PTS'] = df_h['PTS_home']
        df_h['WL'] = df_h['HOME_TEAM_WINS'].apply(lambda x: 'W' if x == 1 else 'L')
        df_h['IS_HOME'] = 1
        # Stats
        df_h['FGA'] = df_h.get('FGA_home', 88)
        df_h['FTA'] = df_h.get('FTA_home', 22)
        df_h['TOV'] = df_h.get('TOV_home', 14)
        df_h['OREB'] = df_h.get('OREB_home', 10)
        
        # Visitante
        df_a = df_raw.copy()
        df_a['TEAM_ID'] = df_a['VISITOR_TEAM_ID']
        df_a['PTS'] = df_a['PTS_away']
        df_a['WL'] = df_a['HOME_TEAM_WINS'].apply(lambda x: 'L' if x == 1 else 'W')
        df_a['IS_HOME'] = 0
        df_a['FGA'] = df_a.get('FGA_away', 88)
        df_a['FTA'] = df_a.get('FTA_away', 22)
        df_a['TOV'] = df_a.get('TOV_away', 14)
        df_a['OREB'] = df_a.get('OREB_away', 10)

        df = pd.concat([df_h, df_a], ignore_index=True)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE_EST'], format='mixed', errors='coerce')
    else:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed', errors='coerce')
        for c in ['FGA', 'FTA', 'TOV', 'OREB']: 
            if c not in df.columns: df[c] = 0

    # --- TRADUCCIÓN DE ID A NOMBRE (LA CLAVE DE LA SOLUCIÓN) ---
    print("📝 Traduciendo IDs a Nombres de Equipos...")
    df['TEAM_NAME'] = df['TEAM_ID'].map(NBA_TEAMS)
    
    # Rellenar desconocidos (por si acaso hay All-Star teams u otros IDs raros)
    df['TEAM_NAME'] = df['TEAM_NAME'].fillna("Unknown Team (" + df['TEAM_ID'].astype(str) + ")")

    # --- FILTRADO FINAL ---
    cols = ['GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'FGA', 'FTA', 'TOV', 'OREB', 'IS_HOME']
    for c in cols:
        if c not in df.columns: df[c] = 0
    
    df = df[cols]
    df = df.dropna(subset=['GAME_DATE'])
    df = df[df['GAME_DATE'].dt.year >= 2015]
    df.sort_values('GAME_DATE', inplace=True)

    df.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"✅ Base de datos NBA guardada: {len(df)} registros.")

except Exception as e:
    print(f"❌ Error: {e}")
    cols = ['GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'FGA', 'FTA', 'TOV', 'OREB']
    pd.DataFrame(columns=cols).to_csv(ARCHIVO_SALIDA, index=False)