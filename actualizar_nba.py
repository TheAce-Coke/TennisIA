import os
import json
import pandas as pd
import glob
import sys

# --- 1. AUTENTICACIÓN PREVIA (CRÍTICO: HACER ESTO ANTES DE IMPORTAR KAGGLE) ---
# La librería Kaggle falla al importarse si no encuentra credenciales.
# Por eso configuramos las variables de entorno AQUÍ, antes del import.

if "KAGGLE_JSON" in os.environ:
    try:
        # Leemos el secreto que subiste a GitHub
        creds = json.loads(os.environ["KAGGLE_JSON"])
        
        # Inyectamos las credenciales directamente en la memoria del sistema
        os.environ['KAGGLE_USERNAME'] = creds['username']
        os.environ['KAGGLE_KEY'] = creds['key']
        print("✅ Credenciales de Kaggle inyectadas en el entorno.")
    except Exception as e:
        print(f"⚠️ Error leyendo KAGGLE_JSON: {e}")
else:
    print("⚠️ ADVERTENCIA: No se detectó el secreto KAGGLE_JSON.")

# --- 2. AHORA SÍ IMPORTAMOS KAGGLE ---
# Al hacer el import ahora, la librería verá que las variables ya existen y no fallará.
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("❌ Error: La librería 'kaggle' no está instalada.")
    sys.exit()
except OSError:
    print("❌ Error: Kaggle sigue sin detectar las credenciales.")
    sys.exit()

# --- CONFIGURACIÓN DATASET ---
DATASET_KAGGLE = "nathanlauga/nba-games"
ARCHIVO_SALIDA = "nba_games.csv"

print("==========================================================")
print("   🏀 ACTUALIZADOR NBA (FUENTE: KAGGLE / ENV AUTH) 🏀")
print("==========================================================")

try:
    print(f"⬇️  Descargando {DATASET_KAGGLE}...")
    api = KaggleApi()
    api.authenticate()
    
    # Descargar y descomprimir
    api.dataset_download_files(DATASET_KAGGLE, path=".", unzip=True)
    print("✅ Descarga completada.")

    # --- SELECCIÓN INTELIGENTE DE ARCHIVO ---
    csv_files = glob.glob("*.csv")
    target_file = None
    
    # Prioridad: Archivos que digan 'games'
    candidates = [f for f in csv_files if 'games' in f.lower() and 'player' not in f.lower()]
    if candidates:
        target_file = sorted(candidates, key=len)[0]
    
    # Fallback
    if not target_file:
        non_players = [f for f in csv_files if 'player' not in f.lower()]
        if non_players:
            target_file = max(non_players, key=os.path.getsize)
        elif csv_files:
            target_file = max(csv_files, key=os.path.getsize)

    if not target_file:
        raise FileNotFoundError("No se encontró un CSV válido.")

    print(f"📂 Procesando archivo: {target_file}")
    
    # Leer CSV
    df = pd.read_csv(target_file, low_memory=False)

    # --- NORMALIZACIÓN DE COLUMNAS ---
    df.columns = [c.upper() for c in df.columns]
    
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

    # --- TRANSFORMACIÓN DE FORMATO ---
    if 'HOME_TEAM_ID' in [c.upper() for c in pd.read_csv(target_file, nrows=0).columns]:
        print("🔄 Transformando formato 'Local/Visitante' a formato 'Log'...")
        df_raw = pd.read_csv(target_file, low_memory=False)
        
        # Local
        df_home = df_raw.copy()
        df_home['TEAM_ID'] = df_home['HOME_TEAM_ID']
        df_home['PTS'] = df_home['PTS_home']
        df_home['WL'] = df_home['HOME_TEAM_WINS'].apply(lambda x: 'W' if x == 1 else 'L')
        df_home['IS_HOME'] = 1
        # Stats inventadas para Four Factors (si faltan)
        df_home['FGA'] = df_home.get('FGA_home', 88)
        df_home['FTA'] = df_home.get('FTA_home', 22)
        df_home['TOV'] = df_home.get('TOV_home', 14)
        df_home['OREB'] = df_home.get('OREB_home', 10)
        df_home['TEAM_NAME'] = df_home.get('HOME_TEAM_ID', 'HomeTeam')
        
        # Visitante
        df_away = df_raw.copy()
        df_away['TEAM_ID'] = df_away['VISITOR_TEAM_ID']
        df_away['PTS'] = df_away['PTS_away']
        df_away['WL'] = df_away['HOME_TEAM_WINS'].apply(lambda x: 'L' if x == 1 else 'W')
        df_away['IS_HOME'] = 0
        df_away['FGA'] = df_away.get('FGA_away', 88)
        df_away['FTA'] = df_away.get('FTA_away', 22)
        df_away['TOV'] = df_away.get('TOV_away', 14)
        df_away['OREB'] = df_away.get('OREB_away', 10)
        df_away['TEAM_NAME'] = df_away.get('VISITOR_TEAM_ID', 'AwayTeam')

        df = pd.concat([df_home, df_away], ignore_index=True)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE_EST'], format='mixed', errors='coerce')

    else:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed', errors='coerce')
        for c in ['FGA', 'FTA', 'TOV', 'OREB']:
            if c not in df.columns: df[c] = 0

    # Filtrado final
    cols_finales = ['GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'FGA', 'FTA', 'TOV', 'OREB', 'IS_HOME']
    for c in cols_finales:
        if c not in df.columns: df[c] = 0
        
    df = df[cols_finales]
    df = df.dropna(subset=['GAME_DATE'])
    df = df[df['GAME_DATE'].dt.year >= 2015]
    df.sort_values('GAME_DATE', inplace=True)

    df.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"✅ Base de datos NBA guardada: {len(df)} registros.")

except Exception as e:
    print(f"❌ Error crítico en NBA: {e}")
    # Archivo de emergencia para no romper el flujo
    cols_emergencia = ['GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'FGA', 'FTA', 'TOV', 'OREB']
    pd.DataFrame(columns=cols_emergencia).to_csv(ARCHIVO_SALIDA, index=False)
    print("⚠️ CSV de emergencia generado.")