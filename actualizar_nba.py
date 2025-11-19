import os
import json
import pandas as pd
import glob
from kaggle.api.kaggle_api_extended import KaggleApi

# --- CONFIGURACIÓN DE CREDENCIALES ---
if "KAGGLE_JSON" in os.environ:
    try:
        creds = json.loads(os.environ["KAGGLE_JSON"])
        os.environ['KAGGLE_USERNAME'] = creds['username']
        os.environ['KAGGLE_KEY'] = creds['key']
        print("✅ Credenciales de Kaggle configuradas.")
    except Exception as e:
        print(f"⚠️ Error procesando KAGGLE_JSON: {e}")

# --- CONFIGURACIÓN ---
# Usamos un dataset alternativo que suele ser más limpio para Game Data si el anterior falla
DATASET_KAGGLE = "nathanlauga/nba-games" 
ARCHIVO_SALIDA = "nba_games.csv"

print("==========================================================")
print("   🏀 ACTUALIZADOR NBA (FUENTE: KAGGLE V2) 🏀")
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
    print(f"📂 Archivos disponibles: {csv_files}")
    
    target_file = None
    
    # 1. Prioridad: Archivos que digan 'games' pero NO 'player' ni 'details'
    candidates = [f for f in csv_files if 'games' in f.lower() and 'player' not in f.lower() and 'detail' not in f.lower()]
    
    if candidates:
        # Preferimos el que tenga 'nba' o sea más corto (generalmente 'games.csv')
        target_file = sorted(candidates, key=len)[0]
    
    # 2. Fallback: Buscar 'ranking' o 'standings' si no hay games (aunque no ideal)
    if not target_file:
        candidates = [f for f in csv_files if 'rank' in f.lower()]
        if candidates: target_file = candidates[0]

    if not target_file and csv_files:
        # Último recurso: el más grande que NO sea de jugadores
        non_player = [f for f in csv_files if 'player' not in f.lower()]
        if non_player:
            target_file = max(non_player, key=os.path.getsize)
        else:
            target_file = max(csv_files, key=os.path.getsize)

    if not target_file:
        raise FileNotFoundError("No se encontró un CSV válido de partidos.")

    print(f"👉 Seleccionado: {target_file}")
    
    # Leer CSV (low_memory=False evita warnings de tipos mixtos)
    df = pd.read_csv(target_file, low_memory=False)

    # --- NORMALIZACIÓN ROBUSTA ---
    df.columns = [c.upper() for c in df.columns]
    
    # Mapa de columnas (Ajustado para datasets comunes de NBA en Kaggle)
    mapa_cols = {
        'GAME_DATE': ['GAME_DATE_EST', 'DATE', 'GAMEDATE'],
        'MATCHUP': ['MATCHUP', 'MATCH_UP'],
        'WL': ['HOME_TEAM_WINS', 'W_L', 'WL'], # Algunos datasets usan 1/0 en HOME_TEAM_WINS
        'PTS': ['PTS_home', 'PTS', 'POINTS'], # Ojo aquí, necesitamos distinguir local/visitante si es estructura ancha
        'TEAM_ID': ['HOME_TEAM_ID', 'TEAM_ID'],
        'GAME_ID': ['GAME_ID']
    }
    
    # Renombrado
    rename_dict = {}
    for std_col, candidates in mapa_cols.items():
        for cand in candidates:
            if cand in df.columns:
                rename_dict[cand] = std_col
                break
    
    df.rename(columns=rename_dict, inplace=True)

    # --- ADAPTACIÓN DE ESTRUCTURA ---
    # Muchos datasets de Kaggle vienen en formato "un partido por fila" (Local vs Visitante)
    # Nuestro sistema espera formato "un equipo por fila" (log).
    # Si detectamos columnas _home y _away, transformamos.
    
    if 'HOME_TEAM_ID' in [c.upper() for c in pd.read_csv(target_file, nrows=0).columns]:
        print("🔄 Detectado formato 'Ancho'. Transformando a formato 'Log'...")
        # Recargar original para mapeo limpio
        df_raw = pd.read_csv(target_file, low_memory=False)
        
        # Crear registros HOME
        df_home = df_raw.copy()
        df_home['TEAM_ID'] = df_home['HOME_TEAM_ID']
        df_home['PTS'] = df_home['PTS_home']
        df_home['WL'] = df_home['HOME_TEAM_WINS'].apply(lambda x: 'W' if x == 1 else 'L')
        df_home['IS_HOME'] = 1
        # Inventar columnas faltantes con medias si no existen
        df_home['FGA'] = df_home.get('FGA_home', 88) 
        df_home['FTA'] = df_home.get('FTA_home', 22)
        df_home['TOV'] = df_home.get('TOV_home', 14)
        df_home['OREB'] = df_home.get('OREB_home', 10)
        df_home['TEAM_NAME'] = df_home.get('HOME_TEAM_ID', 'HomeTeam') # ID temporal
        
        # Crear registros AWAY
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

        # Unir y usar columnas estándar
        df = pd.concat([df_home, df_away], ignore_index=True)
        
        # Mapear Fechas
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE_EST'], format='mixed', errors='coerce')
        
        # Columnas finales necesarias
        cols_needed = ['GAME_DATE', 'GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'PTS', 'WL', 'FGA', 'FTA', 'TOV', 'OREB', 'IS_HOME']
        # Rellenar faltantes
        for c in cols_needed:
            if c not in df.columns: df[c] = 0 
            
        df = df[cols_needed]

    else:
        # Formato Log Estándar (como el de eoinamoore si coges el archivo correcto)
        # Limpieza de Fechas con 'mixed' para evitar error 2025 format
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed', errors='coerce')
        
        # Asegurar columnas numéricas críticas
        for col in ['FGA', 'FTA', 'TOV', 'OREB']:
            if col not in df.columns:
                df[col] = 0 # Rellenar con 0 si faltan para no romper Four Factors
        
    # Filtrar años válidos
    df = df.dropna(subset=['GAME_DATE'])
    df = df[df['GAME_DATE'].dt.year >= 2015]
    df.sort_values('GAME_DATE', inplace=True)

    # Guardar
    df.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"✅ Base de datos NBA guardada: {len(df)} registros.")

except Exception as e:
    print(f"❌ Error crítico: {e}")
    # Crear CSV de emergencia con TODAS las columnas necesarias para que crear_ia_nba no falle
    cols_emergencia = ['GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'FGA', 'FTA', 'TOV', 'OREB']
    pd.DataFrame(columns=cols_emergencia).to_csv(ARCHIVO_SALIDA, index=False)
    print("⚠️ Archivo de emergencia creado (Columnas vacías).")