import os
import json
import pandas as pd
import glob

# --- CONFIGURACIÓN DE CREDENCIALES (CRÍTICO) ---
# Esto debe ir ANTES de importar o usar KaggleApi para evitar el error
if "KAGGLE_JSON" in os.environ:
    try:
        # Leemos el secreto de GitHub
        creds = json.loads(os.environ["KAGGLE_JSON"])
        # Configuramos las variables que la librería Kaggle espera
        os.environ['KAGGLE_USERNAME'] = creds['username']
        os.environ['KAGGLE_KEY'] = creds['key']
        print("✅ Credenciales de Kaggle configuradas desde Secret.")
    except Exception as e:
        print(f"⚠️ Error procesando KAGGLE_JSON: {e}")

# Ahora sí importamos la API
from kaggle.api.kaggle_api_extended import KaggleApi

# --- CONFIGURACIÓN ---
DATASET_KAGGLE = "eoinamoore/historical-nba-data-and-player-box-scores"
ARCHIVO_SALIDA = "nba_games.csv"

print("==========================================================")
print("   🏀 ACTUALIZADOR NBA (FUENTE: KAGGLE) 🏀")
print("==========================================================")

try:
    print(f"⬇️  Conectando a Kaggle y descargando {DATASET_KAGGLE}...")
    api = KaggleApi()
    api.authenticate()
    
    # Descargar y descomprimir
    api.dataset_download_files(DATASET_KAGGLE, path=".", unzip=True)
    print("✅ Descarga completada.")

    # --- BÚSQUEDA DE ARCHIVO ---
    csv_files = glob.glob("*.csv")
    target_file = None
    
    # Buscamos archivos relevantes
    for f in csv_files:
        if "game" in f.lower() and "nba" in f.lower():
            target_file = f
            break
    
    if not target_file and csv_files:
        target_file = max(csv_files, key=os.path.getsize) # Fallback: el más grande

    if not target_file:
        raise FileNotFoundError("No se encontró ningún CSV en el zip descargado.")

    print(f"📂 Procesando archivo: {target_file}")
    df = pd.read_csv(target_file)

    # --- NORMALIZACIÓN DE COLUMNAS ---
    df.columns = [c.upper() for c in df.columns]
    
    mapa_cols = {
        'GAME_DATE': ['DATE', 'GAMEDATE', 'GAME_DATE_EST'],
        'MATCHUP': ['MATCHUP', 'MATCH_UP'],
        'WL': ['W_L', 'WIN_LOSS', 'WL'],
        'PTS': ['PTS', 'POINTS'],
        'FGA': ['FGA'],
        'FTA': ['FTA'],
        'TOV': ['TOV', 'TURNOVERS'],
        'OREB': ['OREB', 'ORB', 'OFF_REB'],
        'TEAM_NAME': ['TEAM', 'TEAM_NAME'],
        'TEAM_ID': ['TEAM_ID'],
        'GAME_ID': ['GAME_ID']
    }
    
    rename_dict = {}
    for std_col, candidates in mapa_cols.items():
        for cand in candidates:
            if cand in df.columns:
                rename_dict[cand] = std_col
                break
    
    df.rename(columns=rename_dict, inplace=True)
    
    # Filtrar años recientes (desde 2020)
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df[df['GAME_DATE'].dt.year >= 2020]
        df.sort_values('GAME_DATE', inplace=True)

    df.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"✅ Base de datos NBA guardada: {len(df)} registros.")

except Exception as e:
    print(f"❌ Error en proceso Kaggle: {e}")
    # Generar archivo vacío de emergencia para no romper el workflow
    if not os.path.exists(ARCHIVO_SALIDA):
        print("⚠️ Generando archivo vacío de emergencia.")
        pd.DataFrame(columns=['GAME_DATE', 'MATCHUP']).to_csv(ARCHIVO_SALIDA, index=False)