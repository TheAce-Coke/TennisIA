import os
import pandas as pd
import json
import glob
from kaggle.api.kaggle_api_extended import KaggleApi

# --- CONFIGURACIÓN ---
DATASET_KAGGLE = "eoinamoore/historical-nba-data-and-player-box-scores"
ARCHIVO_SALIDA = "nba_games.csv"

# Configuración de Autenticación para GitHub Actions o Local
# Si existe la variable de entorno KAGGLE_JSON (en GitHub), la usamos para crear el archivo credentials
if "KAGGLE_JSON" in os.environ:
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
        f.write(os.environ["KAGGLE_JSON"])
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

print("==========================================================")
print("   🏀 ACTUALIZADOR NBA (FUENTE: KAGGLE) 🏀")
print("==========================================================")

try:
    print(f"⬇️  Autenticando en Kaggle y descargando {DATASET_KAGGLE}...")
    api = KaggleApi()
    api.authenticate()
    
    # Descargar y descomprimir en la carpeta actual
    api.dataset_download_files(DATASET_KAGGLE, path=".", unzip=True)
    print("✅ Descarga y descompresión completada.")

    # --- BÚSQUEDA Y PROCESADO ---
    # Kaggle suele cambiar los nombres de los archivos. Buscamos el CSV principal.
    # En este dataset suele haber un 'game_logs.csv' o similar.
    csv_files = glob.glob("*.csv")
    target_file = None
    
    # Buscamos un archivo que parezca tener logs de partidos
    for f in csv_files:
        if "game" in f.lower() or "box_score" in f.lower():
            target_file = f
            break
    
    if not target_file:
        # Fallback: cogemos el más grande
        target_file = max(csv_files, key=os.path.getsize)

    print(f"📂 Procesando archivo: {target_file}")
    df = pd.read_csv(target_file)

    # --- NORMALIZACIÓN DE COLUMNAS (CRÍTICO) ---
    # La IA espera nombres específicos (UPPERCASE). Kaggle suele usar lowercase.
    # Hacemos un mapa de renombrado inteligente.
    
    # Convertimos todo a mayúsculas primero para facilitar
    df.columns = [c.upper() for c in df.columns]
    
    # Mapa de columnas necesarias para NeuralSports
    mapa_cols = {
        'GAME_DATE': ['DATE', 'GAMEDATE'],
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
            # Si ya existe la columna exacta, perfecto
            if std_col in df.columns:
                rename_dict[std_col] = std_col
                break
    
    df.rename(columns=rename_dict, inplace=True)
    
    # Verificar que tenemos las esenciales
    required = ['GAME_DATE', 'PTS', 'TEAM_NAME']
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        print(f"❌ Error: El dataset de Kaggle no tiene las columnas: {missing}")
        print(f"Columnas encontradas: {list(df.columns)}")
        exit()

    # Filtrar y ordenar
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df.sort_values('GAME_DATE', inplace=True)
        
        # Filtrar solo desde 2020 para mantener consistencia con la IA
        df = df[df['GAME_DATE'].dt.year >= 2020]

    df.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"✅ Base de datos NBA guardada: {len(df)} registros.")

except Exception as e:
    print(f"❌ Error descargando de Kaggle: {e}")
    # Generar archivo vacío para no romper el workflow si falla
    if not os.path.exists(ARCHIVO_SALIDA):
        pd.DataFrame(columns=['GAME_DATE', 'MATCHUP']).to_csv(ARCHIVO_SALIDA, index=False)