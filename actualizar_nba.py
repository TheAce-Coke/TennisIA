import pandas as pd
import time
import random
from nba_api.stats.endpoints import leaguegamelog
import requests.adapters

# --- CONFIGURACIÓN ROBUSTA ---
ARCHIVO_SALIDA = "nba_games.csv"
TEMPORADAS = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
TIMEOUT_SEG = 100  # Subimos tiempo de espera a 100 segundos
MAX_RETRIES = 3    # Intentos por temporada

# Headers para parecer un navegador real y evitar bloqueo
HEADERS_CUSTOM = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/',
    'Connection': 'keep-alive'
}

print("==========================================================")
print("   🏀 ACTUALIZADOR NBA (MODO ROBUSTO) 🏀")
print("==========================================================")

dfs = []

def descargar_con_reintentos(season, tipo):
    intentos = 0
    while intentos < MAX_RETRIES:
        try:
            print(f"   ⏳ Intento {intentos+1} para {season} ({tipo})...", end=" ")
            
            # Llamada a la API con Timeout ampliado
            log = leaguegamelog.LeagueGameLog(
                season=season, 
                season_type_all_star=tipo,
                headers=HEADERS_CUSTOM,
                timeout=TIMEOUT_SEG 
            )
            df = log.get_data_frames()[0]
            df['SeasonType'] = tipo
            print(f"✅ Ok ({len(df)} juegos)")
            return df
            
        except Exception as e:
            print(f"❌ Fallo: {e}")
            intentos += 1
            # Espera exponencial: 5s, 10s, 15s... para que no nos baneen
            wait_time = 5 * intentos + random.randint(1, 3)
            time.sleep(wait_time)
            
    print(f"   ⚠️ IMPOSIBLE descargar {season} {tipo} tras {MAX_RETRIES} intentos.")
    return None

# --- PROCESO DE DESCARGA ---
for temp in TEMPORADAS:
    print(f"⬇️ Procesando {temp}...")
    
    # 1. Regular Season
    df_reg = descargar_con_reintentos(temp, 'Regular Season')
    if df_reg is not None: dfs.append(df_reg)
    
    # Pausa de seguridad entre llamadas
    time.sleep(random.randint(2, 5))
    
    # 2. Playoffs
    df_play = descargar_con_reintentos(temp, 'Playoffs')
    if df_play is not None: dfs.append(df_play)
    
    print("-" * 30)

if not dfs:
    print("❌ ERROR CRÍTICO: No se ha podido descargar NINGÚN dato de la NBA.")
    # Creamos un CSV vacío para que el workflow no explote en el siguiente paso, aunque no funcionará la IA
    pd.DataFrame(columns=['GAME_DATE', 'MATCHUP']).to_csv(ARCHIVO_SALIDA, index=False)
    exit()

# --- FUSIÓN ---
print("\n--- 🔄 Fusionando Datos... ---")
df_total = pd.concat(dfs, ignore_index=True)

# Limpieza
if 'GAME_DATE' in df_total.columns:
    df_total['GAME_DATE'] = pd.to_datetime(df_total['GAME_DATE'])
    df_total.sort_values('GAME_DATE', inplace=True)

df_total.to_csv(ARCHIVO_SALIDA, index=False)
print(f"✅ Base de datos NBA guardada: {len(df_total)} registros.")
print("Listo para crear_ia_nba.py")