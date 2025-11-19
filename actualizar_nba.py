import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import teams

# --- CONFIGURACIÓN ---
ARCHIVO_SALIDA = "nba_games.csv"
TEMPORADAS = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25']

print("==========================================================")
print("   🏀 ACTUALIZADOR NBA (OFFICIAL API) 🏀")
print("==========================================================")

dfs = []

for temp in TEMPORADAS:
    print(f"⬇️  Descargando temporada {temp}...", end=" ")
    try:
        # Descargar Regular Season
        log_reg = leaguegamelog.LeagueGameLog(season=temp, season_type_all_star='Regular Season').get_data_frames()[0]
        log_reg['SeasonType'] = 'Regular'
        
        # Descargar Playoffs (Importante para la "memoria" de equipos buenos)
        log_play = leaguegamelog.LeagueGameLog(season=temp, season_type_all_star='Playoffs').get_data_frames()[0]
        log_play['SeasonType'] = 'Playoffs'
        
        # Unir
        df_temp = pd.concat([log_reg, log_play])
        dfs.append(df_temp)
        print(f"✅ {len(df_temp)} partidos.")
        
        # Respetar límites de la API (evitar ban)
        time.sleep(1) 
        
    except Exception as e:
        print(f"❌ Error: {e}")

if not dfs:
    print("❌ No se han podido descargar datos.")
    exit()

print("\n--- 🔄 Procesando Datos... ---")
df_total = pd.concat(dfs, ignore_index=True)

# Limpieza Básica
df_total['GAME_DATE'] = pd.to_datetime(df_total['GAME_DATE'])
df_total.sort_values('GAME_DATE', inplace=True)

# Guardar
df_total.to_csv(ARCHIVO_SALIDA, index=False)
print(f"✅ Base de datos NBA guardada: {len(df_total)} registros.")
print("Ahora ejecuta: python crear_ia_nba.py")