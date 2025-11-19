import os
import pandas as pd
import requests
import io
import sys
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

ARCHIVO_FINAL = "atp_tennis.csv"
YEARS_HISTORIA = range(2015, 2026) # Reducimos rango para ir más rápido, pero con más calidad

# URL Directa a TML
URL_TML = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"

print("--- ACTUALIZADOR DE DATOS ---")

dfs = []

def limpiar_nombre(nombre):
    if pd.isna(nombre): return "Unknown"
    nombre = str(nombre).strip()
    # Conversión simple: "Novak Djokovic" -> "Djokovic N."
    parts = nombre.split()
    if len(parts) >= 2:
        return f"{parts[-1]} {parts[0][0]}."
    return nombre

# 1. Descargar Histórico
for year in YEARS_HISTORIA:
    print(f"⬇️ Descargando {year}...", end="\r")
    try:
        r = requests.get(URL_TML.format(year=year))
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            # Mapeo crítico de columnas
            mapa = {
                'tourney_date': 'Date', 'surface': 'Surface',
                'winner_name': 'Player_1', 'loser_name': 'Player_2',
                'winner_rank': 'Rank_1', 'loser_rank': 'Rank_2',
                'score': 'Score', 'best_of': 'Best of',
                'w_ace': 'P1_Ace', 'l_ace': 'P2_Ace',
                'w_svpt': 'P1_SvPt', 'l_svpt': 'P2_SvPt',
                'w_1stIn': 'P1_1stIn', 'l_1stIn': 'P2_1stIn',
                'w_1stWon': 'P1_1stWon', 'l_1stWon': 'P2_1stWon',
                'w_bpSaved': 'P1_BpSaved', 'l_bpSaved': 'P2_BpSaved',
                'w_bpFaced': 'P1_BpFaced', 'l_bpFaced': 'P2_BpFaced'
            }
            # Renombrar solo las que existen
            cols_ok = {k:v for k,v in mapa.items() if k in df.columns}
            df.rename(columns=cols_ok, inplace=True)
            
            # Formatear nombres
            df['Player_1'] = df['Player_1'].apply(limpiar_nombre)
            df['Player_2'] = df['Player_2'].apply(limpiar_nombre)
            
            dfs.append(df)
    except Exception as e:
        print(f"Error {year}: {e}")

if not dfs:
    print("❌ No se pudieron descargar datos.")
    sys.exit()

df_total = pd.concat(dfs, ignore_index=True)
df_total['Date'] = pd.to_datetime(df_total['Date'], format='%Y%m%d', errors='coerce')
df_total = df_total.sort_values(by='Date')

# Guardar RAW
df_total.to_csv(ARCHIVO_FINAL, index=False)
print(f"\n✅ Base actualizada con {len(df_total)} partidos.")

print("\n--- EJECUTANDO RE-ENTRENAMIENTO ---")
if os.system("python crear_ia.py") == 0:
    os.system("python entrenar_ia.py")
else:
    print("❌ Error en el procesamiento.")