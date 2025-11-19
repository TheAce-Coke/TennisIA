import os
import pandas as pd
import requests
import io
import sys
import time
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# --- CONFIGURACIÓN ---
ARCHIVO_FINAL = "atp_tennis.csv"
YEARS_HISTORIA = range(2010, 2026) 
DIAS_LIVE = 45 

# URLs
URL_TML = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"
URL_JEFF_FUTURES = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_futures_{year}.csv"

print("==========================================================")
print("   💎 ACTUALIZADOR PRO (DATOS RICOS + FUTURES) 💎")
print("==========================================================")

dfs = []

def limpiar_nombre(nombre):
    if pd.isna(nombre) or not isinstance(nombre, str): return "Unknown"
    return " ".join(nombre.split())

# --- 1. TML (Datos Ricos: Aces, Breakpoints, etc.) ---
print("\n--- 1. Descargando Historia Rica (TML) ---")
for year in YEARS_HISTORIA:
    try:
        r = requests.get(URL_TML.format(year=year))
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            
            # MAPEO EXTENDIDO: Guardamos estadísticas técnicas
            cols_map = {
                'tourney_date': 'Date', 'surface': 'Surface', 
                'winner_name': 'Player_1', 'loser_name': 'Player_2', 
                'winner_rank': 'Rank_1', 'loser_rank': 'Rank_2', 
                'score': 'Score', 'best_of': 'Best of',
                # Estadísticas Avanzadas
                'w_ace': 'P1_Ace', 'l_ace': 'P2_Ace',
                'w_df': 'P1_DF', 'l_df': 'P2_DF',
                'w_svpt': 'P1_SvPt', 'l_svpt': 'P2_SvPt', # Puntos totales al saque
                'w_1stIn': 'P1_1stIn', 'l_1stIn': 'P2_1stIn', # Primeros servicios dentro
                'w_1stWon': 'P1_1stWon', 'l_1stWon': 'P2_1stWon', # Puntos ganados con 1er saque
                'w_bpSaved': 'P1_BpSaved', 'l_bpSaved': 'P2_BpSaved',
                'w_bpFaced': 'P1_BpFaced', 'l_bpFaced': 'P2_BpFaced',
                'minutes': 'Minutes'
            }
            
            df.rename(columns=cols_map, inplace=True)
            
            # Filtramos solo columnas que existan en el mapeo + las que ya tienen nombre correcto
            cols_a_guardar = [c for c in cols_map.values() if c in df.columns]
            df = df[cols_a_guardar]
            
            df['Origen'] = 'TML'
            dfs.append(df)
            print(f"   ✅ ATP/Challenger {year}: {len(df)} partidos (Con Stats)")
    except Exception as e: pass

# --- 2. FUTURES (Datos Básicos) ---
print("\n--- 2. Descargando Futures (Jeff Sackmann) ---")
for year in YEARS_HISTORIA:
    try:
        r = requests.get(URL_JEFF_FUTURES.format(year=year))
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            # Jeff tiene columnas básicas, no tiene stats detalladas
            cols_map = {
                'tourney_date': 'Date', 'surface': 'Surface', 
                'winner_name': 'Player_1', 'loser_name': 'Player_2', 
                'winner_rank': 'Rank_1', 'loser_rank': 'Rank_2', 
                'score': 'Score', 'best_of': 'Best of'
            }
            df.rename(columns=cols_map, inplace=True)
            if 'Player_1' in df.columns:
                df = df[[c for c in cols_map.values() if c in df.columns]]
                df['Origen'] = 'Futures'
                dfs.append(df)
                print(f"   ✅ Futures {year}: {len(df)} partidos")
    except: pass

# --- 3. SCRAPER (Datos en Vivo) ---
print(f"\n--- 3. Escaneando TennisExplorer ({DIAS_LIVE} días) ---")
live_matches = []
headers = {'User-Agent': 'Mozilla/5.0'}
session = requests.Session()
session.headers.update(headers)

for i in range(DIAS_LIVE):
    fecha = datetime.now() - timedelta(days=i)
    fecha_str = fecha.strftime('%Y%m%d') # Formato numérico para el CSV
    url = f"https://www.tennisexplorer.com/results/?type=all&year={fecha.year}&month={fecha.month}&day={fecha.day}"
    
    if i % 5 == 0: print(f"   🕷️ {fecha.strftime('%Y-%m-%d')}...", end="\r")
    
    try:
        r = session.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        tablas = soup.find_all('table', {'class': 'result'})
        for tabla in tablas:
            filas = tabla.find_all('tr')
            for fila in filas:
                if 'one' in fila.get('class', []) or 'two' in fila.get('class', []):
                    cols = fila.find_all('td')
                    if len(cols) >= 5:
                        p1 = cols[0].text.strip()
                        p2 = cols[1].text.strip()
                        score = cols[4].text.strip()
                        if p1 and p2 and score and "..." not in p1:
                             live_matches.append({
                                 'Date': fecha_str,
                                 'Surface': 'Hard',
                                 'Player_1': limpiar_nombre(p1),
                                 'Player_2': limpiar_nombre(p2),
                                 'Rank_1': 500, 'Rank_2': 500,
                                 'Score': score, 'Best of': 3,
                                 'Origen': 'Live'
                             })
    except: pass

if live_matches:
    df_live = pd.DataFrame(live_matches)
    dfs.append(df_live)
    print(f"\n   ✅ Rescatados {len(df_live)} partidos live.")

# --- 4. FUSIÓN ---
if not dfs: sys.exit("❌ Error: Sin datos.")

print("\n--- 🔄 Fusionando... ---")
df_total = pd.concat(dfs, ignore_index=True)

if 'Winner' not in df_total.columns: df_total['Winner'] = df_total['Player_1']

# Limpieza
df_total['Player_1'] = df_total['Player_1'].apply(limpiar_nombre)
df_total['Player_2'] = df_total['Player_2'].apply(limpiar_nombre)
df_total['Winner'] = df_total['Player_1']

# Fechas y Orden
print("📅 Unificando fechas...")
df_total['Date'] = pd.to_datetime(df_total['Date'], format='%Y%m%d', errors='coerce')

# Prioridad para eliminar duplicados: TML (0) > Futures (1) > Live (2)
prioridad = {'TML': 0, 'Futures': 1, 'Live': 2}
df_total['prioridad'] = df_total['Origen'].map(prioridad)
df_total.sort_values(by=['Date', 'prioridad'], inplace=True)
df_total.drop_duplicates(subset=['Date', 'Player_1', 'Player_2'], keep='first', inplace=True)
df_total.drop(columns=['prioridad', 'Origen'], inplace=True)

df_total.to_csv(ARCHIVO_FINAL, index=False)
print(f"✅ Base de datos FINAL guardada: {len(df_total)} partidos.")

# --- 5. EJECUCIÓN ---
print("\n--- 🧠 Ejecutando Pipeline de IA ---")
if os.system("python crear_ia.py") != 0: sys.exit()
if os.system("python entrenar_ia.py") != 0: sys.exit()