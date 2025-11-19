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
YEARS_HISTORIA = range(2015, 2026) 
DIAS_LIVE = 45 # Miramos 45 días atrás (suficiente para cubrir huecos)

# URLs
URL_TML = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"
URL_JEFF_FUTURES = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_futures_{year}.csv"

print("==========================================================")
print("   🛡️ ACTUALIZADOR TOTAL (MULTI-TABLA + RAW NAMES) 🛡️")
print("==========================================================")

dfs = []

# --- FUNCIÓN LIMPIEZA MÍNIMA ---
# No cambiamos el nombre. Solo quitamos espacios extra.
def limpiar_nombre(nombre):
    if pd.isna(nombre) or not isinstance(nombre, str): return "Unknown"
    # Quitamos espacios dobles y extremos
    return " ".join(nombre.split())

# --- 1. DESCARGAR HISTORIA ATP/CHALLENGER (TML) ---
print("\n--- 1. Descargando Historia (TML) ---")
for year in YEARS_HISTORIA:
    try:
        r = requests.get(URL_TML.format(year=year))
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            # Mapeo
            cols_map = {'tourney_date': 'Date', 'surface': 'Surface', 'winner_name': 'Player_1', 'loser_name': 'Player_2', 'winner_rank': 'Rank_1', 'loser_rank': 'Rank_2', 'score': 'Score', 'best_of': 'Best of'}
            df.rename(columns=cols_map, inplace=True)
            
            if 'Player_1' in df.columns:
                df = df[[c for c in cols_map.values() if c in df.columns]]
                df['Origen'] = 'TML'
                dfs.append(df)
                print(f"   ✅ {year}: {len(df)} partidos")
    except: pass

# --- 2. DESCARGAR HISTORIA FUTURES (JEFF SACKMANN) ---
print("\n--- 2. Descargando Historia Futures ---")
for year in YEARS_HISTORIA:
    try:
        r = requests.get(URL_JEFF_FUTURES.format(year=year))
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            cols_map = {'tourney_date': 'Date', 'surface': 'Surface', 'winner_name': 'Player_1', 'loser_name': 'Player_2', 'winner_rank': 'Rank_1', 'loser_rank': 'Rank_2', 'score': 'Score', 'best_of': 'Best of'}
            df.rename(columns=cols_map, inplace=True)
            
            if 'Player_1' in df.columns:
                df = df[[c for c in cols_map.values() if c in df.columns]]
                df['Origen'] = 'Futures_Hist'
                dfs.append(df)
                print(f"   ✅ Futures {year}: {len(df)} partidos")
    except: pass

# --- 3. SCRAPING LIVE (CORREGIDO: LEE TODAS LAS TABLAS) ---
print(f"\n--- 3. Escaneando TennisExplorer ({DIAS_LIVE} días) ---")
live_matches = []
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
session = requests.Session()
session.headers.update(headers)

total_rescatados = 0

for i in range(DIAS_LIVE):
    fecha = datetime.now() - timedelta(days=i)
    fecha_str = fecha.strftime('%Y-%m-%d')
    
    # URL para ver TODOS los torneos del día
    url = f"https://www.tennisexplorer.com/results/?type=all&year={fecha.year}&month={fecha.month}&day={fecha.day}"
    
    if i % 5 == 0: print(f"   🕷️ Escaneando {fecha_str}...", end="\r")
    
    try:
        r = session.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        
        # ¡CORRECCIÓN CLAVE! Buscamos todas las tablas, no solo la primera
        tablas = soup.find_all('table', {'class': 'result'})
        
        for tabla in tablas:
            filas = tabla.find_all('tr')
            for fila in filas:
                # Las filas de partidos tienen clase 'one' o 'two'
                if 'one' in fila.get('class', []) or 'two' in fila.get('class', []):
                    cols = fila.find_all('td')
                    if len(cols) >= 5:
                        p1 = cols[0].text.strip()
                        p2 = cols[1].text.strip()
                        score = cols[4].text.strip()
                        
                        # Filtro de calidad básico
                        if p1 and p2 and score and "..." not in p1:
                             live_matches.append({
                                 'Date': fecha,
                                 'Surface': 'Hard', # Default
                                 'Player_1': limpiar_nombre(p1),
                                 'Player_2': limpiar_nombre(p2),
                                 'Rank_1': 500, # Default Futures
                                 'Rank_2': 500,
                                 'Score': score,
                                 'Best of': 3,
                                 'Origen': 'Live_Scraper'
                             })
                             total_rescatados += 1
    except: pass

if live_matches:
    df_live = pd.DataFrame(live_matches)
    dfs.append(df_live)
    print(f"\n   ✅ Se han rescatado {len(df_live)} partidos recientes en vivo.")
else:
    print("\n   ⚠️ No se encontraron partidos en vivo (Revisa tu conexión).")


# --- 4. FUSIÓN Y GUARDADO ---
if not dfs:
    print("❌ Error: Sin datos.")
    sys.exit()

print("\n--- 🔄 Fusionando... ---")
df_total = pd.concat(dfs, ignore_index=True)

if 'Winner' not in df_total.columns:
    df_total['Winner'] = df_total['Player_1']

# Limpieza de Nombres (Solo espacios, mantenemos el nombre completo)
print("🧹 Limpiando espacios en nombres...")
df_total['Player_1'] = df_total['Player_1'].apply(limpiar_nombre)
df_total['Player_2'] = df_total['Player_2'].apply(limpiar_nombre)
df_total['Winner'] = df_total['Player_1']

# Eliminar duplicados (Prioridad: TML > Futures > Scraper)
# Ordenamos por Origen para que al borrar duplicados se quede el de mejor calidad
prioridad = {'TML': 0, 'Futures_Hist': 1, 'Live_Scraper': 2}
df_total['prioridad'] = df_total['Origen'].map(prioridad)
df_total.sort_values(by=['Date', 'prioridad'], inplace=True)
df_total.drop_duplicates(subset=['Date', 'Player_1', 'Player_2'], keep='first', inplace=True)
df_total.drop(columns=['prioridad', 'Origen'], inplace=True)

# Fecha y Guardado
df_total['Date'] = pd.to_datetime(df_total['Date'], format='%Y%m%d', errors='coerce')
df_total.to_csv(ARCHIVO_FINAL, index=False)
print(f"✅ Base de datos FINAL guardada: {len(df_total)} partidos.")

# --- 5. RE-ENTRENAMIENTO ---
print("\n--- 🧠 Entrenando IA... ---")
if os.system("python crear_ia.py") != 0: sys.exit()
if os.system("python entrenar_ia.py") != 0: sys.exit()

print("\n🎉 ¡SISTEMA ACTUALIZADO! Ahora busca en TODAS las tablas de resultados.")