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
DIAS_LIVE = 30  # Miramos 30 días atrás en TennisExplorer para pillar Futures recientes

# URLs
URL_TML = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"
URL_JEFF_FUTURES = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_futures_{year}.csv"

print("==========================================================")
print("   🧬 ACTUALIZADOR TOTAL (HISTORIA + LIVE FUTURES) 🧬")
print("==========================================================")

dfs = []

# --- FUNCIÓN DE NORMALIZACIÓN DE NOMBRES ---
def formatear_nombre(nombre):
    try:
        if pd.isna(nombre) or not isinstance(nombre, str): return nombre
        nombre = nombre.strip()
        partes = nombre.split()
        if len(partes) < 2: return nombre
        # "Daniel Vallejo" -> "Vallejo D."
        nombre_pila = partes[0]
        apellido = " ".join(partes[1:])
        return f"{apellido} {nombre_pila[0]}."
    except:
        return nombre

# --- 1. DESCARGAR HISTORIA ATP/CHALLENGER (TML) ---
print("\n--- 1. Descargando Historia ATP/Challenger (TML) ---")
for year in YEARS_HISTORIA:
    try:
        r = requests.get(URL_TML.format(year=year))
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            cols_map = {'tourney_date': 'Date', 'surface': 'Surface', 'winner_name': 'Player_1', 'loser_name': 'Player_2', 'winner_rank': 'Rank_1', 'loser_rank': 'Rank_2', 'score': 'Score', 'best_of': 'Best of'}
            df.rename(columns=cols_map, inplace=True)
            df = df[[c for c in cols_map.values() if c in df.columns]]
            df['Origen'] = 'TML'
            dfs.append(df)
            print(f"   ✅ {year}: {len(df)} partidos")
    except: pass

# --- 2. DESCARGAR HISTORIA FUTURES (JEFF SACKMANN) ---
print("\n--- 2. Descargando Historia Futures (Jeff Sackmann) ---")
for year in YEARS_HISTORIA:
    if year == 2025: continue # Jeff no suele tener 2025 Futures
    try:
        r = requests.get(URL_JEFF_FUTURES.format(year=year))
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            cols_map = {'tourney_date': 'Date', 'surface': 'Surface', 'winner_name': 'Player_1', 'loser_name': 'Player_2', 'winner_rank': 'Rank_1', 'loser_rank': 'Rank_2', 'score': 'Score', 'best_of': 'Best of'}
            df.rename(columns=cols_map, inplace=True)
            df = df[[c for c in cols_map.values() if c in df.columns]]
            df['Origen'] = 'Futures_Hist'
            dfs.append(df)
            print(f"   ✅ Futures {year}: {len(df)} partidos")
    except: pass

# --- 3. SCRAPING LIVE (TENNIS EXPLORER) ---
print(f"\n--- 3. Buscando Futures RECIENTES ({DIAS_LIVE} días) ---")
live_matches = []
headers = {'User-Agent': 'Mozilla/5.0'}

for i in range(DIAS_LIVE):
    fecha = datetime.now() - timedelta(days=i)
    fecha_str = fecha.strftime('%Y-%m-%d')
    # URL específica para todos los torneos (incluye ITF)
    url = f"https://www.tennisexplorer.com/results/?type=all&year={fecha.year}&month={fecha.month}&day={fecha.day}"
    
    print(f"   🕷️ Scraping {fecha_str}...", end="\r")
    try:
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, 'html.parser') # Usamos html.parser por si lxml no está
        tabla = soup.find('table', {'class': 'result'})
        
        if tabla:
            filas = tabla.find_all('tr')
            for fila in filas:
                if 'one' in fila.get('class', []) or 'two' in fila.get('class', []):
                    cols = fila.find_all('td')
                    if len(cols) >= 5:
                        # Solo nos interesa si parece un partido válido con score
                        p1 = cols[0].text.strip()
                        p2 = cols[1].text.strip()
                        score = cols[4].text.strip()
                        if p1 and p2 and score:
                             live_matches.append({
                                 'Date': fecha,
                                 'Surface': 'Hard', # Default, difícil de sacar aquí
                                 'Player_1': formatear_nombre(p1),
                                 'Player_2': formatear_nombre(p2),
                                 'Rank_1': 500, # Default Futures
                                 'Rank_2': 500,
                                 'Score': score,
                                 'Best of': 3,
                                 'Origen': 'Live_Scraper'
                             })
    except: pass

if live_matches:
    df_live = pd.DataFrame(live_matches)
    dfs.append(df_live)
    print(f"\n   ✅ Se han rescatado {len(df_live)} partidos recientes (Futures 2025).")
else:
    print("\n   ⚠️ No se pudieron sacar partidos en vivo (o error de conexión).")


# --- 4. FUSIÓN Y GUARDADO ---
if not dfs:
    print("❌ Error total: Sin datos.")
    sys.exit()

print("\n--- 🔄 Fusionando todo... ---")
df_total = pd.concat(dfs, ignore_index=True)

# Eliminar duplicados (Si TML y el Scraper tienen el mismo partido, nos quedamos con TML)
# Usamos Fecha y Jugadores como clave
df_total['Winner'] = df_total['Player_1'] # Asunción inicial para scraper
df_total.drop_duplicates(subset=['Date', 'Player_1', 'Player_2'], keep='first', inplace=True)

# Formatear nombres (Una última pasada por seguridad)
print("🧹 Limpiando nombres...")
df_total['Player_1'] = df_total['Player_1'].apply(formatear_nombre)
df_total['Player_2'] = df_total['Player_2'].apply(formatear_nombre)
df_total['Winner'] = df_total['Player_1'] # En nuestros datos RAW, P1 siempre es Winner

# Fechas
df_total['Date'] = pd.to_datetime(df_total['Date'], format='%Y%m%d', errors='coerce')

df_total.to_csv(ARCHIVO_FINAL, index=False)
print(f"✅ Base de datos FINAL: {len(df_total)} partidos.")

# --- 5. RE-ENTRENAMIENTO ---
print("\n--- 🧠 Entrenando IA... ---")
if os.system("python crear_ia.py") != 0: sys.exit()
if os.system("python entrenar_ia.py") != 0: sys.exit()

print("\n🎉 ¡SISTEMA COMPLETO LISTO! (ATP + CHALL + FUTURES 2025)")