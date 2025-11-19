import os
import pandas as pd
import requests
import io
import sys
import time
import re  # Importante para limpiar paréntesis
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# --- CONFIGURACIÓN ---
ARCHIVO_FINAL = "atp_tennis.csv"
YEARS_HISTORIA = range(2010, 2026) 
DIAS_LIVE = 45 

URL_TML = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"
URL_JEFF_FUTURES = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_futures_{year}.csv"

print("==========================================================")
print("   🧹 ACTUALIZADOR CON LIMPIEZA DE NOMBRES AVANZADA 🧹")
print("==========================================================")

dfs = []

# --- NUEVA FUNCIÓN DE LIMPIEZA MAESTRA ---
def limpiar_nombre_pro(nombre):
    if pd.isna(nombre) or not isinstance(nombre, str): return "Unknown"
    
    # 1. Si tiene barra '/' es dobles -> Lo marcaremos para borrar luego
    if '/' in nombre: return "DOUBLES_MATCH"
    
    # 2. Quitar paréntesis y lo que haya dentro: "Passaro F. (4)" -> "Passaro F."
    nombre = re.sub(r'\s*\([^)]*\)', '', nombre)
    
    # 3. Quitar espacios extra
    nombre = nombre.strip()
    
    # 4. Formato "Nombre Apellido" -> "Apellido N." (si no tiene punto)
    # Si ya tiene punto (ej: "Passaro F."), asumimos que está bien.
    if "." not in nombre and " " in nombre:
        partes = nombre.split()
        if len(partes) >= 2:
            # "Adolfo Daniel Vallejo" -> "Vallejo A." (o Vallejo D., es difícil saber cual usa TML)
            # Para ser seguros y coincidir con TML que suele usar "Apellido N.":
            nombre_pila = partes[0]
            apellido = " ".join(partes[1:])
            return f"{apellido} {nombre_pila[0]}."
            
    return nombre

# --- 1. TML ---
print("\n--- 1. Descargando TML ---")
for year in YEARS_HISTORIA:
    try:
        r = requests.get(URL_TML.format(year=year))
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            cols_map = {
                'tourney_date': 'Date', 'surface': 'Surface', 
                'winner_name': 'Player_1', 'loser_name': 'Player_2', 
                'winner_rank': 'Rank_1', 'loser_rank': 'Rank_2', 
                'score': 'Score', 'best_of': 'Best of',
                'w_ace': 'P1_Ace', 'l_ace': 'P2_Ace',
                'w_df': 'P1_DF', 'l_df': 'P2_DF',
                'w_svpt': 'P1_SvPt', 'l_svpt': 'P2_SvPt', 
                'w_1stIn': 'P1_1stIn', 'l_1stIn': 'P2_1stIn', 
                'w_1stWon': 'P1_1stWon', 'l_1stWon': 'P2_1stWon', 
                'w_bpSaved': 'P1_BpSaved', 'l_bpSaved': 'P2_BpSaved'
            }
            df.rename(columns=cols_map, inplace=True)
            cols_keep = [c for c in cols_map.values() if c in df.columns]
            df = df[cols_keep]
            df['Origen'] = 'TML'
            dfs.append(df)
            print(f"   ✅ {year}: {len(df)}")
    except: pass

# --- 2. FUTURES ---
print("\n--- 2. Descargando Futures ---")
for year in YEARS_HISTORIA:
    try:
        r = requests.get(URL_JEFF_FUTURES.format(year=year))
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            cols_map = {'tourney_date': 'Date', 'surface': 'Surface', 'winner_name': 'Player_1', 'loser_name': 'Player_2', 'winner_rank': 'Rank_1', 'loser_rank': 'Rank_2', 'score': 'Score', 'best_of': 'Best of'}
            df.rename(columns=cols_map, inplace=True)
            if 'Player_1' in df.columns:
                df = df[[c for c in cols_map.values() if c in df.columns]]
                df['Origen'] = 'Futures'
                dfs.append(df)
                print(f"   ✅ {year}: {len(df)}")
    except: pass

# --- 3. SCRAPER ---
print(f"\n--- 3. Escaneando Live ({DIAS_LIVE} días) ---")
live_matches = []
headers = {'User-Agent': 'Mozilla/5.0'}
session = requests.Session()
session.headers.update(headers)

for i in range(DIAS_LIVE):
    fecha = datetime.now() - timedelta(days=i)
    fecha_str = fecha.strftime('%Y%m%d')
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
                        
                        if p1 and p2 and score and "..." not in p1 and "/" not in p1:
                             live_matches.append({
                                 'Date': fecha_str, 'Surface': 'Hard',
                                 'Player_1': p1, 'Player_2': p2, # Limpiamos luego
                                 'Rank_1': 500, 'Rank_2': 500,
                                 'Score': score, 'Best of': 3, 'Origen': 'Live'
                             })
    except: pass

if live_matches:
    df_live = pd.DataFrame(live_matches)
    dfs.append(df_live)
    print(f"\n   ✅ Live: {len(df_live)}")

# --- 4. FUSIÓN Y LIMPIEZA ---
if not dfs: sys.exit("❌ Sin datos.")

print("\n--- 🔄 Fusionando y Limpiando Nombres... ---")
df_total = pd.concat(dfs, ignore_index=True)

if 'Winner' not in df_total.columns: df_total['Winner'] = df_total['Player_1']

# APLICAMOS LA LIMPIEZA PRO
print("   -> Eliminando paréntesis (4), (Q) y Dobles...")
df_total['Player_1'] = df_total['Player_1'].apply(limpiar_nombre_pro)
df_total['Player_2'] = df_total['Player_2'].apply(limpiar_nombre_pro)
df_total['Winner'] = df_total['Player_1']

# FILTRO ANTI-DOBLES
antes = len(df_total)
df_total = df_total[df_total['Player_1'] != "DOUBLES_MATCH"]
df_total = df_total[df_total['Player_2'] != "DOUBLES_MATCH"]
print(f"   -> Eliminados {antes - len(df_total)} partidos de dobles.")

# Fechas y Duplicados
df_total['Date'] = pd.to_datetime(df_total['Date'], format='%Y%m%d', errors='coerce')
prioridad = {'TML': 0, 'Futures': 1, 'Live': 2}
df_total['prioridad'] = df_total['Origen'].map(prioridad)
df_total.sort_values(by=['Date', 'prioridad'], inplace=True)
df_total.drop_duplicates(subset=['Date', 'Player_1', 'Player_2'], keep='first', inplace=True)
df_total.drop(columns=['prioridad', 'Origen'], inplace=True)

df_total.to_csv(ARCHIVO_FINAL, index=False)
print(f"✅ Base de datos FINAL: {len(df_total)} partidos.")

# --- 5. RE-ENTRENAMIENTO ---
print("\n--- 🧠 Re-Entrenando... ---")
if os.system("python crear_ia.py") != 0: sys.exit()
if os.system("python entrenar_ia.py") != 0: sys.exit()
print("\n🎉 ¡LISTO! Nombres unificados.")