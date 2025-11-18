import os
import pandas as pd
import requests
import io
import sys
import time

# --- CONFIGURACIÓN ---
ARCHIVO_FINAL = "atp_tennis.csv"
YEARS = range(2015, 2026) 

# Fuentes
URL_TML = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"
URL_JEFF_FUTURES = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_futures_{year}.csv"

# Configuración para GitHub Actions (La Nube)
KAGGLE_SECRET = os.environ.get("KAGGLE_JSON") # (Ya no se usa pero lo dejamos por compatibilidad)

print("==========================================================")
print("   🧬 ACTUALIZADOR HÍBRIDO (TML + FUTURES HISTÓRICO) 🧬")
print("==========================================================")

dfs = []

# Función para formatear nombres
def formatear_nombre(nombre):
    try:
        if pd.isna(nombre) or not isinstance(nombre, str): return nombre
        nombre = nombre.strip()
        partes = nombre.split()
        if len(partes) < 2: return nombre
        
        # Estrategia general: "Nombre Apellido" -> "Apellido N."
        nombre_pila = partes[0]
        apellido = " ".join(partes[1:])
        return f"{apellido} {nombre_pila[0]}."
    except:
        return nombre

# --- 1. DESCARGAR TML (ATP + CHALLENGERS ACTUALIZADOS) ---
print("\n--- 1. Descargando Circuito Principal (TML) ---")
for year in YEARS:
    try:
        r = requests.get(URL_TML.format(year=year))
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            # Estandarizar columnas de TML a nuestro formato
            cols_map = {
                'tourney_date': 'Date', 'surface': 'Surface',
                'winner_name': 'Player_1', 'loser_name': 'Player_2',
                'winner_rank': 'Rank_1', 'loser_rank': 'Rank_2',
                'score': 'Score', 'best_of': 'Best of'
            }
            df.rename(columns=cols_map, inplace=True)
            df = df[[c for c in cols_map.values() if c in df.columns]] # Solo columnas útiles
            df['Origen'] = 'TML'
            dfs.append(df)
            print(f"   ✅ {year}: {len(df)} partidos")
    except Exception as e:
        pass # Si falla un año (ej: 2025 aun no completo), seguimos

# --- 2. DESCARGAR JEFF SACKMANN (SOLO FUTURES) ---
print("\n--- 2. Descargando Futures/ITF (Jeff Sackmann) ---")
for year in YEARS:
    try:
        r = requests.get(URL_JEFF_FUTURES.format(year=year))
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            # Estandarizar columnas de Jeff a nuestro formato
            cols_map = {
                'tourney_date': 'Date', 'surface': 'Surface',
                'winner_name': 'Player_1', 'loser_name': 'Player_2',
                'winner_rank': 'Rank_1', 'loser_rank': 'Rank_2',
                'score': 'Score', 'best_of': 'Best of'
            }
            df.rename(columns=cols_map, inplace=True)
            df = df[[c for c in cols_map.values() if c in df.columns]]
            df['Origen'] = 'Futures'
            dfs.append(df)
            print(f"   ✅ Futures {year}: {len(df)} partidos (Aquí está Gima)")
    except Exception as e:
        pass # Futures 2025 puede no estar aún, no importa

if not dfs:
    print("❌ Error: No se descargó nada.")
    sys.exit()

# --- FUSIÓN Y LIMPIEZA ---
print("\n--- 🔄 Fusionando y Procesando... ---")
df_total = pd.concat(dfs, ignore_index=True)

# Crear columna Winner
df_total['Winner'] = df_total['Player_1']

# Formatear Fechas
df_total['Date'] = pd.to_datetime(df_total['Date'], format='%Y%m%d', errors='coerce')

# Formatear Nombres (CRUCIAL para que coincidan)
print("🧹 Normalizando nombres (esto tarda un poco)...")
df_total['Player_1'] = df_total['Player_1'].apply(formatear_nombre)
df_total['Player_2'] = df_total['Player_2'].apply(formatear_nombre)
df_total['Winner'] = df_total['Player_1']

# Guardar
df_total.to_csv(ARCHIVO_FINAL, index=False)
print(f"✅ Base de datos FINAL guardada: {len(df_total)} partidos.")

# --- RE-ENTRENAMIENTO ---
print("\n--- 🧠 Re-Entrenando IA... ---")

if os.system("python crear_ia.py") != 0: 
    print("❌ Error en crear_ia.py")
    sys.exit()

if os.system("python entrenar_ia.py") != 0:
    print("❌ Error en entrenar_ia.py")
    sys.exit()

print("\n🎉 ¡SISTEMA HÍBRIDO LISTO! Subiendo resultados...")