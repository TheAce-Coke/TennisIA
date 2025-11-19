import os
import pandas as pd
import requests
import io
import sys

# --- CONFIGURACIÓN ---
ARCHIVO_FINAL = "atp_tennis.csv"
# Descargamos desde 2015 para tener una base sólida reciente
YEARS_HISTORIA = range(2015, 2026) 

# Fuente de datos: Tennis My Life (GitHub)
URL_TML = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"

print("==========================================================")
print("   🛡️ ACTUALIZADOR TENIS QUANT (FULL STATS) 🛡️")
print("==========================================================")

dfs = []

def limpiar_nombre(nombre):
    if pd.isna(nombre): return "Unknown"
    nombre = str(nombre).strip()
    # Convertir "Novak Djokovic" -> "Djokovic N."
    parts = nombre.split()
    if len(parts) >= 2:
        return f"{parts[-1]} {parts[0][0]}."
    return nombre

# --- 1. DESCARGA ---
for year in YEARS_HISTORIA:
    print(f"⬇️ Descargando temporada {year}...", end="\r")
    try:
        r = requests.get(URL_TML.format(year=year))
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            
            # --- MAPEO CRÍTICO PARA EL MOTOR MONTE CARLO ---
            # Necesitamos estadísticas detalladas de saque y resto
            mapa = {
                'tourney_date': 'Date', 
                'surface': 'Surface',
                'winner_name': 'Player_1', 
                'loser_name': 'Player_2',
                'winner_rank': 'Rank_1', 
                'loser_rank': 'Rank_2',
                'score': 'Score', 
                'best_of': 'Best of',
                
                # Stats J1 (Ganador)
                'w_ace': 'P1_Ace', 
                'w_df': 'P1_DF', 
                'w_svpt': 'P1_SvPt', 
                'w_1stIn': 'P1_1stIn', 
                'w_1stWon': 'P1_1stWon', 
                'w_2ndWon': 'P1_2ndWon', # <--- IMPORTANTE PARA % SAQUE REAL
                'w_svgms': 'P1_SvGms',   # <--- IMPORTANTE
                'w_bpSaved': 'P1_BpSaved', 
                'w_bpFaced': 'P1_BpFaced',
                
                # Stats J2 (Perdedor)
                'l_ace': 'P2_Ace', 
                'l_df': 'P2_DF', 
                'l_svpt': 'P2_SvPt', 
                'l_1stIn': 'P2_1stIn', 
                'l_1stWon': 'P2_1stWon', 
                'l_2ndWon': 'P2_2ndWon', # <--- IMPORTANTE PARA % SAQUE REAL
                'l_svgms': 'P2_SvGms',   # <--- IMPORTANTE
                'l_bpSaved': 'P2_BpSaved', 
                'l_bpFaced': 'P2_BpFaced'
            }
            
            # Renombrar columnas que existan en el CSV descargado
            cols_ok = {k:v for k,v in mapa.items() if k in df.columns}
            df.rename(columns=cols_ok, inplace=True)
            
            # Formatear nombres de jugadores
            df['Player_1'] = df['Player_1'].apply(limpiar_nombre)
            df['Player_2'] = df['Player_2'].apply(limpiar_nombre)
            
            # Filtrar solo las columnas útiles mapeadas
            final_cols = [c for c in list(mapa.values()) if c in df.columns]
            dfs.append(df[final_cols])
            
    except Exception as e:
        print(f"   ❌ Error descargando {year}: {e}")

# --- 2. FUSIÓN Y GUARDADO ---
if not dfs:
    print("\n❌ Error Crítico: No se han podido descargar datos.")
    sys.exit()

print("\n--- 🔄 Fusionando y Guardando CSV Maestro... ---")
df_total = pd.concat(dfs, ignore_index=True)

# Asegurar formato fecha
df_total['Date'] = pd.to_datetime(df_total['Date'], format='%Y%m%d', errors='coerce')
df_total.sort_values(by='Date', inplace=True)

df_total.to_csv(ARCHIVO_FINAL, index=False)
print(f"✅ Base de datos actualizada: {len(df_total)} partidos.")

# --- 3. AUTOMATIZACIÓN DEL RE-ENTRENAMIENTO ---
print("\n--- 🧠 Ejecutando Procesamiento IA... ---")

# Ejecutar crear_ia.py para calcular Elo, EWMA y Stats Reales
if os.path.exists("crear_ia.py"):
    print("> Ejecutando crear_ia.py...")
    if os.system("python crear_ia.py") != 0:
        print("❌ Fallo en crear_ia.py")
        sys.exit()
else:
    print("⚠️ No encuentro crear_ia.py")

# Ejecutar entrenar_ia.py para calibrar modelos
if os.path.exists("entrenar_ia.py"):
    print("> Ejecutando entrenar_ia.py...")
    if os.system("python entrenar_ia.py") != 0:
        print("❌ Fallo en entrenar_ia.py")
        sys.exit()
else:
    print("⚠️ No encuentro entrenar_ia.py")

print("\n🎉 ¡SISTEMA DE TENIS ACTUALIZADO Y LISTO!")