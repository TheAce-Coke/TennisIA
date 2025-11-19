import os
import pandas as pd
import requests
import io
import sys
from datetime import datetime

# --- CONFIGURACIÓN ---
ARCHIVO_FINAL = "atp_tennis.csv"
# Desde 2015 es suficiente para tener "memoria reciente" de calidad
YEARS_HISTORIA = range(2015, 2026) 

URL_TML = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"

print("==========================================================")
print("   🛡️ ACTUALIZADOR QUANT V3 (FULL STATS) 🛡️")
print("==========================================================")

dfs = []

def limpiar_nombre(nombre):
    if pd.isna(nombre): return "Unknown"
    nombre = str(nombre).strip()
    # Formato: "Novak Djokovic" -> "Djokovic N."
    parts = nombre.split()
    if len(parts) >= 2:
        return f"{parts[-1]} {parts[0][0]}."
    return nombre

# --- DESCARGA Y PROCESADO ---
for year in YEARS_HISTORIA:
    print(f"⬇️ Descargando {year}...", end="\r")
    try:
        r = requests.get(URL_TML.format(year=year))
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            
            # --- MAPEO DE COLUMNAS EXTENDIDO (CRUCIAL PARA MONTE CARLO) ---
            mapa = {
                'tourney_date': 'Date', 
                'surface': 'Surface',
                'winner_name': 'Player_1', 
                'loser_name': 'Player_2',
                'winner_rank': 'Rank_1', 
                'loser_rank': 'Rank_2',
                'score': 'Score', 
                'best_of': 'Best of',
                
                # Stats Saque J1 (Winner)
                'w_ace': 'P1_Ace', 
                'w_df': 'P1_DF', 
                'w_svpt': 'P1_SvPt', 
                'w_1stIn': 'P1_1stIn', 
                'w_1stWon': 'P1_1stWon', 
                'w_2ndWon': 'P1_2ndWon', # <--- NUEVO Y CRITICO
                'w_svgms': 'P1_SvGms',
                'w_bpSaved': 'P1_BpSaved', 
                'w_bpFaced': 'P1_BpFaced',
                
                # Stats Saque J2 (Loser)
                'l_ace': 'P2_Ace', 
                'l_df': 'P2_DF', 
                'l_svpt': 'P2_SvPt', 
                'l_1stIn': 'P2_1stIn', 
                'l_1stWon': 'P2_1stWon', 
                'l_2ndWon': 'P2_2ndWon', # <--- NUEVO Y CRITICO
                'l_svgms': 'P2_SvGms',
                'l_bpSaved': 'P2_BpSaved', 
                'l_bpFaced': 'P2_BpFaced'
            }
            
            # Renombrar solo lo que exista
            cols_ok = {k:v for k,v in mapa.items() if k in df.columns}
            df.rename(columns=cols_ok, inplace=True)
            
            # Formatear Nombres
            df['Player_1'] = df['Player_1'].apply(limpiar_nombre)
            df['Player_2'] = df['Player_2'].apply(limpiar_nombre)
            
            # Filtrar columnas irrelevantes para ahorrar espacio, manteniendo las del mapa
            cols_finales = list(cols_ok.values())
            # Aseguramos que existan en el df ya renombrado
            cols_finales = [c for c in cols_finales if c in df.columns]
            
            dfs.append(df[cols_finales])
            
    except Exception as e:
        print(f"   ❌ Error {year}: {e}")

if not dfs:
    print("\n❌ Error Crítico: No se descargaron datos.")
    sys.exit()

# --- FUSIÓN ---
print("\n--- 🔄 Fusionando Histórico... ---")
df_total = pd.concat(dfs, ignore_index=True)

# Limpieza de Fechas
df_total['Date'] = pd.to_datetime(df_total['Date'], format='%Y%m%d', errors='coerce')
df_total.sort_values(by='Date', inplace=True)

# Guardar RAW
df_total.to_csv(ARCHIVO_FINAL, index=False)
print(f"✅ Base de datos MAESTRA guardada: {len(df_total)} partidos.")

# --- EJECUCIÓN EN CADENA ---
print("\n--- 🧠 Iniciando Re-Entrenamiento del Sistema Quant ---")

# 1. Ingeniería de Datos (Calcula EWMA, Saque Real, Resto Real)
print("> Ejecutando crear_ia.py...")
if os.system("python crear_ia.py") != 0:
    print("❌ Error en crear_ia.py"); sys.exit()

# 2. Entrenamiento y Calibración (Genera modelos calibrados)
print("> Ejecutando entrenar_ia.py...")
if os.system("python entrenar_ia.py") != 0:
    print("❌ Error en entrenar_ia.py"); sys.exit()

print("\n🎉 ¡SISTEMA ACTUALIZADO COMPLETAMENTE!")
print("Ahora puedes abrir la app con: streamlit run app.py")