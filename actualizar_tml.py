import pandas as pd
import requests
import io
import sys
import os

# --- CONFIGURACIÓN ---
ARCHIVO_FINAL = "atp_tennis.csv"
# Descargamos desde 2010 para tener una base sólida de veteranos y retirados recientes
YEARS = range(2010, 2026) 

# URL Base de TML (Tennis My Life)
URL_BASE = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"

print("==================================================")
print("   🚀 ACTUALIZADOR TML (FUENTE: TENNIS MY LIFE) 🚀")
print("   (Incluye ATP, Challengers y Futures actualizados)")
print("==================================================")

dfs = []

# Función para formatear nombres: "Novak Djokovic" -> "Djokovic N."
def formatear_nombre(nombre):
    try:
        if pd.isna(nombre) or not isinstance(nombre, str): return nombre
        nombre = nombre.strip()
        partes = nombre.split()
        
        if len(partes) < 2: return nombre
        
        # TML a veces pone "Vallejo Adolfo Daniel". 
        # Estrategia: Usar la última palabra como apellido principal y la primera letra del primero.
        # Ojo: En nombres compuestos latinos esto puede fallar levemente, pero es consistente.
        
        # Estrategia estándar: "Nombre Apellido" -> "Apellido N."
        nombre_pila = partes[0]
        apellido = " ".join(partes[1:])
        
        return f"{apellido} {nombre_pila[0]}."
    except:
        return nombre

# --- DESCARGA ---
for year in YEARS:
    print(f"⬇️ Descargando {year}...", end=" ")
    url = URL_BASE.format(year=year)
    
    try:
        r = requests.get(url)
        if r.status_code == 200:
            # TML usa comas estándar
            df = pd.read_csv(io.StringIO(r.text))
            
            # Filtro de seguridad: Aseguramos que tenga las columnas clave
            if 'winner_name' in df.columns and 'loser_name' in df.columns:
                dfs.append(df)
                print(f"✅ {len(df)} partidos.")
            else:
                print(f"⚠️ Formato desconocido.")
        else:
            print(f"❌ No encontrado (HTTP {r.status_code})")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if not dfs:
    print("❌ Error Crítico: No se descargó nada.")
    sys.exit()

# --- FUSIÓN ---
print("\n--- 🔄 Procesando y Estandarizando Datos... ---")
df_total = pd.concat(dfs, ignore_index=True)

# Selección y Renombrado de columnas para tu sistema
# TML tiene muchas columnas ricas (w_ace, minutes), las guardamos por si acaso
# pero renombramos las esenciales para que 'crear_ia.py' no se rompa.

cols_map = {
    'tourney_date': 'Date',
    'surface': 'Surface',
    'winner_name': 'Player_1',  # Asumimos ganador en P1 para el formato
    'loser_name': 'Player_2',
    'winner_rank': 'Rank_1',
    'loser_rank': 'Rank_2',
    'score': 'Score',
    'best_of': 'Best of',
    'tourney_level': 'Tourney_Level' # Importante para saber si es Challenger
}

# Nos aseguramos de que existan antes de renombrar
cols_existentes = {k: v for k, v in cols_map.items() if k in df_total.columns}
df_total.rename(columns=cols_existentes, inplace=True)

# Crear columna Winner explícita (Player_1 siempre es el ganador en el raw data)
df_total['Winner'] = df_total['Player_1']

# Formatear Fechas (TML usa YYYYMMDD o YYYY-MM-DD, pandas lo suele detectar)
df_total['Date'] = pd.to_datetime(df_total['Date'], format='%Y%m%d', errors='coerce')

# --- FORMATEO DE NOMBRES (CRUCIAL PARA EL BUSCADOR) ---
print("🧹 Formateando nombres (esto tarda unos segundos)...")
# Hacemos esto porque TML usa nombres completos y nosotros queremos 'Apellido N.'
df_total['Player_1'] = df_total['Player_1'].apply(formatear_nombre)
df_total['Player_2'] = df_total['Player_2'].apply(formatear_nombre)
df_total['Winner'] = df_total['Player_1'] # Actualizamos Winner con el nombre formateado

# Guardar
df_total.to_csv(ARCHIVO_FINAL, index=False)
print(f"✅ Base de datos guardada: {len(df_total)} partidos.")


# --- RE-ENTRENAMIENTO ---
print("\n--- 🧠 Entrenando IA... ---")

print("> 1. Ejecutando crear_ia.py...")
if os.system("python crear_ia.py") != 0: 
    print("❌ Error en crear_ia.py")
    sys.exit()

print("> 2. Ejecutando entrenar_ia.py...")
if os.system("python entrenar_ia.py") != 0:
    print("❌ Error en entrenar_ia.py")
    sys.exit()

print("\n🎉 ¡SISTEMA ACTUALIZADO Y LISTO! 🎉")
print("Ahora puedes buscar a 'Vallejo D.' en tu web.")