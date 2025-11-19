import pandas as pd
import joblib
import numpy as np
import os

print("--- Cargando IA Completa (Ganador + Juegos) ---")

# --- CARGA ROBUSTA DE MODELOS ---
archivos = ['modelo_ganador.joblib', 'features_ganador.joblib', 
            'modelo_juegos.joblib', 'features_juegos.joblib', 
            'atp_matches_procesados.csv']

if not all(os.path.exists(f) for f in archivos):
    print("❌ Faltan archivos. Ejecuta 'actualizar_auto.py' primero.")
    exit()

model_win = joblib.load('modelo_ganador.joblib')
feat_win = joblib.load('features_ganador.joblib')
model_games = joblib.load('modelo_juegos.joblib')
feat_games = joblib.load('features_juegos.joblib')

# Cargar datos
df = pd.read_csv("atp_matches_procesados.csv")
# Optimizamos para búsqueda rápida (convertimos a string para evitar problemas de tipos)
df['player_name'] = df['player_name'].astype(str)
df['tourney_date'] = pd.to_datetime(df['tourney_date'])
df = df.sort_values(by='tourney_date')

print("¡Sistema listo!\n")

def obtener_stats(jugador):
    # Buscamos el nombre exacto (o el último que contenga ese string)
    datos = df[df['player_name'] == jugador]
    
    if len(datos) == 0:
        return None
    
    ultimo = datos.iloc[-1]
    return {
        'rank': ultimo['player_rank'],
        'form': ultimo['player_form'],
        'avg_games': ultimo['player_avg_games']
    }

def calcular_h2h(j1, j2):
    matches = df[(df['player_name'] == j1) & (df['opponent_name'] == j2)]
    return matches['result'].sum(), len(matches)

# --- BUCLE PRINCIPAL ---
while True:
    print("\n" + "="*40)
    print(" Escribe los nombres EXACTOS (copialos del Detective si hace falta)")
    j1 = input("Jugador 1 (ej: Stefano Napolitano): ").strip()
    if j1.lower() == 'salir': break
    
    j2 = input("Jugador 2 (ej: Otto Virtanen): ").strip()
    
    # --- ¡AQUÍ ESTABA EL FALLO! AÑADIMOS PREGUNTA DE SETS ---
    try:
        surf_input = input("Superficie (Hard, Clay, Grass) [Enter = Hard]: ").strip().capitalize()
        if not surf_input: surf_input = "Hard"
        
        bo_input = input("¿Mejor de 3 o 5 sets? [Enter = 3]: ").strip()
        best_of = int(bo_input) if bo_input else 3
    except:
        best_of = 3
    # --------------------------------------------------------
    
    s1 = obtener_stats(j1)
    s2 = obtener_stats(j2)
    
    if not s1: print(f"❌ No encuentro a '{j1}'"); continue
    if not s2: print(f"❌ No encuentro a '{j2}'"); continue

    # --- 1. PREDICCIÓN GANADOR ---
    h2h_wins, h2h_total = calcular_h2h(j1, j2)
    
    input_win = {
        'best_of': best_of, # <-- Ahora sí enviamos el dato correcto
        'player_rank': s1['rank'], 'opponent_rank': s2['rank'],
        'player_form': s1['form'], 'opponent_form': s2['form'],
        'h2h_wins': h2h_wins, 'h2h_total': h2h_total
    }
    # Superficie (One Hot Encoding manual)
    for f in feat_win: 
        if f.startswith('surface_'): input_win[f] = 0
    if f'surface_{surf_input}' in input_win: input_win[f'surface_{surf_input}'] = 1
    
    # Reindex asegura que el orden de columnas sea IDÉNTICO al entrenamiento
    df_win = pd.DataFrame([input_win]).reindex(columns=feat_win, fill_value=0)
    prob = model_win.predict_proba(df_win)[0][1]
    
    # --- 2. PREDICCIÓN JUEGOS ---
    input_games = {
        'best_of': best_of, # <-- Ahora sí enviamos el dato correcto
        'player_rank': s1['rank'], 'opponent_rank': s2['rank'],
        'player_avg_games': s1['avg_games'], 'opponent_avg_games': s2['avg_games']
    }
    for f in feat_games:
        if f.startswith('surface_'): input_games[f] = 0
    if f'surface_{surf_input}' in input_games: input_games[f'surface_{surf_input}'] = 1
    
    df_games = pd.DataFrame([input_games]).reindex(columns=feat_games, fill_value=0)
    pred_games = model_games.predict(df_games)[0]

    # --- RESULTADOS ---
    print(f"\n📊 {j1} vs {j2} ({surf_input} - Bo{best_of})")
    print(f"   Stats: Rank {s1['rank']:.0f} vs {s2['rank']:.0f} | Forma {s1['form']:.2f} vs {s2['form']:.2f}")
    print("-" * 30)
    print(f"🏆 GANADOR: {j1} tiene {prob*100:.1f}% (Cuota justa: {1/prob:.2f})")
    print("-" * 30)
    print(f"🎾 TOTAL JUEGOS: {pred_games:.1f} juegos estimados")
    print(f"   👉 Línea recomendada: Más/Menos de {round(pred_games)} juegos")
    print("="*40)