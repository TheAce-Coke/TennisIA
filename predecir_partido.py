import pandas as pd
import joblib
import numpy as np

print("--- Cargando IA Completa (Ganador + Juegos) ---")
try:
    model_win = joblib.load('modelo_ganador.joblib')
    feat_win = joblib.load('features_ganador.joblib')
    
    model_games = joblib.load('modelo_juegos.joblib')
    feat_games = joblib.load('features_juegos.joblib')
    
    # Cargamos datos para buscar stats
    df = pd.read_csv("atp_matches_procesados.csv")
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    df = df.sort_values(by='tourney_date')
except:
    print("❌ Faltan archivos. Ejecuta 'entrenar_ia.py' de nuevo.")
    exit()

def obtener_stats(jugador):
    datos = df[df['player_name'] == jugador]
    if len(datos) == 0: return None
    ultimo = datos.iloc[-1]
    return {
        'rank': ultimo['player_rank'],
        'form': ultimo['player_form'],
        'avg_games': ultimo['player_avg_games']
    }

def calcular_h2h(j1, j2):
    matches = df[(df['player_name'] == j1) & (df['opponent_name'] == j2)]
    return matches['result'].sum(), len(matches)

while True:
    print("\n" + "="*30)
    j1 = input("Jugador 1 (ej: Sinner J.): ").strip()
    if j1 == 'salir': break
    j2 = input("Jugador 2 (ej: Shelton B.): ").strip()
    surf = input("Superficie (Hard, Clay, Grass): ").strip().capitalize()
    
    s1 = obtener_stats(j1)
    s2 = obtener_stats(j2)
    
    if not s1 or not s2:
        print("❌ Jugador no encontrado.")
        continue

    # --- 1. PREDICCIÓN GANADOR ---
    h2h_wins, h2h_total = calcular_h2h(j1, j2)
    
    input_win = {
        'player_rank': s1['rank'], 'opponent_rank': s2['rank'],
        'player_form': s1['form'], 'opponent_form': s2['form'],
        'h2h_wins': h2h_wins, 'h2h_total': h2h_total
    }
    # Superficie
    for f in feat_win: 
        if f.startswith('surface_'): input_win[f] = 0
    if f'surface_{surf}' in input_win: input_win[f'surface_{surf}'] = 1
    
    df_win = pd.DataFrame([input_win]).reindex(columns=feat_win, fill_value=0)
    prob = model_win.predict_proba(df_win)[0][1]
    
    # --- 2. PREDICCIÓN JUEGOS ---
    input_games = {
        'player_rank': s1['rank'], 'opponent_rank': s2['rank'],
        'player_avg_games': s1['avg_games'], 'opponent_avg_games': s2['avg_games']
    }
    for f in feat_games:
        if f.startswith('surface_'): input_games[f] = 0
    if f'surface_{surf}' in input_games: input_games[f'surface_{surf}'] = 1
    
    df_games = pd.DataFrame([input_games]).reindex(columns=feat_games, fill_value=0)
    pred_games = model_games.predict(df_games)[0]

    # --- RESULTADOS ---
    print(f"\n📊 {j1} vs {j2} ({surf})")
    print(f"   Stats: Rank {s1['rank']} vs {s2['rank']} | Forma {s1['form']:.2f} vs {s2['form']:.2f}")
    print("-" * 30)
    print(f"🏆 GANADOR: {j1} tiene {prob*100:.1f}% (Cuota justa: {1/prob:.2f})")
    print("-" * 30)
    print(f"🎾 TOTAL JUEGOS: {pred_games:.1f} juegos estimados")
    print(f"   👉 Línea recomendada: Más/Menos de {round(pred_games)} juegos")
    print("="*30)