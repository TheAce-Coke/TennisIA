import pandas as pd
import numpy as np
import re

# --- FUNCIÓN AUXILIAR PARA CONTAR JUEGOS ---
def calcular_total_juegos(score_str):
    if not isinstance(score_str, str) or 'RET' in score_str or 'W/O' in score_str:
        return np.nan
    total = 0
    sets = re.findall(r'(\d+)-(\d+)', score_str)
    if not sets: return np.nan
    for games_p1, games_p2 in sets:
        total += int(games_p1) + int(games_p2)
    return total

# --- 1. CARGAR DATOS ---
NOMBRE_ARCHIVO = "atp_tennis.csv"
print(f"--- 1. Cargando {NOMBRE_ARCHIVO} ---")

try:
    df_completo = pd.read_csv(NOMBRE_ARCHIVO)
except FileNotFoundError:
    print(f"¡ERROR! No se encontró el archivo '{NOMBRE_ARCHIVO}'.")
    exit()

# Limpieza básica
try:
    df_completo['Date'] = pd.to_datetime(df_completo['Date'], format='%d/%m/%Y')
except ValueError:
    df_completo['Date'] = pd.to_datetime(df_completo['Date'], errors='coerce')

df_completo = df_completo.sort_values(by='Date')
df_completo['Rank_1'] = df_completo['Rank_1'].fillna(2000)
df_completo['Rank_2'] = df_completo['Rank_2'].fillna(2000)

# Leemos la columna "Best of"
df_completo = df_completo.dropna(subset=['Player_1', 'Player_2', 'Winner', 'Best of'])

print("Calculando total de juegos por partido...")
col_score = 'Score' if 'Score' in df_completo.columns else 'score'
if col_score in df_completo.columns:
    df_completo['total_games'] = df_completo[col_score].apply(calcular_total_juegos)
else:
    df_completo['total_games'] = np.nan

df_completo = df_completo.dropna(subset=['total_games'])
print(f"Datos cargados. Total de {len(df_completo)} partidos históricos.")


# --- 2. REESTRUCTURAR DATOS ---
print("--- 2. Reestructurando datos ---")
df_player1_is_winner = (df_completo['Winner'] == df_completo['Player_1'])

# Perspectiva J1
df_p1 = pd.DataFrame()
df_p1['tourney_date'] = df_completo['Date']
df_p1['surface'] = df_completo['Surface']
df_p1['best_of'] = df_completo['Best of'] # <-- ¡NUEVA COLUMNA!
df_p1['player_name'] = df_completo['Player_1']
df_p1['player_rank'] = df_completo['Rank_1']
df_p1['opponent_name'] = df_completo['Player_2']
df_p1['opponent_rank'] = df_completo['Rank_2']
df_p1['result'] = np.where(df_player1_is_winner, 1, 0)
df_p1['match_games'] = df_completo['total_games']

# Perspectiva J2
df_p2 = pd.DataFrame()
df_p2['tourney_date'] = df_completo['Date']
df_p2['surface'] = df_completo['Surface']
df_p2['best_of'] = df_completo['Best of'] # <-- ¡NUEVA COLUMNA!
df_p2['player_name'] = df_completo['Player_2']
df_p2['player_rank'] = df_completo['Rank_2']
df_p2['opponent_name'] = df_completo['Player_1']
df_p2['opponent_rank'] = df_completo['Rank_1']
df_p2['result'] = np.where(df_player1_is_winner, 0, 1)
df_p2['match_games'] = df_completo['total_games']

df_modelo = pd.concat([df_p1, df_p2], ignore_index=True)
df_modelo = df_modelo.sort_values(by='tourney_date')


# --- 3. INGENIERÍA DE CARACTERÍSTICAS ---
print("--- 3. Creando Estadísticas ---")

# A. Forma (Victorias)
df_modelo['player_form'] = df_modelo.groupby('player_name')['result'].transform(
    lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
).fillna(0.5)

# B. Promedio de Juegos
print("Calculando promedio de juegos recientes...")
df_modelo['player_avg_games'] = df_modelo.groupby('player_name')['match_games'].transform(
    lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
).fillna(22)

# Guardar
df_modelo.to_csv("atp_matches_procesados.csv", index=False)
print("\n¡Archivo 'atp_matches_procesados.csv' actualizado con 'Best of'!")