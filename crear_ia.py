import pandas as pd
import numpy as np
import re

# --- CONFIGURACIÓN ---
NOMBRE_ARCHIVO = "atp_tennis.csv"

# --- FUNCIONES AUXILIARES ---
def calcular_juegos(score_str):
    if not isinstance(score_str, str) or 'RET' in score_str or 'W/O' in score_str: return np.nan
    total = 0
    sets = re.findall(r'(\d+)-(\d+)', score_str)
    if not sets: return np.nan
    for g1, g2 in sets: total += int(g1) + int(g2)
    return total

print(f"--- 1. Cargando {NOMBRE_ARCHIVO} y Limpiando ---")
df = pd.read_csv(NOMBRE_ARCHIVO)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# Rellenar Nulos básicos
df['Rank_1'] = df['Rank_1'].fillna(1500)
df['Rank_2'] = df['Rank_2'].fillna(1500)
df['Best of'] = df['Best of'].fillna(3)

# Calcular Juegos
col_score = 'Score' if 'Score' in df.columns else 'score'
df['total_games'] = df[col_score].apply(calcular_juegos)

# --- 2. REESTRUCTURACIÓN (DUPLICAR FILAS) ---
print("--- 2. Creando estructura Player vs Opponent ---")
# Esto es complejo porque tenemos muchas columnas de stats.
# P1_Ace va con Player_1, P2_Ace va con Player_2.

# Definimos qué columnas van con quién
cols_base = ['Date', 'Surface', 'Best of', 'total_games']
cols_p1 = ['Player_1', 'Rank_1', 'P1_Ace', 'P1_DF', 'P1_SvPt', 'P1_1stIn', 'P1_1stWon', 'P1_BpSaved', 'P1_BpFaced']
cols_p2 = ['Player_2', 'Rank_2', 'P2_Ace', 'P2_DF', 'P2_SvPt', 'P2_1stIn', 'P2_1stWon', 'P2_BpSaved', 'P2_BpFaced']

# Renombrado estándar
renames_p1 = {c: c.replace('Player_1', 'player_name').replace('Rank_1', 'player_rank').replace('P1_', 'stats_') for c in cols_p1}
renames_p2 = {c: c.replace('Player_2', 'player_name').replace('Rank_2', 'player_rank').replace('P2_', 'stats_') for c in cols_p2}

# Opponent renames
renames_op_p1 = {c: c.replace('Player_1', 'opponent_name').replace('Rank_1', 'opponent_rank') for c in ['Player_1', 'Rank_1']}
renames_op_p2 = {c: c.replace('Player_2', 'opponent_name').replace('Rank_2', 'opponent_rank') for c in ['Player_2', 'Rank_2']}

# --- LADO 1 (Ganador en datos TML) ---
df_1 = df[cols_base + cols_p1 + ['Player_2', 'Rank_2']].copy()
df_1.rename(columns=renames_p1, inplace=True)
df_1.rename(columns=renames_op_p2, inplace=True)
df_1['result'] = 1 # P1 siempre es winner en raw

# --- LADO 2 (Perdedor en datos TML) ---
df_2 = df[cols_base + cols_p2 + ['Player_1', 'Rank_1']].copy()
df_2.rename(columns=renames_p2, inplace=True)
df_2.rename(columns=renames_op_p1, inplace=True)
df_2['result'] = 0

df_full = pd.concat([df_1, df_2], ignore_index=True).sort_values(by='Date')

# --- 3. INGENIERÍA DE CARACTERÍSTICAS AVANZADA ---
print("--- 3. Calculando Métricas Avanzadas (Rolling Stats) ---")

# Función para calcular media móvil
def get_rolling(col, window=10):
    return df_full.groupby('player_name')[col].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )

# A. Forma Básica
df_full['player_form'] = get_rolling('result').fillna(0.5)

# B. Stats de Servicio (Puede haber NaNs en Futures)
# % de Aces: Aces / Puntos de Saque
df_full['ace_pct'] = df_full['stats_Ace'] / df_full['stats_SvPt']
df_full['player_ace_avg'] = get_rolling('ace_pct').fillna(0.05) # Default 5%

# % Primer Servicio
df_full['first_in_pct'] = df_full['stats_1stIn'] / df_full['stats_SvPt']
df_full['player_1st_in_avg'] = get_rolling('first_in_pct').fillna(0.60) # Default 60%

# % Puntos Ganados con 1er Saque
df_full['first_won_pct'] = df_full['stats_1stWon'] / df_full['stats_1stIn']
df_full['player_1st_won_avg'] = get_rolling('first_won_pct').fillna(0.70) 

# C. Stats de Presión (Break Points)
# % BP Salvados
df_full['bp_save_pct'] = df_full['stats_BpSaved'] / df_full['stats_BpFaced']
df_full['player_bp_save_avg'] = get_rolling('bp_save_pct').fillna(0.55)

# D. Superficie Específica
print("Calculando Forma por Superficie...")
df_full['player_surf_win'] = df_full.groupby(['player_name', 'Surface'])['result'].transform(
    lambda x: x.shift(1).rolling(window=20, min_periods=1).mean()
).fillna(0.5)

# Limpieza final de columnas temporales
cols_keep = [
    'Date', 'Surface', 'Best of', 'player_name', 'player_rank', 
    'opponent_name', 'opponent_rank', 'result', 'total_games',
    'player_form', 'player_ace_avg', 'player_1st_in_avg', 'player_1st_won_avg', 
    'player_bp_save_avg', 'player_surf_win'
]
df_final = df_full[cols_keep].copy()

# Guardar
df_final.to_csv("atp_matches_procesados.csv", index=False)
print("✅ Datos Procesados con Métricas Avanzadas.")