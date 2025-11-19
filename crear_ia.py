import pandas as pd
import numpy as np
import re

NOMBRE_ARCHIVO = "atp_tennis.csv"

# --- CONFIGURACIÓN ELO ---
K_FACTOR = 32
START_ELO = 1500

def calcular_juegos(score_str):
    if not isinstance(score_str, str) or 'RET' in score_str or 'W/O' in score_str: return np.nan
    total = 0
    sets = re.findall(r'(\d+)-(\d+)', score_str)
    if not sets: return np.nan
    for g1, g2 in sets: total += int(g1) + int(g2)
    return total

def calcular_elo(df):
    # Inicializar diccionario de Elo
    elo_dict = {}
    
    # Listas para almacenar el Elo histórico
    p1_elo_list = []
    p2_elo_list = []
    
    print("   ⮑ Calculando ELO histórico punto a punto...")
    
    for index, row in df.iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        
        # Obtener Elo actual o inicial
        elo_p1 = elo_dict.get(p1, START_ELO)
        elo_p2 = elo_dict.get(p2, START_ELO)
        
        # Guardar el Elo PREVIO al partido (esto es lo que usará el modelo)
        p1_elo_list.append(elo_p1)
        p2_elo_list.append(elo_p2)
        
        # Calcular probabilidad esperada
        prob_p1 = 1 / (1 + 10 ** ((elo_p2 - elo_p1) / 400))
        
        # Actualizar Elo (Asumimos Player_1 siempre es el ganador en el CSV raw)
        # Si usas un CSV donde P1 no siempre gana, necesitas la columna 'Winner'
        actual_score = 1 # Player 1 ganó
        
        new_elo_p1 = elo_p1 + K_FACTOR * (actual_score - prob_p1)
        new_elo_p2 = elo_p2 + K_FACTOR * ((1 - actual_score) - (1 - prob_p1))
        
        elo_dict[p1] = new_elo_p1
        elo_dict[p2] = new_elo_p2
        
    df['elo_1'] = p1_elo_list
    df['elo_2'] = p2_elo_list
    return df

print(f"--- 1. Cargando {NOMBRE_ARCHIVO} ---")
df = pd.read_csv(NOMBRE_ARCHIVO)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# Limpieza básica
df['Rank_1'] = pd.to_numeric(df['Rank_1'], errors='coerce').fillna(2000)
df['Rank_2'] = pd.to_numeric(df['Rank_2'], errors='coerce').fillna(2000)

# Rellenar nulos en stats
cols_stats_raw = ['P1_Ace', 'P1_DF', 'P1_SvPt', 'P1_1stIn', 'P1_1stWon', 'P1_BpSaved', 'P1_BpFaced',
                  'P2_Ace', 'P2_DF', 'P2_SvPt', 'P2_1stIn', 'P2_1stWon', 'P2_BpSaved', 'P2_BpFaced']

for c in cols_stats_raw:
    if c not in df.columns: df[c] = 0
    else: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

# Calcular Elo antes de duplicar filas
df = calcular_elo(df)

# Calcular juegos
col_score = 'Score' if 'Score' in df.columns else 'score'
df['total_games'] = df[col_score].apply(calcular_juegos)

print("--- 2. Reestructurando ---")
# Necesitamos duplicar el dataset para que el modelo aprenda desde la perspectiva de ambos jugadores
cols_p1 = ['Player_1', 'Rank_1', 'elo_1'] + [c for c in cols_stats_raw if 'P1_' in c]
cols_p2 = ['Player_2', 'Rank_2', 'elo_2'] + [c for c in cols_stats_raw if 'P2_' in c]

rn_p1 = {'Player_1': 'player_name', 'Rank_1': 'player_rank', 'elo_1': 'player_elo'}
rn_p1.update({c: c.replace('P1_', 'stats_') for c in cols_p1 if 'P1_' in c})

rn_p2 = {'Player_2': 'player_name', 'Rank_2': 'player_rank', 'elo_2': 'player_elo'}
rn_p2.update({c: c.replace('P2_', 'stats_') for c in cols_p2 if 'P2_' in c})

# Perspectiva P1 (Ganador)
df_1 = df.copy()
df_1.rename(columns=rn_p1, inplace=True)
df_1['opponent_name'] = df['Player_2']
df_1['opponent_rank'] = df['Rank_2']
df_1['opponent_elo'] = df['elo_2']
df_1['result'] = 1

# Perspectiva P2 (Perdedor)
df_2 = df.copy()
df_2.rename(columns=rn_p2, inplace=True)
df_2['opponent_name'] = df['Player_1']
df_2['opponent_rank'] = df['Rank_1']
df_2['opponent_elo'] = df['elo_1']
df_2['result'] = 0

df_full = pd.concat([df_1, df_2], ignore_index=True).sort_values(by='Date')

print("--- 3. Métricas Avanzadas (Rolling Windows) ---")
def safe_div(a, b):
    return np.where(b > 0, a / b, 0)

# Stats base
df_full['ace_pct'] = safe_div(df_full['stats_Ace'], df_full['stats_SvPt'])
df_full['1st_won_pct'] = safe_div(df_full['stats_1stWon'], df_full['stats_1stIn'])
df_full['bp_save_pct'] = safe_div(df_full['stats_BpSaved'], df_full['stats_BpFaced'])

# Rolling function optimizada
def get_rolling(col, window=20):
    return df_full.groupby('player_name')[col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean()).fillna(0)

# Stats históricas (Forma reciente)
df_full['player_form_last_5'] = df_full.groupby('player_name')['result'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0.5)
df_full['player_ace_avg'] = get_rolling('ace_pct', 50)
df_full['player_1st_won_avg'] = get_rolling('1st_won_pct', 50)
df_full['player_bp_save_avg'] = get_rolling('bp_save_pct', 50)

# Stats específicas de superficie
df_full['player_surf_win'] = df_full.groupby(['player_name', 'Surface'])['result'].transform(
    lambda x: x.shift(1).rolling(30, min_periods=1).mean()
).fillna(df_full['player_form_last_5'])

# Días desde último partido (Fatiga)
df_full['last_match_date'] = df_full.groupby('player_name')['Date'].shift(1)
df_full['days_rest'] = (df_full['Date'] - df_full['last_match_date']).dt.days.fillna(30)
# Capar descanso a 30 días para no distorsionar
df_full['days_rest'] = df_full['days_rest'].clip(upper=30)

cols_final = [
    'Date', 'Surface', 'Best of', 
    'player_name', 'player_rank', 'player_elo', 'days_rest',
    'opponent_name', 'opponent_rank', 'opponent_elo',
    'result', 'total_games', 
    'player_form_last_5', 'player_ace_avg', 'player_1st_won_avg', 'player_bp_save_avg', 'player_surf_win'
]

# Rellenar columnas faltantes con 0
for c in cols_final:
    if c not in df_full.columns: df_full[c] = 0

df_final = df_full[cols_final]
df_final.to_csv("atp_matches_procesados.csv", index=False)
print(f"✅ Procesado Completado. Filas: {len(df_final)}")