import pandas as pd
import numpy as np
import re

NOMBRE_ARCHIVO = "atp_tennis.csv"
K_FACTOR = 32
START_ELO = 1500

# --- FUNCIONES AUXILIARES ---
def calcular_juegos(score_str):
    if not isinstance(score_str, str) or 'RET' in score_str or 'W/O' in score_str: return np.nan
    total = 0
    sets = re.findall(r'(\d+)-(\d+)', score_str)
    if not sets: return np.nan
    for g1, g2 in sets: total += int(g1) + int(g2)
    return total

def calcular_elo_optimizado(df):
    elo_dict = {}
    p1_elo, p2_elo = [], []
    for index, row in df.iterrows():
        p1, p2 = row['Player_1'], row['Player_2']
        e1, e2 = elo_dict.get(p1, START_ELO), elo_dict.get(p2, START_ELO)
        p1_elo.append(e1); p2_elo.append(e2)
        prob_p1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        # Update
        elo_dict[p1] = e1 + K_FACTOR * (1 - prob_p1)
        elo_dict[p2] = e2 + K_FACTOR * (0 - (1 - prob_p1))
    df['elo_1'] = p1_elo
    df['elo_2'] = p2_elo
    return df

def safe_div(a, b): return np.where(b > 0, a / b, 0)

print(f"--- 1. Ingeniería de Datos Quant (Stats Reales) ---")
df = pd.read_csv(NOMBRE_ARCHIVO)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# Limpieza y Nulos
cols_stats = ['P1_SvPt', 'P1_1stIn', 'P1_1stWon', 'P1_2ndWon', 'P2_SvPt', 'P2_1stIn', 'P2_1stWon', 'P2_2ndWon']
for c in cols_stats:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    else: df[c] = 0

df = calcular_elo_optimizado(df)
col_score = 'Score' if 'Score' in df.columns else 'score'
df['total_games'] = df[col_score].apply(calcular_juegos)

# --- ESTADÍSTICAS REALES DE SAQUE/RESTO ---
# Calculamos % de puntos ganados al saque (Serve Points Won)
df['P1_SrvPtsWon'] = df['P1_1stWon'] + df['P1_2ndWon']
df['P2_SrvPtsWon'] = df['P2_1stWon'] + df['P2_2ndWon']

df['P1_Serve_Pct'] = safe_div(df['P1_SrvPtsWon'], df['P1_SvPt'])
df['P2_Serve_Pct'] = safe_div(df['P2_SrvPtsWon'], df['P2_SvPt'])

# Calculamos % de puntos ganados al resto (Return Points Won)
# Puntos Resto Ganados = Puntos Saque del Rival Totales - Puntos Saque del Rival Ganados
df['P1_Rtn_Pct'] = 1 - df['P2_Serve_Pct']
df['P2_Rtn_Pct'] = 1 - df['P1_Serve_Pct']

# --- DUPLICACIÓN ---
cols_p1 = ['Player_1', 'Rank_1', 'elo_1', 'P1_Serve_Pct', 'P1_Rtn_Pct']
cols_p2 = ['Player_2', 'Rank_2', 'elo_2', 'P2_Serve_Pct', 'P2_Rtn_Pct']

rn_p1 = {'Player_1': 'player_name', 'Rank_1': 'player_rank', 'elo_1': 'player_elo', 'P1_Serve_Pct': 'stats_serve', 'P1_Rtn_Pct': 'stats_return'}
rn_p2 = {'Player_2': 'player_name', 'Rank_2': 'player_rank', 'elo_2': 'player_elo', 'P2_Serve_Pct': 'stats_serve', 'P2_Rtn_Pct': 'stats_return'}

df_1 = df.copy()
df_1.rename(columns=rn_p1, inplace=True)
df_1['opponent_name'] = df['Player_2']
df_1['opponent_rank'] = df['Rank_2']
df_1['opponent_elo'] = df['elo_2']
df_1['result'] = 1

df_2 = df.copy()
df_2.rename(columns=rn_p2, inplace=True)
df_2['opponent_name'] = df['Player_1']
df_2['opponent_rank'] = df['Rank_1']
df_2['opponent_elo'] = df['elo_1']
df_2['result'] = 0

df_full = pd.concat([df_1, df_2], ignore_index=True).sort_values(by='Date')

# --- EWMA (Medias Móviles) ---
def get_ewma(col, span=20):
    return df_full.groupby('player_name')[col].transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean()).fillna(0.60) # Default tenis 60%

df_full['ewma_form'] = df_full.groupby('player_name')['result'].transform(lambda x: x.shift(1).ewm(span=5).mean()).fillna(0.5)
df_full['ewma_serve'] = get_ewma('stats_serve', span=30) # Estabilidad al saque
df_full['ewma_return'] = get_ewma('stats_return', span=30) # Calidad de resto
df_full['ewma_surface'] = df_full.groupby(['player_name', 'Surface'])['result'].transform(lambda x: x.shift(1).ewm(span=15).mean()).fillna(0.5)

# Fatiga
df_full['last_match'] = df_full.groupby('player_name')['Date'].shift(1)
df_full['days_rest'] = (df_full['Date'] - df_full['last_match']).dt.days.fillna(10).clip(upper=30)

cols_final = [
    'Date', 'Surface', 'Best of', 'player_name', 'opponent_name',
    'player_rank', 'player_elo', 'opponent_rank', 'opponent_elo',
    'ewma_form', 'ewma_serve', 'ewma_return', 'ewma_surface', 'days_rest',
    'result', 'total_games'
]

df_final = df_full[cols_final].fillna(0)
df_final.to_csv("atp_matches_procesados.csv", index=False)
print(f"✅ Datos procesados (Saque/Resto Real): {len(df_final)} registros.")