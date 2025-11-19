import pandas as pd
import numpy as np
import os

ARCHIVO_INPUT = "nba_games.csv"
ARCHIVO_OUTPUT = "nba_processed.csv"

# Configuración Elo NBA
K_FACTOR = 20
HOME_ADVANTAGE = 100 
START_ELO = 1500

def calcular_four_factors(row):
    fga = row.get('FGA', 0)
    fta = row.get('FTA', 0)
    tov = row.get('TOV', 0)
    oreb = row.get('OREB', 0)
    pts = row.get('PTS', 0)
    
    poss = fga + 0.44 * fta + tov - oreb
    if poss <= 0: poss = 1 
    
    off_rtg = (pts / poss) * 100
    pace = poss
    
    return pd.Series([off_rtg, pace])

print("--- 1. Ingeniería de Datos NBA (Four Factors) ---")
if not os.path.exists(ARCHIVO_INPUT):
    print(f"❌ Error: No existe {ARCHIVO_INPUT}. Ejecuta actualizar_nba.py primero.")
    exit()

df = pd.read_csv(ARCHIVO_INPUT)
df.fillna(0, inplace=True)
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.sort_values('GAME_DATE')

df[['OFF_RTG', 'PACE']] = df.apply(calcular_four_factors, axis=1)

# --- CÁLCULO DE ELO ---
print("--- 2. Calculando Elo Histórico... ---")
elo_dict = {}
games = df.groupby('GAME_ID')
elo_records = []

for g_id, game_df in games:
    if len(game_df) != 2: continue
    
    row1 = game_df.iloc[0]
    row2 = game_df.iloc[1]
    
    # Detección Local/Visitante usando IS_HOME
    if row1.get('IS_HOME', 0) == 1:
        row_home, row_away = row1, row2
    elif row2.get('IS_HOME', 0) == 1:
        row_home, row_away = row2, row1
    elif '@' in str(row1.get('MATCHUP', '')): 
        row_away, row_home = row1, row2
    else:
        row_home, row_away = row1, row2

    t_home = row_home['TEAM_ID']
    t_away = row_away['TEAM_ID']

    elo_h = elo_dict.get(t_home, START_ELO)
    elo_a = elo_dict.get(t_away, START_ELO)
    
    # Guardamos datos. NOTA: Guardamos IS_HOME aquí pero lo borraremos antes del merge
    elo_records.append({'GAME_ID': g_id, 'TEAM_ID': t_home, 'ELO_START': elo_h})
    elo_records.append({'GAME_ID': g_id, 'TEAM_ID': t_away, 'ELO_START': elo_a})
    
    # Cálculo Elo
    elo_diff = (elo_h + HOME_ADVANTAGE) - elo_a
    prob_h = 1 / (1 + 10 ** (-elo_diff / 400))
    res_h = 1 if row_home['WL'] == 'W' else 0
    
    new_elo_h = elo_h + K_FACTOR * (res_h - prob_h)
    new_elo_a = elo_a + K_FACTOR * ((1-res_h) - (1-prob_h))
    
    # Multiplicador por margen de victoria
    mov = abs(row_home['PTS'] - row_away['PTS'])
    mult = np.log(mov + 1) * (2.2 / ((elo_diff if res_h == 1 else -elo_diff)*0.001 + 2.2))
    
    elo_dict[t_home] = elo_h + (new_elo_h - elo_h) * mult
    elo_dict[t_away] = elo_a + (new_elo_a - elo_a) * mult

# --- MERGE CORREGIDO ---
df_elo = pd.DataFrame(elo_records)

# El DataFrame original 'df' YA TIENE la columna IS_HOME.
# 'df_elo' NO debe tenerla para evitar que se creen IS_HOME_x e IS_HOME_y.
# Hacemos el merge solo con GAME_ID, TEAM_ID y ELO_START.
df = df.merge(df_elo, on=['GAME_ID', 'TEAM_ID'], how='inner')

# --- EWMA ---
print("--- 3. Calculando Momentum (EWMA)... ---")
df.sort_values(['TEAM_ID', 'GAME_DATE'], inplace=True)

def get_ewma(col, span=10):
    return df.groupby('TEAM_ID')[col].transform(lambda x: x.shift(1).ewm(span=span).mean())

df['EWMA_OFF_RTG'] = get_ewma('OFF_RTG', span=10)
df['EWMA_PACE'] = get_ewma('PACE', span=10)
df['EWMA_PTS'] = get_ewma('PTS', span=10)

df.dropna(subset=['ELO_START', 'EWMA_OFF_RTG'], inplace=True)

# Verificación de seguridad antes de guardar
if 'IS_HOME' not in df.columns:
    print("⚠️ Advertencia: Regenerando columna IS_HOME...")
    # Fallback por si acaso
    df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 0 if '@' in str(x) else 1)

df.to_csv(ARCHIVO_OUTPUT, index=False)
print(f"✅ NBA Procesada: {len(df)} registros listos para IA.")