import pandas as pd
import numpy as np
import os

ARCHIVO_INPUT = "nba_games.csv"
ARCHIVO_OUTPUT = "nba_processed.csv"

# Configuración Elo NBA
K_FACTOR = 20
HOME_ADVANTAGE = 100 # Puntos extra de Elo por jugar en casa
START_ELO = 1500

def calcular_four_factors(row):
    # Estimación de Posesiones (Fórmula Básica Dean Oliver)
    # Poss = FGA + 0.44*FTA + TOV - ORB
    # Protegemos contra nulos convirtiendo a 0
    fga = row.get('FGA', 0)
    fta = row.get('FTA', 0)
    tov = row.get('TOV', 0)
    oreb = row.get('OREB', 0)
    pts = row.get('PTS', 0)
    
    poss = fga + 0.44 * fta + tov - oreb
    if poss <= 0: poss = 1 # Evitar división por cero
    
    off_rtg = (pts / poss) * 100
    pace = poss
    
    return pd.Series([off_rtg, pace])

print("--- 1. Ingeniería de Datos NBA (Four Factors) ---")
if not os.path.exists(ARCHIVO_INPUT):
    print(f"❌ Error: No existe {ARCHIVO_INPUT}. Ejecuta actualizar_nba.py primero.")
    exit()

df = pd.read_csv(ARCHIVO_INPUT)

# Limpieza de nulos críticos
df.fillna(0, inplace=True)

# Asegurar tipos
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.sort_values('GAME_DATE')

# Calcular métricas avanzadas por partido
df[['OFF_RTG', 'PACE']] = df.apply(calcular_four_factors, axis=1)

# --- CÁLCULO DE ELO DINÁMICO ---
print("--- 2. Calculando Elo Histórico... ---")
elo_dict = {}
current_elos = {} 

# Agrupamos por GameID
games = df.groupby('GAME_ID')
elo_records = []

for g_id, game_df in games:
    # Necesitamos exactamente 2 filas (Local vs Visitante)
    if len(game_df) != 2: continue
    
    row1 = game_df.iloc[0]
    row2 = game_df.iloc[1]
    
    # --- CORRECCIÓN DEL ERROR ---
    # Usamos la columna IS_HOME que generamos en el actualizador.
    # Es mucho más seguro que buscar '@' en strings.
    
    # Si row1 es local (IS_HOME == 1)
    if row1.get('IS_HOME', 0) == 1:
        row_home, row_away = row1, row2
    # Si row2 es local
    elif row2.get('IS_HOME', 0) == 1:
        row_home, row_away = row2, row1
    # Fallback por seguridad (Si no hay IS_HOME, usamos lógica antigua con str)
    elif '@' in str(row1.get('MATCHUP', '')): 
        row_away, row_home = row1, row2
    else:
        # Asumimos por defecto orden estándar
        row_home, row_away = row1, row2

    t_home = row_home['TEAM_ID'] # Usamos ID mejor que nombre para evitar duplicados
    t_away = row_away['TEAM_ID']

    elo_h = elo_dict.get(t_home, START_ELO)
    elo_a = elo_dict.get(t_away, START_ELO)
    
    # Guardar Elo PREVIO al partido
    elo_records.append({'GAME_ID': g_id, 'TEAM_ID': t_home, 'ELO_START': elo_h, 'IS_HOME': 1})
    elo_records.append({'GAME_ID': g_id, 'TEAM_ID': t_away, 'ELO_START': elo_a, 'IS_HOME': 0})
    
    # Probabilidad (Con ventaja de campo)
    elo_diff = (elo_h + HOME_ADVANTAGE) - elo_a
    prob_h = 1 / (1 + 10 ** (-elo_diff / 400))
    
    # Resultado real (1 si gana Home)
    res_h = 1 if row_home['WL'] == 'W' else 0
    
    # Update Elo
    new_elo_h = elo_h + K_FACTOR * (res_h - prob_h)
    new_elo_a = elo_a + K_FACTOR * ((1-res_h) - (1-prob_h))
    
    # Margin of Victory Multiplier
    pts_h = row_home['PTS']
    pts_a = row_away['PTS']
    mov = abs(pts_h - pts_a)
    
    # Factor multiplicador de margen (evita errores logarítmicos con mov=0)
    mult = np.log(mov + 1) * (2.2 / ((elo_diff if res_h == 1 else -elo_diff)*0.001 + 2.2))
    
    elo_dict[t_home] = elo_h + (new_elo_h - elo_h) * mult
    elo_dict[t_away] = elo_a + (new_elo_a - elo_a) * mult

# Fusionar Elos con el DF principal
df_elo = pd.DataFrame(elo_records)
# Merge usando GAME_ID y TEAM_ID para exactitud
df = df.merge(df_elo, on=['GAME_ID', 'TEAM_ID'], how='inner')

# --- EWMA (Forma Reciente) ---
print("--- 3. Calculando Momentum (EWMA)... ---")
df.sort_values(['TEAM_ID', 'GAME_DATE'], inplace=True)

def get_ewma(col, span=10):
    return df.groupby('TEAM_ID')[col].transform(lambda x: x.shift(1).ewm(span=span).mean())

df['EWMA_OFF_RTG'] = get_ewma('OFF_RTG', span=10)
df['EWMA_PACE'] = get_ewma('PACE', span=10)
df['EWMA_PTS'] = get_ewma('PTS', span=10)

# Rellenar nulos iniciales (primeros partidos de la historia)
df.dropna(subset=['ELO_START', 'EWMA_OFF_RTG'], inplace=True)

df.to_csv(ARCHIVO_OUTPUT, index=False)
print(f"✅ NBA Procesada: {len(df)} registros listos para IA.")