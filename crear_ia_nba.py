import pandas as pd
import numpy as np

ARCHIVO_INPUT = "nba_games.csv"
ARCHIVO_OUTPUT = "nba_processed.csv"

# Configuración Elo NBA
K_FACTOR = 20
HOME_ADVANTAGE = 100 # Puntos extra de Elo por jugar en casa
START_ELO = 1500

def calcular_four_factors(row):
    # Estimación de Posesiones (Fórmula Básica Dean Oliver)
    # Poss = FGA + 0.44*FTA + TOV - ORB
    poss = row['FGA'] + 0.44 * row['FTA'] + row['TOV'] - row['OREB']
    if poss == 0: poss = 1
    
    off_rtg = (row['PTS'] / poss) * 100
    pace = poss # Ritmo del partido (para este equipo)
    
    return pd.Series([off_rtg, pace])

print("--- 1. Ingeniería de Datos NBA (Four Factors) ---")
df = pd.read_csv(ARCHIVO_INPUT)
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.sort_values('GAME_DATE')

# Calcular métricas avanzadas por partido
df[['OFF_RTG', 'PACE']] = df.apply(calcular_four_factors, axis=1)

# --- CÁLCULO DE ELO DINÁMICO ---
print("--- 2. Calculando Elo Histórico... ---")
elo_dict = {}
current_elos = {} # Para guardar en el CSV

# Agrupamos por GameID para tener los dos equipos del partido juntos
games = df.groupby('GAME_ID')

elo_records = []

for g_id, game_df in games:
    # Identificar equipos
    if len(game_df) != 2: continue
    
    row1 = game_df.iloc[0]
    row2 = game_df.iloc[1]
    
    t1, t2 = row1['TEAM_NAME'], row2['TEAM_NAME']
    
    # Detectar local/visitante (La API suele poner @ para visitante vs local)
    # Asumiremos que row1 es local si contiene 'vs', visitante si '@'
    # Simplificación: Usamos MATCHUP
    if '@' in row1['MATCHUP']: # T1 es visitante (@ T2)
        team_away, team_home = t1, t2
        row_away, row_home = row1, row2
    else: # T1 es local (vs T2)
        team_home, team_away = t1, t2
        row_home, row_away = row1, row2

    elo_h = elo_dict.get(team_home, START_ELO)
    elo_a = elo_dict.get(team_away, START_ELO)
    
    # Guardar Elo PREVIO al partido (para entrenar la IA)
    elo_records.append({'GAME_ID': g_id, 'TEAM_ID': row_home['TEAM_ID'], 'ELO_START': elo_h, 'IS_HOME': 1})
    elo_records.append({'GAME_ID': g_id, 'TEAM_ID': row_away['TEAM_ID'], 'ELO_START': elo_a, 'IS_HOME': 0})
    
    # Probabilidad (Con ventaja de campo)
    elo_diff = (elo_h + HOME_ADVANTAGE) - elo_a
    prob_h = 1 / (1 + 10 ** (-elo_diff / 400))
    
    # Resultado real (1 si gana Home)
    res_h = 1 if row_home['WL'] == 'W' else 0
    
    # Update Elo
    new_elo_h = elo_h + K_FACTOR * (res_h - prob_h)
    new_elo_a = elo_a + K_FACTOR * ((1-res_h) - (1-prob_h))
    
    # Margin of Victory Multiplier (NBA specific: ganar de 20 sube más el Elo)
    mov = abs(row_home['PTS'] - row_away['PTS'])
    mult = np.log(mov + 1) * (2.2 / ((elo_diff if res_h == 1 else -elo_diff)*0.001 + 2.2))
    
    elo_dict[team_home] = elo_h + (new_elo_h - elo_h) * mult
    elo_dict[team_away] = elo_a + (new_elo_a - elo_a) * mult

# Fusionar Elos con el DF principal
df_elo = pd.DataFrame(elo_records)
df = df.merge(df_elo, on=['GAME_ID', 'TEAM_ID'], how='left')

# --- EWMA (Forma Reciente) ---
print("--- 3. Calculando Momentum (EWMA)... ---")
# Ordenar por equipo y fecha
df.sort_values(['TEAM_ID', 'GAME_DATE'], inplace=True)

def get_ewma(col, span=10):
    return df.groupby('TEAM_ID')[col].transform(lambda x: x.shift(1).ewm(span=span).mean())

df['EWMA_OFF_RTG'] = get_ewma('OFF_RTG', span=10) # Eficiencia Ofensiva reciente
df['EWMA_PACE'] = get_ewma('PACE', span=10)       # Ritmo reciente
df['EWMA_PTS'] = get_ewma('PTS', span=10)         # Puntos recientes

# Defensa = Puntos permitidos (aproximado cruzando datos)
# Para simplificar en este paso, usaremos el OFF_RTG del rival como proxy en el entrenamiento
# Pero guardaremos lo esencial.

df.dropna(subset=['ELO_START', 'EWMA_OFF_RTG'], inplace=True)
df.to_csv(ARCHIVO_OUTPUT, index=False)
print("✅ NBA Procesada. Listo para entrenar.")