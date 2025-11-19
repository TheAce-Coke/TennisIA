import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

print("--- ENTRENANDO IA NBA (QUANT) ---")

df = pd.read_csv("nba_processed.csv")
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# --- CRUCE DE RIVALES ---
# Necesitamos saber contra quién jugaron para calcular los Deltas
df_opp = df[['GAME_ID', 'TEAM_ID', 'ELO_START', 'EWMA_OFF_RTG', 'EWMA_PACE', 'EWMA_PTS']].copy()
df_opp.rename(columns={
    'TEAM_ID': 'OPP_ID', 
    'ELO_START': 'OPP_ELO', 
    'EWMA_OFF_RTG': 'OPP_OFF_RTG',
    'EWMA_PACE': 'OPP_PACE',
    'EWMA_PTS': 'OPP_PTS'
}, inplace=True)

# Merge del partido consigo mismo pero invirtiendo equipos
# Truco: Un game tiene 2 filas. Hacemos merge por GameID pero filtramos que TEAM_ID != OPP_ID
df_full = pd.merge(df, df_opp, on='GAME_ID')
df_full = df_full[df_full['TEAM_ID'] != df_full['OPP_ID']]

# Features
df_full['home_adv'] = df_full['IS_HOME'] # 1 o 0
df_full['diff_elo'] = df_full['ELO_START'] - df_full['OPP_ELO']
df_full['diff_off'] = df_full['EWMA_OFF_RTG'] - df_full['OPP_OFF_RTG']

features = ['home_adv', 'diff_elo', 'diff_off', 'ELO_START', 'OPP_ELO', 'EWMA_OFF_RTG', 'OPP_OFF_RTG', 'EWMA_PACE', 'OPP_PACE']
target_win = df_full['WL'].apply(lambda x: 1 if x == 'W' else 0)
target_points = df_full['PTS']

# Split
split = int(len(df_full) * 0.90)
X_train, X_test = df_full[features].iloc[:split], df_full[features].iloc[split:]
y_win_train, y_win_test = target_win.iloc[:split], target_win.iloc[split:]
y_pts_train, y_pts_test = target_points.iloc[:split], target_points.iloc[split:]

# 1. Modelo Ganador
print("🚀 Entrenando Winner Model...")
model_win = HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=0.1)
model_win.fit(X_train, y_win_train)
acc = accuracy_score(y_win_test, model_win.predict(X_test))
print(f"✅ Accuracy NBA: {acc:.1%}")

# 2. Modelo Puntos (Para calibrar el Pace en Monte Carlo)
print("🚀 Entrenando Points Model...")
model_pts = HistGradientBoostingRegressor(max_iter=200, max_depth=5)
model_pts.fit(X_train, y_pts_train)
mae = mean_absolute_error(y_pts_test, model_pts.predict(X_test))
print(f"✅ Error Medio Puntos: +/- {mae:.1f}")

# Guardar Modelos
joblib.dump(model_win, 'nba_model_win.joblib')
joblib.dump(model_pts, 'nba_model_pts.joblib')
joblib.dump(features, 'nba_features.joblib')

# Guardar DB Reciente (Último partido de cada equipo)
print("💾 Guardando Stats Actuales...")
last_games = df_full.sort_values('GAME_DATE').groupby('TEAM_NAME').tail(1)
cols_db = ['TEAM_NAME', 'ELO_START', 'EWMA_OFF_RTG', 'EWMA_PACE', 'EWMA_PTS']
joblib.dump(last_games[cols_db], 'nba_db_teams.joblib')

print("¡Sistema NBA Listo!")