import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
import joblib

print("--- 1. Cargando datos procesados ---")
df = pd.read_csv("atp_matches_procesados.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

print("--- 2. Ingeniería de Features ---")
cols_stats = ['days_rest', 'player_form_last_5', 'player_ace_avg', 'player_1st_won_avg', 'player_bp_save_avg', 'player_surf_win']
lookup = df[['Date', 'player_name'] + cols_stats].copy()
lookup.rename(columns={c: c.replace('player_', 'opponent_').replace('days_rest', 'opponent_rest') for c in lookup.columns if 'player_' in c or 'days' in c}, inplace=True)

df = pd.merge(df, lookup, left_on=['Date', 'opponent_name'], right_on=['Date', 'opponent_name'], how='left')

df['diff_elo'] = df['player_elo'] - df['opponent_elo']
df['diff_rank'] = df['opponent_rank'] - df['player_rank']
df['diff_form'] = df['player_form_last_5'] - df['opponent_form_last_5']
df['diff_surf'] = df['player_surf_win'] - df['opponent_surf_win']

df = pd.get_dummies(df, columns=['Surface'], drop_first=True)

features_base = [
    'Best of', 'diff_elo', 'diff_rank', 'diff_form', 'diff_surf',
    'player_elo', 'opponent_elo', 'player_surf_win', 'opponent_surf_win',
    'player_ace_avg', 'opponent_ace_avg', 'days_rest', 'opponent_rest',
    'player_bp_save_avg', 'opponent_bp_save_avg'
]
features = features_base + [c for c in df.columns if c.startswith('Surface_')]

df_train = df.dropna(subset=['diff_elo', 'result'])
X = df_train[features]
y_win = df_train['result']
y_games = df_train['total_games']

split_idx = int(len(df_train) * 0.90)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y_win.iloc[:split_idx], y_win.iloc[split_idx:]
y_train_g, y_test_g = y_games.iloc[:split_idx], y_games.iloc[split_idx:]

# --- MODELO WINNER ---
print("\n🚀 Entrenando Modelo Winner...")
model_win = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_depth=5, l2_regularization=1.0, random_state=42)
model_win.fit(X_train, y_train)
acc = accuracy_score(y_test, model_win.predict(X_test))
print(f"✅ Precisión Test: {acc*100:.2f}%")

# --- MODELO JUEGOS ---
print("\n🚀 Entrenando Modelo Juegos...")
mask_g = y_train_g > 0
model_games = HistGradientBoostingRegressor(max_iter=300, max_depth=5, random_state=42)
model_games.fit(X_train[mask_g], y_train_g[mask_g])

# --- CÁLCULO DE DESVIACIÓN ESTÁNDAR (CRUCIAL PARA OVER/UNDER) ---
# Calculamos cuánto se desvía normalmente el modelo para crear la curva de probabilidad
std_dev_games = 2.5 # Valor por defecto seguro
if len(y_test_g) > 0:
    mask_test_g = y_test_g > 0
    if mask_test_g.sum() > 0:
        preds_g = model_games.predict(X_test[mask_test_g])
        residuals = y_test_g[mask_test_g] - preds_g
        std_dev_games = residuals.std()
        print(f"✅ Desviación Estándar (Sigma): {std_dev_games:.2f} juegos")

# --- GUARDADO ---
print("\n💾 Guardando sistema completo...")
joblib.dump(model_win, 'modelo_ganador.joblib')
joblib.dump(model_games, 'modelo_juegos.joblib')
joblib.dump(features, 'features.joblib')
joblib.dump(std_dev_games, 'std_juegos.joblib') # <--- NUEVO ARCHIVO IMPORTANTE

cols_db = ['player_name', 'Date', 'player_elo', 'player_rank', 'player_form_last_5', 'player_surf_win', 'player_ace_avg', 'player_1st_won_avg', 'player_bp_save_avg']
joblib.dump(df[cols_db], 'database_reciente.joblib')
print("¡Listo!")