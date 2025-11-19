import os
# --- PARCHE PARA WINDOWS Y JOBLIB ---
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
# ------------------------------------

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

print("--- 2. Ingeniería de Features (Cruce Oponente) ---")

# CORRECCIÓN DEL ERROR:
# Quitamos 'player_elo' de aquí porque 'opponent_elo' YA EXISTE en el CSV.
# Solo buscamos las métricas rodantes (rolling stats) y el descanso.
cols_stats = [
    'days_rest', 
    'player_form_last_5', 
    'player_ace_avg', 
    'player_1st_won_avg', 
    'player_bp_save_avg', 
    'player_surf_win'
]

# Creamos la tabla de búsqueda (Lookup)
lookup = df[['Date', 'player_name'] + cols_stats].copy()

# Renombramos todo lo que sea 'player_' a 'opponent_' para poder pegarlo
lookup.rename(columns={
    c: c.replace('player_', 'opponent_').replace('days_rest', 'opponent_rest') 
    for c in lookup.columns 
    if 'player_' in c or 'days' in c
}, inplace=True)

# Merge (Unión) izquierda
df = pd.merge(df, lookup, left_on=['Date', 'opponent_name'], right_on=['Date', 'opponent_name'], how='left')

# --- CREACIÓN DE VARIABLES DIFERENCIALES (DELTA) ---
# Ahora sí funcionará porque opponent_elo no se duplicó
df['diff_elo'] = df['player_elo'] - df['opponent_elo']
df['diff_rank'] = df['opponent_rank'] - df['player_rank'] # Ojo al orden: Mayor rank es peor
df['diff_form'] = df['player_form_last_5'] - df['opponent_form_last_5']
df['diff_surf'] = df['player_surf_win'] - df['opponent_surf_win']

# One Hot Encoding para Superficie
df = pd.get_dummies(df, columns=['Surface'], drop_first=True)

# Definición de Features Finales
features_base = [
    'Best of',
    'diff_elo', 'diff_rank', 'diff_form', 'diff_surf',
    'player_elo', 'opponent_elo',
    'player_surf_win', 'opponent_surf_win',
    'player_ace_avg', 'opponent_ace_avg',
    'days_rest', 'opponent_rest',
    'player_bp_save_avg', 'opponent_bp_save_avg'
]
# Añadimos dinámicamente las columnas de superficie (Surface_Clay, Surface_Grass...)
features = features_base + [c for c in df.columns if c.startswith('Surface_')]

print(f"   Variables utilizadas: {len(features)}")

# Limpieza para entrenar (Borramos filas donde falten datos críticos)
# Importante: dropna en diff_elo asegura que tengamos datos de ambos jugadores
df_train = df.dropna(subset=['diff_elo', 'result'])

X = df_train[features]
y_win = df_train['result']
y_games = df_train['total_games']

# Split Temporal (Último 10% para test para simular futuro real)
split_idx = int(len(df_train) * 0.90)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y_win.iloc[:split_idx]
y_test = y_win.iloc[split_idx:]

y_train_g = y_games.iloc[:split_idx]
y_test_g = y_games.iloc[split_idx:]

# --- MODELO 1: GANADOR ---
print("\n🚀 Entrenando Modelo Winner (Elo Enhanced)...")
model_win = HistGradientBoostingClassifier(
    max_iter=500, 
    learning_rate=0.05, 
    max_depth=5, 
    l2_regularization=1.0,
    random_state=42
)
model_win.fit(X_train, y_train)

# Evaluación
preds = model_win.predict(X_test)
probs = model_win.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, preds)
ll = log_loss(y_test, probs)

print(f"✅ Precisión Test: {acc*100:.2f}%")
print(f"✅ Log Loss: {ll:.4f} (Menor es mejor)")

# --- MODELO 2: JUEGOS ---
print("\n🚀 Entrenando Modelo Juegos...")
# Solo entrenamos juegos donde tengamos dato válido (>0)
mask_g = y_train_g > 0
model_games = HistGradientBoostingRegressor(max_iter=300, max_depth=5, random_state=42)
model_games.fit(X_train[mask_g], y_train_g[mask_g])

# Evaluación Juegos
if len(y_test_g) > 0:
    mask_test_g = y_test_g > 0
    if mask_test_g.sum() > 0:
        pred_g = model_games.predict(X_test[mask_test_g])
        mae = mean_absolute_error(y_test_g[mask_test_g], pred_g)
        print(f"✅ Error Medio Juegos: +/- {mae:.2f}")

# --- GUARDADO ---
print("\n💾 Guardando archivos del sistema...")
joblib.dump(model_win, 'modelo_ganador.joblib')
joblib.dump(model_games, 'modelo_juegos.joblib')
joblib.dump(features, 'features.joblib')

# Guardamos una mini-base de datos para la APP (solo lo último de cada jugador)
# Esto hace que la app cargue instantáneamente
print("💾 Creando base de datos optimizada para la App...")
cols_db = ['player_name', 'Date', 'player_elo', 'player_rank', 
           'player_form_last_5', 'player_surf_win', 
           'player_ace_avg', 'player_1st_won_avg', 'player_bp_save_avg']
joblib.dump(df[cols_db], 'database_reciente.joblib')

print("¡Todo listo!")