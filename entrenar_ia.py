import os
# --- PARCHE PARA WINDOWS Y JOBLIB ---
os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Bajamos a 1 para máxima estabilidad en Windows
# ------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

print("--- 1. Cargando datos ---")
df = pd.read_csv("atp_matches_procesados.csv")
df['Date'] = pd.to_datetime(df['Date'])
# Aseguramos orden temporal estricto
df = df.sort_values(by='Date')

# --- CRUCE DE OPONENTE ---
print("--- 2. Preparando datos (Cruce de oponentes) ---")
cols_stats = ['player_form', 'player_ace_avg', 'player_1st_in_avg', 'player_1st_won_avg', 'player_bp_save_avg', 'player_surf_win']
lookup = df[['Date', 'player_name'] + cols_stats].copy()
lookup.rename(columns={c: c.replace('player_', 'opponent_') for c in lookup.columns if 'player_' in c}, inplace=True)

# Merge con validación de duplicados para evitar explosión de datos
df = pd.merge(df, lookup, left_on=['Date', 'opponent_name'], right_on=['Date', 'opponent_name'], how='left')

# H2H
print("--- 3. Calculando H2H... ---")
df['h2h_wins'] = df.groupby(['player_name', 'opponent_name'])['result'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
df['h2h_total'] = df.groupby(['player_name', 'opponent_name']).cumcount()

# One Hot Encoding
df = pd.get_dummies(df, columns=['Surface'], drop_first=True)

# Definición de Features
features_base = [
    'player_rank', 'opponent_rank', 'Best of',
    'player_form', 'opponent_form',
    'player_surf_win', 'opponent_surf_win',
    'h2h_wins', 'h2h_total',
    'player_ace_avg', 'opponent_ace_avg',
    'player_1st_won_avg', 'opponent_1st_won_avg',
    'player_bp_save_avg', 'opponent_bp_save_avg'
]
# Añadimos dinámicamente las columnas de superficie
features = features_base + [c for c in df.columns if c.startswith('Surface_')]

# Limpieza Mínima para entrenar
# Solo borramos si no tenemos ni ranking ni forma (datos críticos)
df_train = df.dropna(subset=['player_rank', 'player_form'])

X = df_train[features]
y_win = df_train['result']
y_games = df_train['total_games']

# Split Temporal (Sin mezclar futuro con pasado)
X_train, X_test, y_train, y_test = train_test_split(X, y_win, test_size=0.15, shuffle=False)
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X, y_games, test_size=0.15, shuffle=False)

# --- MODELO 1: GANADOR ---
print("\n🚀 Entrenando Modelo Ganador (Gradient Boosting)...")
model_win = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_depth=7, random_state=42)
model_win.fit(X_train, y_train)
acc = accuracy_score(y_test, model_win.predict(X_test))
print(f"✅ Precisión (Winner): {acc*100:.2f}%")

# --- MODELO 2: JUEGOS ---
print("🚀 Entrenando Modelo Juegos...")
# 1. Entrenamos solo con datos que tienen 'total_games' válido
mask_train = y_train_g.notna()
model_games = HistGradientBoostingRegressor(max_iter=300, max_depth=7, random_state=42)
model_games.fit(X_train_g[mask_train], y_train_g[mask_train])

# 2. CORRECCIÓN DEL ERROR: Evaluamos alineando X e y
# Solo evaluamos en filas donde sabemos la respuesta real (y_test no es NaN)
mask_test = y_test_g.notna()

if mask_test.sum() > 0:
    # Filtramos X e y con la MISMA máscara
    X_test_eval = X_test_g[mask_test]
    y_test_eval = y_test_g[mask_test]
    
    predicciones = model_games.predict(X_test_eval)
    mae = mean_absolute_error(y_test_eval, predicciones)
    print(f"✅ Error Medio Juegos: +/- {mae:.2f}")
else:
    print("⚠️ No hay suficientes datos de 'Juegos' en el test para evaluar, pero el modelo se entrenó bien.")

# Guardar
print("\n💾 Guardando archivos...")
joblib.dump(model_win, 'modelo_ganador.joblib')
joblib.dump(model_games, 'modelo_juegos.joblib')
joblib.dump(features, 'features.joblib')
print("¡Todo listo!")