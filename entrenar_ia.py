import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score

print("--- ENTRENAMIENTO QUANT (CALIBRADO) ---")

df = pd.read_csv("atp_matches_procesados.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# --- CRUCE DE OPONENTES ---
cols_stats = ['ewma_form', 'ewma_serve', 'ewma_surface', 'days_rest']
lookup = df[['Date', 'player_name'] + cols_stats].copy()
lookup.rename(columns={c: c.replace('player_', 'opponent_').replace('ewma_', 'opp_ewma_').replace('days_', 'opp_days_') 
                       for c in lookup.columns}, inplace=True)

df = pd.merge(df, lookup, left_on=['Date', 'opponent_name'], right_on=['Date', 'opponent_name'], how='left')

# Delta Features (La clave de la predicción)
df['delta_elo'] = df['player_elo'] - df['opponent_elo']
df['delta_form'] = df['ewma_form'] - df['opp_ewma_form']
df['delta_serve'] = df['ewma_serve'] - df['opp_ewma_serve']
df['delta_surf'] = df['ewma_surface'] - df['opp_ewma_surface']

df = pd.get_dummies(df, columns=['Surface'], drop_first=True)

features = [
    'Best of', 'delta_elo', 'delta_form', 'delta_serve', 'delta_surf',
    'player_elo', 'opponent_elo', 'ewma_serve', 'opp_ewma_serve',
    'days_rest', 'opp_days_rest'
] + [c for c in df.columns if 'Surface_' in c]

# Split temporal estricto
mask_valid = df['delta_elo'].notna() & df['result'].notna()
df_train = df[mask_valid].copy()

split = int(len(df_train) * 0.90)
X_train = df_train[features].iloc[:split]
X_test = df_train[features].iloc[split:]
y_train = df_train['result'].iloc[:split]
y_test = df_train['result'].iloc[split:]

# --- MODELO BASE + CALIBRACIÓN ---
print("🚀 Entrenando y Calibrando Modelo...")
base_model = HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=300, max_depth=5, l2_regularization=1, random_state=42
)

# CalibratedClassifierCV ajusta las probabilidades para que sean "reales"
# method='isotonic' es mejor para grandes datasets
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrated_model.fit(X_train, y_train)

# Evaluación Profesional
probs = calibrated_model.predict_proba(X_test)[:, 1]
preds = (probs > 0.5).astype(int)
acc = accuracy_score(y_test, preds)
brier = brier_score_loss(y_test, probs)

print(f"✅ Accuracy: {acc:.2%}")
print(f"✅ Brier Score: {brier:.4f} (Objetivo < 0.20 para rentabilidad)")

# Guardado
joblib.dump(calibrated_model, 'modelo_calibrado.joblib')
joblib.dump(features, 'features.joblib')

# Base de datos ligera para la APP (último registro por jugador)
print("💾 Generando DB optimizada...")
cols_db = ['player_name', 'Date', 'player_rank', 'player_elo', 'ewma_form', 'ewma_serve', 'ewma_return', 'ewma_surface', 'days_rest']
df_last = df.sort_values('Date').groupby('player_name').tail(1)[cols_db]
joblib.dump(df_last, 'db_players.joblib')

print("¡Sistema Quant Listo!")