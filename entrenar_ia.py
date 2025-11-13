import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

print("--- 1. Cargando datos ---")
df = pd.read_csv("atp_matches_procesados.csv")
df['tourney_date'] = pd.to_datetime(df['tourney_date'])

print("--- 2. Preparando cruce de datos (Oponente) ---")
lookup = df[['tourney_date', 'player_name', 'player_form', 'player_avg_games']].copy()
lookup.rename(columns={
    'player_name': 'opponent_name', 
    'player_form': 'opponent_form',
    'player_avg_games': 'opponent_avg_games'
}, inplace=True)
lookup = lookup.drop_duplicates(subset=['tourney_date', 'opponent_name'])
df = pd.merge(df, lookup, on=['tourney_date', 'opponent_name'], how='left')

# Calcular H2H
df = df.sort_values(by='tourney_date')
df['h2h_wins'] = df.groupby(['player_name', 'opponent_name'])['result'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
df['h2h_total'] = df.groupby(['player_name', 'opponent_name']).cumcount()

df_final = df.dropna(subset=['player_form', 'opponent_form', 'player_avg_games', 'opponent_avg_games'])
df_final = pd.get_dummies(df_final, columns=['surface'], drop_first=True)


# === ENTRENAMIENTO MODELO 1: GANADOR (WINNER) ===
print("\n--- 🚂 Entrenando Modelo 1: GANADOR ---")

# ¡AÑADIMOS 'best_of' A LAS CARACTERÍSTICAS!
features_win = ['best_of', 'player_rank', 'opponent_rank', 'player_form', 'opponent_form', 'h2h_wins', 'h2h_total']
features_win.extend([c for c in df_final.columns if c.startswith('surface_')])

X = df_final[features_win]
y = df_final['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model_win = LogisticRegression(max_iter=2000)
model_win.fit(X_train, y_train)
acc = accuracy_score(y_test, model_win.predict(X_test))
print(f"✅ Precisión Ganador: {acc*100:.2f}%")


# === ENTRENAMIENTO MODELO 2: TOTAL JUEGOS (GAMES) ===
print("\n--- 🚂 Entrenando Modelo 2: TOTAL JUEGOS ---")

# ¡AÑADIMOS 'best_of' A LAS CARACTERÍSTICAS!
features_games = ['best_of', 'player_rank', 'opponent_rank', 'player_avg_games', 'opponent_avg_games']
features_games.extend([c for c in df_final.columns if c.startswith('surface_')])

X_g = df_final[features_games]
y_g = df_final['match_games']
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_g, y_g, test_size=0.2, shuffle=False)

model_games = LinearRegression()
model_games.fit(X_train_g, y_train_g)
mae = mean_absolute_error(y_test_g, model_games.predict(X_test_g))
print(f"✅ Margen de error promedio en juegos: +/- {mae:.2f} juegos")


# --- GUARDAR TODO ---
print("\n💾 Guardando cerebros...")
joblib.dump(model_win, 'modelo_ganador.joblib')
joblib.dump(features_win, 'features_ganador.joblib')
joblib.dump(model_games, 'modelo_juegos.joblib')
joblib.dump(features_games, 'features_juegos.joblib')
print("¡Listo!")