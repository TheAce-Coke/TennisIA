import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Tennis AI Pro", page_icon="🎾", layout="centered")

@st.cache_resource
def cargar_todo():
    # Comprobamos si existen los modelos nuevos o viejos para evitar errores
    if os.path.exists('features.joblib'):
        # Versión V3 (Nueva)
        feats = joblib.load('features.joblib')
    elif os.path.exists('features_ganador.joblib'):
        # Versión V2 (Antigua - Parche por si acaso)
        feats = joblib.load('features_ganador.joblib')
    else:
        return None

    if not os.path.exists('modelo_ganador.joblib'): return None
    
    m_win = joblib.load('modelo_ganador.joblib')
    m_games = joblib.load('modelo_juegos.joblib')
    
    # Carga de datos con corrección automática de nombres
    df = pd.read_csv("atp_matches_procesados.csv")
    
    # --- PARCHE DE SEGURIDAD PARA COLUMNAS ---
    # Si el CSV es antiguo, la columna se llama 'tourney_date'. La renombramos.
    if 'tourney_date' in df.columns:
        df.rename(columns={'tourney_date': 'Date'}, inplace=True)
    
    # Si el CSV es antiguo, los nombres de columnas de jugadores pueden variar
    if 'player_name' not in df.columns and 'Player_1' in df.columns:
         df.rename(columns={'Player_1': 'player_name', 'Player_2': 'opponent_name'}, inplace=True)
    # -----------------------------------------

    # Ahora sí podemos convertir la fecha sin miedo
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
    
    return m_win, m_games, feats, df

res = cargar_todo()

if not res: 
    st.error("❌ Error crítico: No encuentro los archivos (.joblib o .csv). Asegúrate de haberlos subido a GitHub.")
    st.stop()

model_win, model_games, feats, df = res

def get_last_stats(player):
    # Buscamos en la columna que exista
    col_jugador = 'player_name' if 'player_name' in df.columns else 'Player_1'
    
    row = df[df[col_jugador] == player]
    if row.empty: return None
    return row.iloc[-1]

st.title("🎾 Tennis AI Pro V3")
st.sidebar.header("Configuración")

# Obtener lista de jugadores de forma segura
col_p1 = 'player_name' if 'player_name' in df.columns else 'Player_1'
col_p2 = 'opponent_name' if 'opponent_name' in df.columns else 'Player_2'

players = sorted(list(set(df[col_p1].unique()) | set(df[col_p2].unique())))

p1 = st.sidebar.selectbox("Jugador 1", players, index=None, placeholder="Buscar...")
p2 = st.sidebar.selectbox("Jugador 2", players, index=None, placeholder="Buscar...")
surf = st.sidebar.selectbox("Superficie", ["Hard", "Clay", "Grass"])
bo = st.sidebar.selectbox("Sets", [3, 5])

if st.sidebar.button("⚡ Analizar"):
    if p1 and p2 and p1 != p2:
        s1, s2 = get_last_stats(p1), get_last_stats(p2)
        
        if s1 is not None and s2 is not None:
            # H2H
            h2h_data = df[(df[col_p1] == p1) & (df[col_p2] == p2)]
            h2h_w = len(h2h_data[h2h_data['result'] == 1]) if 'result' in df.columns else 0
            h2h_t = len(h2h_data)
            
            # Preparar Input con valores por defecto si faltan columnas nuevas
            input_row = {
                'player_rank': s1.get('player_rank', 500), 
                'opponent_rank': s2.get('player_rank', 500), 
                'Best of': bo,
                'player_form': s1.get('player_form', 0.5), 
                'opponent_form': s2.get('player_form', 0.5),
                'player_surf_win': s1.get('player_surf_win', 0.5), 
                'opponent_surf_win': s2.get('player_surf_win', 0.5),
                'h2h_wins': h2h_w, 'h2h_total': h2h_t,
                'player_ace_avg': s1.get('player_ace_avg', 0.05), 
                'opponent_ace_avg': s2.get('player_ace_avg', 0.05),
                'player_1st_won_avg': s1.get('player_1st_won_avg', 0.6), 
                'opponent_1st_won_avg': s2.get('player_1st_won_avg', 0.6),
                'player_bp_save_avg': s1.get('player_bp_save_avg', 0.5), 
                'opponent_bp_save_avg': s2.get('player_bp_save_avg', 0.5)
            }
            
            # Surfaces
            for f in feats:
                if 'Surface_' in f: input_row[f] = 1 if f == f'Surface_{surf}' else 0
            
            # Predecir
            try:
                # Filtramos el input_row para que solo tenga las columnas que el modelo espera
                # (Esto evita errores si el modelo pide menos cosas que las que calculamos)
                cols_modelo = [c for c in feats if c in input_row or 'Surface_' in c]
                X_in = pd.DataFrame([input_row])
                
                # Rellenamos con 0 cualquier columna de superficie que falte
                for col in feats:
                    if col not in X_in.columns:
                        X_in[col] = 0
                
                # Reordenamos columnas para que coincidan EXACTAMENTE con el entrenamiento
                X_in = X_in[feats]

                prob = model_win.predict_proba(X_in)[0][1]
                games = model_games.predict(X_in)[0]
                
                # --- DISPLAY VISUAL ---
                col1, col2, col3 = st.columns([1,2,1])
                col1.subheader(p1)
                col3.subheader(p2)
                
                # Probabilidad
                st.progress(prob, text=f"Probabilidad {p1}: {prob*100:.1f}%")
                
                st.info(f"🎾 Se estiman **{games:.1f} juegos** en el partido.")
                
                if prob > 0.65: st.success(f"💎 Pick Claro: **{p1}** (Cuota justa < {1/prob:.2f})")
                elif prob < 0.35: st.success(f"💎 Pick Claro: **{p2}** (Cuota justa < {1/(1-prob):.2f})")
                else: st.warning("⚠️ Partido reñido.")

            except Exception as e:
                st.error(f"Error al calcular: {e}")
                st.write("Posible causa: Los archivos .joblib no coinciden con el código actual.")
            
    else: st.error("Selecciona jugadores válidos")