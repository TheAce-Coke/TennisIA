import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Tennis AI Predictor", page_icon="🎾", layout="centered")

# --- TÍTULO ---
st.title("🎾 ATP Tennis AI Predictor")
st.write("Inteligencia Artificial entrenada para predecir ganadores y total de juegos.")

# --- CARGA DE MODELOS (CON CACHÉ) ---
@st.cache_resource
def cargar_modelos():
    archivos = ['modelo_ganador.joblib', 'features_ganador.joblib', 
                'modelo_juegos.joblib', 'features_juegos.joblib', 
                'atp_matches_procesados.csv']
    
    if not all(os.path.exists(f) for f in archivos):
        return None, None, None, None, None
    
    m_win = joblib.load('modelo_ganador.joblib')
    f_win = joblib.load('features_ganador.joblib')
    m_games = joblib.load('modelo_juegos.joblib')
    f_games = joblib.load('features_juegos.joblib')
    
    df = pd.read_csv("atp_matches_procesados.csv")
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    df = df.sort_values(by='tourney_date')
    
    return m_win, f_win, m_games, f_games, df

model_win, feat_win, model_games, feat_games, df = cargar_modelos()

if model_win is None:
    st.error("❌ No encuentro los archivos del modelo. Ejecuta 'actualizar_auto.py' primero.")
    st.stop()

# --- FUNCIONES AUXILIARES ---
@st.cache_data
def obtener_stats(jugador):
    datos = df[df['player_name'] == jugador]
    if len(datos) == 0: return None
    ultimo = datos.iloc[-1]
    return {
        'rank': ultimo['player_rank'],
        'form': ultimo['player_form'],
        'avg_games': ultimo['player_avg_games']
    }

@st.cache_data
def calcular_h2h(j1, j2):
    matches = df[(df['player_name'] == j1) & (df['opponent_name'] == j2)]
    return matches['result'].sum(), len(matches)

# --- INTERFAZ LATERAL (SIDEBAR) ---
st.sidebar.header("Configuración del Partido")

# Lista de jugadores para el buscador
todos_jugadores = pd.concat([df['player_name'], df['opponent_name']]).unique()
todos_jugadores.sort()

# --- ¡MEJORA APLICADA AQUÍ! ---
# Cambiamos el 'selectbox' para que parezca un buscador

j1 = st.sidebar.selectbox(
    "Buscar Jugador 1",
    todos_jugadores,
    placeholder="Escribe para buscar (ej: Alcaraz C.)",
    index=None  # Empieza vacío
)

j2 = st.sidebar.selectbox(
    "Buscar Jugador 2",
    todos_jugadores,
    placeholder="Escribe para buscar (ej: Sinner J.)",
    index=None  # Empieza vacío
)
# --- FIN DE LA MEJORA ---

surf = st.sidebar.selectbox("Superficie", ["Hard", "Clay", "Grass"])
btn_predecir = st.sidebar.button("🔮 Calcular Predicción")

# --- LÓGICA DE PREDICCIÓN ---
if btn_predecir:
    # Comprobamos que se han seleccionado ambos jugadores
    if not j1 or not j2:
        st.warning("⚠️ Debes seleccionar a ambos jugadores.")
    elif j1 == j2:
        st.warning("⚠️ ¡Elige dos jugadores distintos!")
    else:
        s1 = obtener_stats(j1)
        s2 = obtener_stats(j2)

        if not s1 or not s2:
            st.error("Faltan datos de alguno de los jugadores.")
        else:
            # 1. CALCULAR DATOS
            h2h_wins, h2h_total = calcular_h2h(j1, j2)
            
            # Preparar entrada Ganador
            input_win = {
                'player_rank': s1['rank'], 'opponent_rank': s2['rank'],
                'player_form': s1['form'], 'opponent_form': s2['form'],
                'h2h_wins': h2h_wins, 'h2h_total': h2h_total
            }
            for f in feat_win: 
                if f.startswith('surface_'): input_win[f] = 0
            if f'surface_{surf}' in input_win: input_win[f'surface_{surf}'] = 1
            
            df_win = pd.DataFrame([input_win]).reindex(columns=feat_win, fill_value=0)
            prob = model_win.predict_proba(df_win)[0][1]
            
            # Preparar entrada Juegos
            input_games = {
                'player_rank': s1['rank'], 'opponent_rank': s2['rank'],
                'player_avg_games': s1['avg_games'], 'opponent_avg_games': s2['avg_games']
            }
            for f in feat_games:
                if f.startswith('surface_'): input_games[f] = 0
            if f'surface_{surf}' in input_games: input_games[f'surface_{surf}'] = 1
            
            df_games = pd.DataFrame([input_games]).reindex(columns=feat_games, fill_value=0)
            pred_games = model_games.predict(df_games)[0]

            # --- MOSTRAR RESULTADOS VISUALES ---
            st.divider()
            
            # Columnas para stats
            col1, col2, col3 = st.columns(3)
            col1.metric(label=f"Ranking {j1}", value=int(s1['rank']))
            col2.metric(label="H2H Histórico", value=f"{h2h_wins} - {h2h_total - h2h_wins}")
            col3.metric(label=f"Ranking {j2}", value=int(s2['rank']))
            
            st.subheader(f"🏆 Probabilidad de Victoria")
            
            # Barra de progreso visual
            prob_display = float(prob)
            st.progress(prob_display, text=f"{prob*100:.1f}%")
            
            c_win1, c_win2 = st.columns(2)
            
            if prob > 0.5:
                c_win1.success(f"**{j1}**")
                c_win1.write(f"Probabilidad: **{prob*100:.1f}%**")
                c_win1.write(f"Cuota Justa: **{1/prob:.2f}**")
                
                c_win2.error(f"{j2}")
                c_win2.write(f"{((1-prob)*100):.1f}%")
            else:
                c_win1.error(f"{j1}")
                c_win1.write(f"{prob*100:.1f}%")
                
                c_win2.success(f"**{j2}**")
                c_win2.write(f"Probabilidad: **{((1-prob)*100):.1f}%**")
                c_win2.write(f"Cuota Justa: **{1/(1-prob):.2f}**")

            st.divider()
            st.subheader("🎾 Predicción de Juegos")
            st.info(f"Se estiman **{pred_games:.1f} juegos** en total.")
            st.caption(f"Línea recomendada: Over/Under {round(pred_games)} juegos")

else:
    st.info("👈 Selecciona los jugadores en el menú de la izquierda y pulsa 'Calcular'.")