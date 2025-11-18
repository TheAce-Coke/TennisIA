import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tennis AI", page_icon="🎾", layout="centered")

# --- CARGA DE DATOS ---
@st.cache_resource
def cargar_datos():
    archivos = ['modelo_ganador.joblib', 'features_ganador.joblib', 
                'modelo_juegos.joblib', 'features_juegos.joblib', 
                'atp_matches_procesados.csv']
    
    if not all(os.path.exists(f) for f in archivos): return None, None, None, None, None
    
    m_win = joblib.load('modelo_ganador.joblib')
    f_win = joblib.load('features_ganador.joblib')
    m_games = joblib.load('modelo_juegos.joblib')
    f_games = joblib.load('features_juegos.joblib')
    
    df = pd.read_csv("atp_matches_procesados.csv")
    # Optimizamos fechas y texto
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    df = df.sort_values(by='tourney_date')
    
    return m_win, f_win, m_games, f_games, df

model_win, feat_win, model_games, feat_games, df = cargar_datos()

if model_win is None:
    st.error("❌ Error: Faltan archivos. Ejecuta el actualizador en tu PC y sube los archivos a GitHub.")
    st.stop()

# --- MENÚ DE NAVEGACIÓN ---
st.sidebar.title("Menú Principal")
modo = st.sidebar.radio("Selecciona una herramienta:", ["🔮 Predictor de Partidos", "🕵️‍♂️ Detective de Nombres"])

# ==============================================================================
# MODO 1: DETECTIVE DE NOMBRES (Buscador)
# ==============================================================================
if modo == "🕵️‍♂️ Detective de Nombres":
    st.title("🕵️‍♂️ Detective de Jugadores")
    st.info("Usa esta herramienta si no encuentras a un jugador en el desplegable. Te dirá cómo está escrito en la base de datos.")
    
    busqueda = st.text_input("Escribe parte del nombre (ej: Gima, Vallejo, Nadal):")
    
    if busqueda:
        # Buscamos en ambas columnas ignorando mayúsculas/minúsculas
        res1 = df[df['player_name'].str.contains(busqueda, case=False, na=False)]['player_name'].unique()
        res2 = df[df['opponent_name'].str.contains(busqueda, case=False, na=False)]['opponent_name'].unique()
        
        todos_encontrados = sorted(list(set(list(res1) + list(res2))))
        
        if todos_encontrados:
            st.success(f"✅ He encontrado {len(todos_encontrados)} coincidencias:")
            for nombre in todos_encontrados:
                st.code(nombre) # Lo ponemos en formato código para copiar fácil
            st.caption("Copia el nombre exacto y úsalo en el Predictor.")
        else:
            st.warning(f"❌ No he encontrado nada que contenga '{busqueda}'.")
            st.write("Posibles razones:")
            st.write("- El jugador es muy nuevo y aún no está en la base de datos TML.")
            st.write("- El apellido se escribe diferente (prueba con menos letras).")

# ==============================================================================
# MODO 2: PREDICTOR (La IA)
# ==============================================================================
elif modo == "🔮 Predictor de Partidos":
    st.title("🎾 ATP Tennis AI Predictor")
    
    # Funciones del predictor
    def obtener_stats(jugador):
        datos = df[df['player_name'] == jugador]
        if len(datos) == 0: return None
        ultimo = datos.iloc[-1]
        return {'rank': ultimo['player_rank'], 'form': ultimo['player_form'], 'avg_games': ultimo['player_avg_games']}

    def calcular_h2h(j1, j2):
        matches = df[(df['player_name'] == j1) & (df['opponent_name'] == j2)]
        return matches['result'].sum(), len(matches)

    # Inputs
    st.sidebar.header("Configuración")
    todos_jugadores = pd.concat([df['player_name'], df['opponent_name']]).unique()
    todos_jugadores.sort()

    j1 = st.sidebar.selectbox("Jugador 1", todos_jugadores, index=None, placeholder="Escribe para buscar...")
    j2 = st.sidebar.selectbox("Jugador 2", todos_jugadores, index=None, placeholder="Escribe para buscar...")
    surf = st.sidebar.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    best_of = st.sidebar.selectbox("Sets (Best of)", [3, 5])

    if st.sidebar.button("🔮 Calcular"):
        if not j1 or not j2:
            st.error("Selecciona dos jugadores.")
        elif j1 == j2:
            st.error("Elige jugadores distintos.")
        else:
            s1 = obtener_stats(j1)
            s2 = obtener_stats(j2)
            
            if s1 and s2:
                h2h_wins, h2h_total = calcular_h2h(j1, j2)
                
                # Predicción Ganador
                input_win = {
                    'best_of': best_of, 'player_rank': s1['rank'], 'opponent_rank': s2['rank'],
                    'player_form': s1['form'], 'opponent_form': s2['form'],
                    'h2h_wins': h2h_wins, 'h2h_total': h2h_total
                }
                for f in feat_win: 
                    if f.startswith('surface_'): input_win[f] = 0
                if f'surface_{surf}' in input_win: input_win[f'surface_{surf}'] = 1
                
                df_win = pd.DataFrame([input_win]).reindex(columns=feat_win, fill_value=0)
                prob = model_win.predict_proba(df_win)[0][1]
                
                # Predicción Juegos
                input_games = {
                    'best_of': best_of, 'player_rank': s1['rank'], 'opponent_rank': s2['rank'],
                    'player_avg_games': s1['avg_games'], 'opponent_avg_games': s2['avg_games']
                }
                for f in feat_games:
                    if f.startswith('surface_'): input_games[f] = 0
                if f'surface_{surf}' in input_games: input_games[f'surface_{surf}'] = 1
                
                df_games = pd.DataFrame([input_games]).reindex(columns=feat_games, fill_value=0)
                pred_games = model_games.predict(df_games)[0]

                # Resultados
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Rank J1", int(s1['rank']))
                c2.metric("H2H", f"{h2h_wins} - {h2h_total-h2h_wins}")
                c3.metric("Rank J2", int(s2['rank']))
                
                st.subheader("🏆 Ganador Estimado")
                st.progress(float(prob), text=f"Probabilidad {j1}: {prob*100:.1f}%")
                
                cw1, cw2 = st.columns(2)
                if prob > 0.5:
                    cw1.success(f"**{j1}**")
                    cw1.write(f"Cuota Justa: **{1/prob:.2f}**")
                    cw2.write(f"{j2}: {((1-prob)*100):.1f}%")
                else:
                    cw1.write(f"{j1}: {prob*100:.1f}%")
                    cw2.success(f"**{j2}**")
                    cw2.write(f"Cuota Justa: **{1/(1-prob):.2f}**")
                
                st.divider()
                st.subheader("🎾 Total de Juegos")
                st.info(f"Estimación: **{pred_games:.1f}** juegos")