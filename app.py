import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="Tennis AI Pro", page_icon="🎾", layout="centered")

# --- CARGA DE DATOS ---
@st.cache_resource
def cargar_datos():
    if os.path.exists('features.joblib'):
        feats = joblib.load('features.joblib')
    elif os.path.exists('features_ganador.joblib'):
        feats = joblib.load('features_ganador.joblib')
    else: return None

    if not os.path.exists('modelo_ganador.joblib'): return None
    
    m_win = joblib.load('modelo_ganador.joblib')
    m_games = joblib.load('modelo_juegos.joblib')
    
    try:
        df = pd.read_csv("atp_matches_procesados.csv")
        # Parches
        if 'tourney_date' in df.columns: df.rename(columns={'tourney_date': 'Date'}, inplace=True)
        if 'player_name' not in df.columns and 'Player_1' in df.columns:
             df.rename(columns={'Player_1': 'player_name', 'Player_2': 'opponent_name'}, inplace=True)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')
        return m_win, m_games, feats, df
    except: return None

res = cargar_datos()
if not res:
    st.error("❌ Error crítico: Faltan archivos en GitHub.")
    st.stop()

model_win, model_games, feats, df = res

# --- MENÚ ---
st.sidebar.title("Menú")
modo = st.sidebar.radio("Ir a:", ["🔮 Predictor V4 (Robusto)", "🕵️‍♂️ Detective"])

# --- MODO DETECTIVE ---
if modo == "🕵️‍♂️ Detective":
    st.title("🕵️‍♂️ Detective")
    busqueda = st.text_input("Buscar jugador:")
    if busqueda:
        col_p1 = 'player_name' if 'player_name' in df.columns else 'Player_1'
        col_p2 = 'opponent_name' if 'opponent_name' in df.columns else 'Player_2'
        r1 = df[df[col_p1].astype(str).str.contains(busqueda, case=False, na=False)][col_p1].unique()
        r2 = df[df[col_p2].astype(str).str.contains(busqueda, case=False, na=False)][col_p2].unique()
        resultados = sorted(list(set(list(r1) + list(r2))))
        for r in resultados: st.code(r)

# --- MODO PREDICTOR V4 ---
elif modo == "🔮 Predictor V4 (Robusto)":
    st.title("🎾 Tennis AI Pro V4")
    st.caption("Sistema de Análisis de Perfil Histórico")
    
    # --- FUNCIÓN INTELIGENTE DE PERFIL ---
    def get_player_profile(player_name):
        # Buscamos todos los partidos del jugador
        col_p = 'player_name' if 'player_name' in df.columns else 'Player_1'
        matches = df[df[col_p] == player_name].copy()
        
        if matches.empty: return None
        
        # Cogemos los últimos 20 partidos
        last_matches = matches.tail(20)
        
        # Estadísticas clave
        stats_cols = ['player_ace_avg', 'player_1st_won_avg', 'player_bp_save_avg', 'player_form']
        
        profile = {}
        # Para cada estadística, calculamos la media IGNORANDO LOS CEROS
        for col in stats_cols:
            if col in last_matches.columns:
                # Filtramos valores válidos (> 0.01)
                valid_data = last_matches[last_matches[col] > 0.01][col]
                if not valid_data.empty:
                    profile[col] = valid_data.mean() # Media real de sus últimos partidos válidos
                else:
                    # Si no tiene NINGÚN dato válido en 20 partidos, usamos promedio ATP
                    defaults = {'player_ace_avg': 0.05, 'player_1st_won_avg': 0.68, 'player_bp_save_avg': 0.58, 'player_form': 0.5}
                    profile[col] = defaults.get(col, 0.5)
            else:
                profile[col] = 0.5
        
        # Ranking (usamos el del último partido sí o sí)
        profile['player_rank'] = last_matches.iloc[-1]['player_rank']
        
        # Superficie (Calculamos su winrate real en esta superficie)
        return profile

    # Inputs
    st.sidebar.header("1. Jugadores")
    col_p = 'player_name' if 'player_name' in df.columns else 'Player_1'
    players = sorted(df[col_p].unique())
    
    p1 = st.sidebar.selectbox("Jugador 1", players, index=None, placeholder="Buscar...")
    p2 = st.sidebar.selectbox("Jugador 2", players, index=None, placeholder="Buscar...")
    
    # Lógica de Ranking manual
    r1_def, r2_def = 500, 500
    s1, s2 = {}, {}
    
    if p1: 
        s1 = get_player_profile(p1)
        if s1: r1_def = int(s1['player_rank'])
    if p2: 
        s2 = get_player_profile(p2)
        if s2: r2_def = int(s2['player_rank'])

    st.sidebar.header("2. Ajustes")
    r1_in = st.sidebar.number_input(f"Ranking {p1 if p1 else 'J1'}", value=r1_def)
    r2_in = st.sidebar.number_input(f"Ranking {p2 if p2 else 'J2'}", value=r2_def)
    surf = st.sidebar.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    bo = st.sidebar.selectbox("Sets", [3, 5])

    if st.sidebar.button("⚡ Analizar"):
        if p1 and p2 and p1 != p2 and s1 and s2:
            # H2H Simple
            h2h = df[(df[col_p] == p1) & (df['opponent_name'] == p2)]
            h2h_w = h2h['result'].sum() if 'result' in df.columns else len(h2h)
            
            # Construir fila de predicción con DATOS PROMEDIADOS (NO DE LA ÚLTIMA FILA)
            row = {
                'player_rank': r1_in, 'opponent_rank': r2_in, 'Best of': bo,
                'player_form': s1['player_form'], 'opponent_form': s2['player_form'],
                'h2h_wins': h2h_w, 'h2h_total': len(h2h),
                
                # Aquí usamos los perfiles promediados que calculamos arriba
                'player_ace_avg': s1['player_ace_avg'], 'opponent_ace_avg': s2['player_ace_avg'],
                'player_1st_won_avg': s1['player_1st_won_avg'], 'opponent_1st_won_avg': s2['player_1st_won_avg'],
                'player_bp_save_avg': s1['player_bp_save_avg'], 'opponent_bp_save_avg': s2['player_bp_save_avg'],
                
                # Superficie (default 0.5 si no hay dato específico)
                'player_surf_win': 0.5, 'opponent_surf_win': 0.5 
            }
            
            # One Hot
            for f in feats: 
                if 'Surface_' in f: row[f] = 1 if f == f'Surface_{surf}' else 0
            
            try:
                # Crear DF para predicción
                X_in = pd.DataFrame([row])
                for c in feats:
                    if c not in X_in.columns: X_in[c] = 0
                X_in = X_in[feats]
                
                prob = model_win.predict_proba(X_in)[0][1]
                games = model_games.predict(X_in)[0]
                
                # --- RESULTADOS ---
                st.divider()
                c1, c2, c3 = st.columns([5,2,5])
                c1.markdown(f"### {p1}")
                c1.caption(f"Rank: {r1_in}")
                c3.markdown(f"### {p2}")
                c3.caption(f"Rank: {r2_in}")
                
                st.progress(prob, text=f"Probabilidad {p1}: {prob*100:.1f}%")
                
                if prob > 0.5: st.success(f"🏆 Favorito: **{p1}** (Cuota: {1/prob:.2f})")
                else: st.success(f"🏆 Favorito: **{p2}** (Cuota: {1/(1-prob):.2f})")
                
                st.markdown("### 📊 Estadísticas Reales (Media últimos partidos)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Forma", f"{s1['player_form']:.0%}", f"{(s1['player_form']-s2['player_form'])*100:.0f}%")
                m2.metric("Saque (1st Won)", f"{s1['player_1st_won_avg']:.0%}", f"{(s1['player_1st_won_avg']-s2['player_1st_won_avg'])*100:.0f}%")
                m3.metric("Mental (BP Saved)", f"{s1['player_bp_save_avg']:.0%}", f"{(s1['player_bp_save_avg']-s2['player_bp_save_avg'])*100:.0f}%")
                
                st.info(f"🎾 Juegos estimados: **{games:.1f}**")
                
            except Exception as e: st.error(f"Error de cálculo: {e}")
        else: st.error("Faltan datos de jugadores.")