import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Tennis AI Pro", page_icon="🎾", layout="centered")

@st.cache_resource
def cargar_datos():
    if os.path.exists('features.joblib'):
        feats = joblib.load('features.joblib')
    elif os.path.exists('features_ganador.joblib'):
        feats = joblib.load('features_ganador.joblib')
    else:
        return None

    if not os.path.exists('modelo_ganador.joblib'): return None
    
    m_win = joblib.load('modelo_ganador.joblib')
    m_games = joblib.load('modelo_juegos.joblib')
    
    try:
        df = pd.read_csv("atp_matches_procesados.csv")
        if 'tourney_date' in df.columns: df.rename(columns={'tourney_date': 'Date'}, inplace=True)
        if 'player_name' not in df.columns and 'Player_1' in df.columns:
             df.rename(columns={'Player_1': 'player_name', 'Player_2': 'opponent_name'}, inplace=True)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')
        return m_win, m_games, feats, df
    except:
        return None

res = cargar_datos()

if not res:
    st.error("❌ Error: Faltan archivos.")
    st.stop()

model_win, model_games, feats, df = res

# --- MENÚ ---
st.sidebar.title("Menú")
modo = st.sidebar.radio("Ir a:", ["🔮 Predictor PRO", "🕵️‍♂️ Detective"])

# --- DETECTIVE ---
if modo == "🕵️‍♂️ Detective":
    st.title("🕵️‍♂️ Detective")
    busqueda = st.text_input("Buscar jugador:")
    if busqueda:
        col_p1 = 'player_name' if 'player_name' in df.columns else 'Player_1'
        col_p2 = 'opponent_name' if 'opponent_name' in df.columns else 'Player_2'
        r1 = df[df[col_p1].astype(str).str.contains(busqueda, case=False, na=False)][col_p1].unique()
        r2 = df[df[col_p2].astype(str).str.contains(busqueda, case=False, na=False)][col_p2].unique()
        resultados = sorted(list(set(list(r1) + list(r2))))
        if resultados:
            for r in resultados: st.code(r)
        else: st.warning("No encontrado.")

# --- PREDICTOR ---
elif modo == "🔮 Predictor PRO":
    st.title("🎾 Tennis AI Pro V3.3 (Anti-Ceros)")
    
    def get_stats(player):
        col_jugador = 'player_name' if 'player_name' in df.columns else 'Player_1'
        row = df[df[col_jugador] == player]
        if row.empty: return None
        return row.iloc[-1]
        
    # --- FUNCIÓN DE LIMPIEZA DE DATOS (NUEVA) ---
    def sanear_dato(valor, defecto):
        """Si el valor es 0.0, NaN o nulo, devuelve el defecto."""
        if pd.isna(valor) or valor == 0.0:
            return defecto
        return valor

    st.sidebar.header("1. Jugadores")
    col_p1 = 'player_name' if 'player_name' in df.columns else 'Player_1'
    col_p2 = 'opponent_name' if 'opponent_name' in df.columns else 'Player_2'
    players = sorted(list(set(df[col_p1].unique()) | set(df[col_p2].unique())))
    
    p1 = st.sidebar.selectbox("Jugador 1", players, index=None, placeholder="Buscar...")
    p2 = st.sidebar.selectbox("Jugador 2", players, index=None, placeholder="Buscar...")
    
    rank1_def, rank2_def = 500, 500
    s1, s2 = None, None
    if p1 and p2:
        s1 = get_stats(p1)
        s2 = get_stats(p2)
        if s1 is not None: rank1_def = int(s1.get('player_rank', 500))
        if s2 is not None: rank2_def = int(s2.get('player_rank', 500))

    st.sidebar.header("2. Ajustes")
    r1_in = st.sidebar.number_input(f"Ranking {p1 if p1 else 'J1'}", value=rank1_def)
    r2_in = st.sidebar.number_input(f"Ranking {p2 if p2 else 'J2'}", value=rank2_def)
    surf = st.sidebar.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    bo = st.sidebar.selectbox("Sets", [3, 5])
    
    if st.sidebar.button("⚡ Analizar"):
        if p1 and p2 and p1 != p2 and s1 is not None and s2 is not None:
            # H2H
            h2h_data = df[(df[col_p1] == p1) & (df[col_p2] == p2)]
            h2h_w = len(h2h_data)
            if 'result' in df.columns: h2h_w = h2h_data['result'].sum()
            
            # --- APLICAMOS EL SANITIZADOR AQUÍ ---
            # Si la IA ve un 0 en el saque, le ponemos un 65% (0.65) por defecto.
            # Si ve un 0 en Aces, le ponemos un 5% (0.05).
            
            row = {
                'player_rank': r1_in, 'opponent_rank': r2_in, 'Best of': bo,
                'player_form': sanear_dato(s1.get('player_form'), 0.5),
                'opponent_form': sanear_dato(s2.get('player_form'), 0.5),
                'h2h_wins': h2h_w, 'h2h_total': len(h2h_data),
                'player_surf_win': sanear_dato(s1.get('player_surf_win'), 0.5),
                'opponent_surf_win': sanear_dato(s2.get('player_surf_win'), 0.5),
                
                # Stats críticas (Aquí estaba el fallo de los ceros)
                'player_1st_won_avg': sanear_dato(s1.get('player_1st_won_avg'), 0.68), # 68% es la media ATP
                'opponent_1st_won_avg': sanear_dato(s2.get('player_1st_won_avg'), 0.68),
                
                'player_ace_avg': sanear_dato(s1.get('player_ace_avg'), 0.05),
                'opponent_ace_avg': sanear_dato(s2.get('player_ace_avg'), 0.05),
                
                'player_bp_save_avg': sanear_dato(s1.get('player_bp_save_avg'), 0.60),
                'opponent_bp_save_avg': sanear_dato(s2.get('player_bp_save_avg'), 0.60)
            }
            
            for f in feats:
                if 'Surface_' in f: row[f] = 1 if f == f'Surface_{surf}' else 0
            
            try:
                X_in = pd.DataFrame([row])
                for c in feats: 
                    if c not in X_in.columns: X_in[c] = 0
                X_in = X_in[feats]

                prob = model_win.predict_proba(X_in)[0][1]
                games = model_games.predict(X_in)[0]
                
                # Visualización
                st.divider()
                c1, c2, c3 = st.columns([5, 2, 5])
                c1.markdown(f"### {p1}")
                c1.caption(f"Rank: {r1_in}")
                c3.markdown(f"### {p2}")
                c3.caption(f"Rank: {r2_in}")
                
                st.progress(prob, text=f"Probabilidad {p1}: {prob*100:.1f}%")
                
                if prob > 0.5:
                    st.success(f"🏆 **{p1}** favorito (Cuota justa: {1/prob:.2f})")
                else:
                    st.success(f"🏆 **{p2}** favorito (Cuota justa: {1/(1-prob):.2f})")
                
                st.markdown("### 📊 Duelo de Estilos")
                m1, m2, m3 = st.columns(3)
                
                # Mostramos los datos YA SANEADOS para que veas que no son 0
                m1.metric("Racha", f"{row['player_form']:.0%}", f"{(row['player_form']-row['opponent_form'])*100:.0f}%")
                m2.metric("Efectividad Saque", f"{row['player_1st_won_avg']:.0%}", f"{(row['player_1st_won_avg']-row['opponent_1st_won_avg'])*100:.0f}%")
                m3.metric("Mental (BP)", f"{row['player_bp_save_avg']:.0%}", f"{(row['player_bp_save_avg']-row['opponent_bp_save_avg'])*100:.0f}%")

                st.divider()
                st.info(f"🎾 Se estiman **{games:.1f} juegos**.")
                
            except Exception as e: st.error(f"Error: {e}")
            
    else:
        st.info("Selecciona jugadores.")