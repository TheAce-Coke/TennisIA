import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Tennis AI Pro", page_icon="🎾", layout="centered")

@st.cache_resource
def cargar_todo():
    if not os.path.exists('modelo_ganador.joblib'): return None
    m_win = joblib.load('modelo_ganador.joblib')
    m_games = joblib.load('modelo_juegos.joblib')
    feats = joblib.load('features.joblib')
    df = pd.read_csv("atp_matches_procesados.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    return m_win, m_games, feats, df

res = cargar_todo()
if not res: st.stop()
model_win, model_games, feats, df = res

def get_last_stats(player):
    row = df[df['player_name'] == player]
    if row.empty: return None
    return row.iloc[-1]

st.title("🎾 Tennis AI Pro V3")
st.sidebar.header("Configuración")

players = sorted(list(set(df['player_name'].unique()) | set(df['opponent_name'].unique())))
p1 = st.sidebar.selectbox("Jugador 1", players, index=None, placeholder="Buscar...")
p2 = st.sidebar.selectbox("Jugador 2", players, index=None, placeholder="Buscar...")
surf = st.sidebar.selectbox("Superficie", ["Hard", "Clay", "Grass"])
bo = st.sidebar.selectbox("Sets", [3, 5])

if st.sidebar.button("⚡ Analizar"):
    if p1 and p2 and p1 != p2:
        s1, s2 = get_last_stats(p1), get_last_stats(p2)
        
        if s1 is not None and s2 is not None:
            # H2H
            h2h_data = df[(df['player_name'] == p1) & (df['opponent_name'] == p2)]
            h2h_w = h2h_data['result'].sum()
            h2h_t = len(h2h_data)
            
            # Preparar Input
            input_row = {
                'player_rank': s1['player_rank'], 'opponent_rank': s2['player_rank'], 'Best of': bo,
                'player_form': s1['player_form'], 'opponent_form': s2['player_form'],
                'player_surf_win': s1['player_surf_win'], 'opponent_surf_win': s2['player_surf_win'],
                'h2h_wins': h2h_w, 'h2h_total': h2h_t,
                'player_ace_avg': s1['player_ace_avg'], 'opponent_ace_avg': s2['player_ace_avg'],
                'player_1st_won_avg': s1['player_1st_won_avg'], 'opponent_1st_won_avg': s2['player_1st_won_avg'],
                'player_bp_save_avg': s1['player_bp_save_avg'], 'opponent_bp_save_avg': s2['player_bp_save_avg']
            }
            
            # Surfaces
            for f in feats:
                if 'Surface_' in f: input_row[f] = 1 if f == f'Surface_{surf}' else 0
            
            # Predecir
            X_in = pd.DataFrame([input_row])[feats]
            prob = model_win.predict_proba(X_in)[0][1]
            games = model_games.predict(X_in)[0]
            
            # --- DISPLAY VISUAL ---
            col1, col2, col3 = st.columns([1,2,1])
            col1.subheader(p1)
            col1.caption(f"Rank {int(s1['player_rank'])}")
            col3.subheader(p2)
            col3.caption(f"Rank {int(s2['player_rank'])}")
            
            # Probabilidad
            st.progress(prob, text=f"Probabilidad {p1}: {prob*100:.1f}%")
            
            # Métricas Clave (Comparativa)
            st.write("### 📊 Duelo de Estilos")
            m1, m2, m3 = st.columns(3)
            m1.metric("Forma Reciente", f"{s1['player_form']:.0%}", f"{s1['player_form']-s2['player_form']:.0%}")
            m2.metric("Efectividad Saque", f"{s1['player_1st_won_avg']:.0%}", f"{s1['player_1st_won_avg']-s2['player_1st_won_avg']:.0%}")
            m3.metric("Mental (BP Saved)", f"{s1['player_bp_save_avg']:.0%}", f"{s1['player_bp_save_avg']-s2['player_bp_save_avg']:.0%}")
            
            st.info(f"🎾 Se estiman **{games:.1f} juegos** en el partido.")
            
            if prob > 0.65: st.success(f"💎 Pick Claro: **{p1}** (Cuota justa < {1/prob:.2f})")
            elif prob < 0.35: st.success(f"💎 Pick Claro: **{p2}** (Cuota justa < {1/(1-prob):.2f})")
            else: st.warning("⚠️ Partido muy reñido. Evitar apostar al ganador.")
            
    else: st.error("Error en jugadores")