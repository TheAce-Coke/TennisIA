import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="NeuralTennis AI", page_icon="🎾", layout="wide")

# --- ESTILOS CSS AVANZADOS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .player-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }
    .player-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .player-name { font-size: 22px; font-weight: 700; color: #ffffff; margin-bottom: 5px; }
    .player-rank {
        font-size: 14px; color: #a0a0a0; text-transform: uppercase; letter-spacing: 1px;
        background: rgba(0,0,0,0.3); padding: 4px 10px; border-radius: 20px; display: inline-block;
    }
    .vs-badge {
        font-size: 24px; font-weight: 900; color: #ff4b4b; text-align: center;
        margin-top: 40px; text-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
    }
    .prob-box {
        text-align: center; padding: 20px; margin-top: 20px;
        background: linear-gradient(180deg, rgba(30,30,30,0) 0%, rgba(30,30,30,0.5) 100%);
        border-radius: 15px;
    }
    .prob-percent { font-size: 48px; font-weight: 800; font-family: 'Helvetica Neue', sans-serif; }
    .winner-green { color: #00cc66; text-shadow: 0 0 15px rgba(0, 204, 102, 0.4); }
    .loser-red { color: #ff4b4b; opacity: 0.7; }
    .stat-row { display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 14px; color: #ccc; }
</style>
""", unsafe_allow_html=True)

# --- CARGA ROBUSTA ---
@st.cache_resource
def cargar_cerebro():
    try:
        if os.path.exists('features.joblib'): feats = joblib.load('features.joblib')
        elif os.path.exists('features_ganador.joblib'): feats = joblib.load('features_ganador.joblib')
        else: return None

        if not os.path.exists('modelo_ganador.joblib'): return None
        m_win = joblib.load('modelo_ganador.joblib')
        m_games = joblib.load('modelo_juegos.joblib')
        
        df = pd.read_csv("atp_matches_procesados.csv")
        
        # --- PARCHES DE COMPATIBILIDAD (Aquí estaba el fallo) ---
        # 1. Fecha
        if 'tourney_date' in df.columns: df.rename(columns={'tourney_date': 'Date'}, inplace=True)
        # 2. Nombres
        if 'player_name' not in df.columns and 'Player_1' in df.columns:
             df.rename(columns={'Player_1': 'player_name', 'Player_2': 'opponent_name'}, inplace=True)
        # 3. SUPERFICIE (¡Corrección Nueva!)
        if 'surface' in df.columns: df.rename(columns={'surface': 'Surface'}, inplace=True)
        # -------------------------------------------------------

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')
        return m_win, m_games, feats, df
    except: return None

sistema = cargar_cerebro()
if not sistema:
    st.error("⚠️ Error de sistema: Faltan archivos en el repositorio.")
    st.stop()

model_win, model_games, feats, df = sistema

# --- LÓGICA DE PERFILADO ---
def get_smart_profile(player_name, surface_filter):
    col_p = 'player_name' if 'player_name' in df.columns else 'Player_1'
    matches = df[df[col_p] == player_name].copy()
    if matches.empty: return None
    
    last_15 = matches.tail(15)
    
    stats = {}
    stats['player_rank'] = last_15.iloc[-1]['player_rank']
    if pd.isna(stats['player_rank']): stats['player_rank'] = 500
    
    cols_tech = ['player_ace_avg', 'player_1st_won_avg', 'player_bp_save_avg', 'player_form']
    for c in cols_tech:
        if c in last_15.columns:
            val = last_15[last_15[c] > 0.01][c].mean()
            stats[c] = val if not pd.isna(val) else 0
        else: stats[c] = 0

    # Corrección superficie segura
    if 'Surface' in matches.columns:
        surf_matches = matches[matches['Surface'] == surface_filter]
        if not surf_matches.empty and len(surf_matches) > 2:
            stats['player_surf_win'] = surf_matches.tail(10)['result'].mean()
        else:
            stats['player_surf_win'] = stats['player_form'] * 0.85
    else:
        stats['player_surf_win'] = 0.5 # Default si falla columna

    return stats

def imputar_logica_ranking(stats):
    r = stats['player_rank']
    if r <= 30:   base = {'1st': 0.75, 'bp': 0.64, 'ace': 0.09, 'form': 0.65}
    elif r <= 100: base = {'1st': 0.70, 'bp': 0.60, 'ace': 0.06, 'form': 0.55}
    else:          base = {'1st': 0.64, 'bp': 0.55, 'ace': 0.04, 'form': 0.48}

    if stats['player_1st_won_avg'] < 0.4: stats['player_1st_won_avg'] = base['1st']
    if stats['player_bp_save_avg'] < 0.3: stats['player_bp_save_avg'] = base['bp']
    if stats['player_ace_avg'] == 0:      stats['player_ace_avg'] = base['ace']
    if stats['player_form'] == 0:         stats['player_form'] = base['form']
    return stats

# --- UI PRINCIPAL ---
with st.sidebar:
    st.header("🎛️ Centro de Control")
    col_p = 'player_name' if 'player_name' in df.columns else 'Player_1'
    all_p = sorted(list(set(df[col_p].unique())))
    
    p1 = st.selectbox("Jugador 1", all_p, index=None, placeholder="Seleccionar...")
    p2 = st.selectbox("Jugador 2", all_p, index=None, placeholder="Seleccionar...")
    st.markdown("---")
    surf = st.selectbox("Pista", ["Hard", "Clay", "Grass"])
    bo = st.selectbox("Sets", [3, 5])
    
    r1_val, r2_val = 500, 500
    if p1 and p2:
        s1_pre = get_smart_profile(p1, surf)
        s2_pre = get_smart_profile(p2, surf)
        if s1_pre: r1_val = int(s1_pre['player_rank'])
        if s2_pre: r2_val = int(s2_pre['player_rank'])

    rank1 = st.number_input("Rank J1", value=r1_val)
    rank2 = st.number_input("Rank J2", value=r2_val)
    analizar = st.button("🚀 ANALIZAR", type="primary", use_container_width=True)

# --- PANTALLA PRINCIPAL ---
st.markdown("<h1 style='text-align: center; margin-bottom: 40px;'>🎾 NEURAL TENNIS <span style='color:#58a6ff; font-size: 20px;'>v6.1</span></h1>", unsafe_allow_html=True)

if analizar and p1 and p2 and p1 != p2:
    prof1 = get_smart_profile(p1, surf)
    prof2 = get_smart_profile(p2, surf)
    
    if prof1 and prof2:
        prof1['player_rank'] = rank1
        prof2['player_rank'] = rank2
        prof1 = imputar_logica_ranking(prof1)
        prof2 = imputar_logica_ranking(prof2)
        
        h2h_df = df[(df[col_p] == p1) & (df['opponent_name' if 'opponent_name' in df.columns else 'Player_2'] == p2)]
        h2h_w = len(h2h_df[h2h_df['result']==1]) if 'result' in h2h_df.columns else len(h2h_df)
        
        row = {
            'player_rank': rank1, 'opponent_rank': rank2, 'Best of': bo,
            'player_form': prof1['player_form'], 'opponent_form': prof2['player_form'],
            'h2h_wins': h2h_w, 'h2h_total': len(h2h_df),
            'player_surf_win': prof1['player_surf_win'], 'opponent_surf_win': prof2['player_surf_win'],
            'player_ace_avg': prof1['player_ace_avg'], 'opponent_ace_avg': prof2['player_ace_avg'],
            'player_1st_won_avg': prof1['player_1st_won_avg'], 'opponent_1st_won_avg': prof2['player_1st_won_avg'],
            'player_bp_save_avg': prof1['player_bp_save_avg'], 'opponent_bp_save_avg': prof2['player_bp_save_avg']
        }
        
        for f in feats: 
            if 'Surface_' in f: row[f] = 1 if f == f'Surface_{surf}' else 0
            
        X_in = pd.DataFrame([row])
        for c in feats:
            if c not in X_in.columns: X_in[c] = 0
        X_in = X_in[feats]

        prob_p1 = model_win.predict_proba(X_in)[0][1]
        games_pred = model_games.predict(X_in)[0]
        
        c1, c2, c3 = st.columns([4, 1, 4])
        with c1:
            st.markdown(f"<div class='player-card'><div class='player-name'>{p1}</div><div class='player-rank'>RANK #{rank1}</div></div>", unsafe_allow_html=True)
        with c2: st.markdown("<div class='vs-badge'>VS</div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='player-card'><div class='player-name'>{p2}</div><div class='player-rank'>RANK #{rank2}</div></div>", unsafe_allow_html=True)
            
        st.markdown("<div class='prob-box'>", unsafe_allow_html=True)
        if prob_p1 > 0.5:
            st.markdown(f"<div class='prob-percent winner-green'>{prob_p1*100:.1f}%</div><div>Victoria estimada: <b>{p1}</b></div>", unsafe_allow_html=True)
            if prob_p1 > 0.55: st.success(f"💎 PICK: {p1} (Cuota > {1/prob_p1:.2f})")
        else:
            st.markdown(f"<div class='prob-percent winner-green'>{((1-prob_p1)*100):.1f}%</div><div>Victoria estimada: <b>{p2}</b></div>", unsafe_allow_html=True)
            if prob_p1 < 0.45: st.success(f"💎 PICK: {p2} (Cuota > {1/(1-prob_p1):.2f})")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.write("### 📊 Estadísticas")
        def barra(l, v1, v2):
            col = "#58a6ff" if v1 >= v2 else "#8b949e"
            st.write(f"**{l}**")
            cols = st.columns([1, 4, 1])
            cols[0].write(f"{v1:.1%}")
            tot = v1+v2
            cols[1].progress(int((v1/tot)*100) if tot>0 else 50)
            cols[2].write(f"{v2:.1%}")

        barra("Saque (1st Won)", prof1['player_1st_won_avg'], prof2['player_1st_won_avg'])
        barra("Mental (BP Saved)", prof1['player_bp_save_avg'], prof2['player_bp_save_avg'])
        barra("Superficie", prof1['player_surf_win'], prof2['player_surf_win'])
        st.info(f"🎾 Duración: **{games_pred:.1f} juegos**")

    else: st.error("Datos insuficientes.")
elif not analizar:
    st.markdown("<div style='text-align:center; margin-top: 50px; color:#666;'>Selecciona jugadores para comenzar</div>", unsafe_allow_html=True)