import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="NeuralTennis AI", page_icon="🎾", layout="wide")

# --- ESTILOS CSS AVANZADOS (GLASSMORPHISM & NEON) ---
st.markdown("""
<style>
    /* Fondo general */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Tarjetas de Jugadores */
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
    
    /* Textos */
    .player-name {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
    }
    .player-rank {
        font-size: 14px;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: rgba(0,0,0,0.3);
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
    }
    
    /* VS Badge */
    .vs-badge {
        font-size: 24px;
        font-weight: 900;
        color: #ff4b4b;
        text-align: center;
        margin-top: 40px;
        text-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
    }
    
    /* Probabilidad Gigante */
    .prob-box {
        text-align: center;
        padding: 20px;
        margin-top: 20px;
        background: linear-gradient(180deg, rgba(30,30,30,0) 0%, rgba(30,30,30,0.5) 100%);
        border-radius: 15px;
    }
    .prob-percent {
        font-size: 48px;
        font-weight: 800;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .winner-green { color: #00cc66; text-shadow: 0 0 15px rgba(0, 204, 102, 0.4); }
    .loser-red { color: #ff4b4b; opacity: 0.7; }
    
    /* Métricas */
    .stat-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
        font-size: 14px;
        color: #ccc;
    }
    .stat-bar-bg {
        width: 100%;
        height: 6px;
        background: #333;
        border-radius: 3px;
        overflow: hidden;
    }
    .stat-bar-fill {
        height: 100%;
        background: #58a6ff;
    }
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
        # Parches rápidos
        if 'tourney_date' in df.columns: df.rename(columns={'tourney_date': 'Date'}, inplace=True)
        if 'player_name' not in df.columns and 'Player_1' in df.columns:
             df.rename(columns={'Player_1': 'player_name', 'Player_2': 'opponent_name'}, inplace=True)
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

# --- LÓGICA DE PERFILADO (CORREGIDA) ---
def get_smart_profile(player_name, surface_filter):
    col_p = 'player_name' if 'player_name' in df.columns else 'Player_1'
    # 1. Buscamos historial general
    matches = df[df[col_p] == player_name].copy()
    if matches.empty: return None
    
    last_15 = matches.tail(15)
    
    stats = {}
    # Ranking actual
    stats['player_rank'] = last_15.iloc[-1]['player_rank']
    if pd.isna(stats['player_rank']): stats['player_rank'] = 500
    
    # Stats Generales
    cols_tech = ['player_ace_avg', 'player_1st_won_avg', 'player_bp_save_avg', 'player_form']
    for c in cols_tech:
        if c in last_15.columns:
            val = last_15[last_15[c] > 0.01][c].mean()
            stats[c] = val if not pd.isna(val) else 0
        else: stats[c] = 0

    # --- CORRECCIÓN SUPERFICIE (LA CLAVE) ---
    # Intentamos buscar datos en ESTA superficie
    surf_matches = matches[matches['Surface'] == surface_filter]
    
    if not surf_matches.empty and len(surf_matches) > 2:
        # Si tiene historia en esta superficie, la usamos
        stats['player_surf_win'] = surf_matches.tail(10)['result'].mean()
    else:
        # Si NO tiene historia (0 partidos en Hard), NO le ponemos 0.
        # Le ponemos su forma general * 0.85 (Penalización leve por inexperiencia, pero no fatal)
        stats['player_surf_win'] = stats['player_form'] * 0.85

    return stats

def imputar_logica_ranking(stats):
    """Si las stats son 0 (basura), las inventamos según el ranking."""
    r = stats['player_rank']
    
    # Valores Base según calidad del jugador
    if r <= 30:   base = {'1st': 0.75, 'bp': 0.64, 'ace': 0.09, 'form': 0.65}
    elif r <= 100: base = {'1st': 0.70, 'bp': 0.60, 'ace': 0.06, 'form': 0.55}
    else:          base = {'1st': 0.64, 'bp': 0.55, 'ace': 0.04, 'form': 0.48}

    if stats['player_1st_won_avg'] < 0.4: stats['player_1st_won_avg'] = base['1st']
    if stats['player_bp_save_avg'] < 0.3: stats['player_bp_save_avg'] = base['bp']
    if stats['player_ace_avg'] == 0:      stats['player_ace_avg'] = base['ace']
    # Si la forma es 0, la forzamos también
    if stats['player_form'] == 0:         stats['player_form'] = base['form']
    
    return stats

# --- UI PRINCIPAL ---

# Barra Lateral Minimalista
with st.sidebar:
    st.header("🎛️ Centro de Control")
    
    col_p = 'player_name' if 'player_name' in df.columns else 'Player_1'
    all_p = sorted(list(set(df[col_p].unique())))
    
    p1 = st.selectbox("Jugador 1", all_p, index=None, placeholder="Seleccionar...")
    p2 = st.selectbox("Jugador 2", all_p, index=None, placeholder="Seleccionar...")
    
    st.markdown("---")
    surf = st.selectbox("Pista", ["Hard", "Clay", "Grass"])
    bo = st.selectbox("Sets", [3, 5])
    
    # Auto-Ranking
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

st.markdown("<h1 style='text-align: center; margin-bottom: 40px;'>🎾 NEURAL TENNIS <span style='color:#58a6ff; font-size: 20px;'>v6.0</span></h1>", unsafe_allow_html=True)

if analizar and p1 and p2 and p1 != p2:
    
    # 1. PROCESAMIENTO DE DATOS
    prof1 = get_smart_profile(p1, surf)
    prof2 = get_smart_profile(p2, surf)
    
    if prof1 and prof2:
        prof1['player_rank'] = rank1
        prof2['player_rank'] = rank2
        
        # Imputación de emergencia (Anti-Ceros)
        prof1 = imputar_logica_ranking(prof1)
        prof2 = imputar_logica_ranking(prof2)
        
        # H2H
        h2h_df = df[(df[col_p] == p1) & (df['opponent_name' if 'opponent_name' in df.columns else 'Player_2'] == p2)]
        h2h_w = len(h2h_df[h2h_df['result']==1]) if 'result' in h2h_df.columns else len(h2h_df)
        
        # Vector
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
        # Rellenar columnas que falten con 0
        for c in feats:
            if c not in X_in.columns: X_in[c] = 0
        X_in = X_in[feats] # Ordenar

        # Predicción
        prob_p1 = model_win.predict_proba(X_in)[0][1]
        games_pred = model_games.predict(X_in)[0]
        
        # --- INTERFAZ GRÁFICA ---
        
        # Layout de Combate
        c1, c2, c3 = st.columns([4, 1, 4])
        
        with c1:
            st.markdown(f"""
            <div class="player-card">
                <div class="player-name">{p1}</div>
                <div class="player-rank">RANK #{rank1}</div>
                <div style="margin-top: 15px; font-size: 30px;">
                    {'🔥' if prof1['player_form'] > 0.6 else '😐'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown("<div class='vs-badge'>VS</div>", unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class="player-card">
                <div class="player-name">{p2}</div>
                <div class="player-rank">RANK #{rank2}</div>
                 <div style="margin-top: 15px; font-size: 30px;">
                    {'🔥' if prof2['player_form'] > 0.6 else '😐'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Caja de Predicción
        st.markdown("<div class='prob-box'>", unsafe_allow_html=True)
        if prob_p1 > 0.5:
            st.markdown(f"<div class='prob-percent winner-green'>{prob_p1*100:.1f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div>Victoria estimada para <b>{p1}</b></div>", unsafe_allow_html=True)
            st.success(f"💰 Cuota de Valor: > {1/prob_p1:.2f}")
        else:
            st.markdown(f"<div class='prob-percent winner-green'>{((1-prob_p1)*100):.1f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div>Victoria estimada para <b>{p2}</b></div>", unsafe_allow_html=True)
            st.success(f"💰 Cuota de Valor: > {1/(1-prob_p1):.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Estadísticas Comparadas (Barras visuales)
        st.write("### 📊 Estadísticas Clave")
        
        def barra_comparativa(label, val1, val2):
            # Normalizar para visualización
            total = val1 + val2
            if total == 0: pct1 = 50
            else: pct1 = (val1 / total) * 100
            
            color = "#58a6ff" if val1 >= val2 else "#8b949e"
            
            st.write(f"**{label}**")
            cols = st.columns([1, 4, 1])
            cols[0].write(f"{val1:.1%}")
            cols[1].progress(int(pct1))
            cols[2].write(f"{val2:.1%}")

        barra_comparativa("Efectividad Saque (1st Won)", prof1['player_1st_won_avg'], prof2['player_1st_won_avg'])
        barra_comparativa("Resistencia Mental (BP Saved)", prof1['player_bp_save_avg'], prof2['player_bp_save_avg'])
        barra_comparativa("Adaptación a Superficie", prof1['player_surf_win'], prof2['player_surf_win'])
        
        st.info(f"🎾 Duración Estimada: **{games_pred:.1f} juegos**")

    else:
        st.error("Datos insuficientes en el histórico para generar predicción fiable.")

elif not analizar:
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #666;">
        <h3>👈 Selecciona jugadores en el menú para comenzar</h3>
        <p>Sistema listo con 320.000+ partidos procesados.</p>
    </div>
    """, unsafe_allow_html=True)