import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="NeuralTennis Pro", page_icon="🎾", layout="centered")

# --- ESTILOS CSS PROFESIONALES ---
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .main-card {
        background: #1e293b; border-radius: 12px; padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155; margin-bottom: 20px;
    }
    .player-header { font-size: 20px; font-weight: bold; color: #f8fafc; text-align: center; }
    .player-sub { font-size: 12px; color: #94a3b8; text-align: center; margin-bottom: 10px; }
    .vs { font-size: 18px; font-weight: 900; color: #64748b; text-align: center; padding: 10px 0; }
    .stat-box { background: #0f172a; padding: 10px; border-radius: 8px; text-align: center; }
    .stat-val { font-size: 18px; font-weight: bold; color: #38bdf8; }
    .stat-label { font-size: 11px; color: #64748b; text-transform: uppercase; }
    .win-prob { font-size: 42px; font-weight: 800; text-align: center; color: #4ade80; margin: 10px 0; }
    .bet-box { background: #3b2a1e; border: 1px solid #7c2d12; padding: 15px; border-radius: 8px; margin-top: 15px; }
    .bet-title { color: #fdba74; font-weight: bold; font-size: 14px; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# --- CARGA ---
@st.cache_resource
def load_system():
    try:
        m_win = joblib.load('modelo_ganador.joblib')
        m_games = joblib.load('modelo_juegos.joblib')
        feats = joblib.load('features.joblib')
        # Base de datos reducida solo con lo último de cada jugador
        db = joblib.load('database_reciente.joblib')
        return m_win, m_games, feats, db
    except Exception as e:
        return None, None, None, None

model_win, model_games, features, db = load_system()

if db is None:
    st.error("❌ Error: Faltan los modelos. Ejecuta 'entrenar_ia.py' primero.")
    st.stop()

# --- LÓGICA DE EXTRACCIÓN DE PERFIL REAL ---
def get_last_stats(player_name):
    # Buscar el registro más reciente de este jugador en la base de datos histórica
    p_data = db[db['player_name'] == player_name].sort_values(by='Date').tail(1)
    
    if p_data.empty:
        return None
        
    row = p_data.iloc[0]
    return {
        'name': player_name,
        'rank': row['player_rank'],
        'elo': row['player_elo'],
        'form': row['player_form_last_5'],
        'surf_win': row['player_surf_win'],
        'ace': row['player_ace_avg'],
        'saved': row['player_bp_save_avg'],
        'rest': 7 # Default si no hay datos recientes
    }

# --- INTERFAZ ---
st.title("🎾 NeuralTennis Pro")
st.markdown("Inteligencia Artificial basada en Elo Dinámico y Performance Reciente.")

# Selectores
all_players = sorted(db['player_name'].unique())

c1, c2 = st.columns(2)
with c1: p1 = st.selectbox("Jugador 1", all_players, index=all_players.index("Alcaraz C.") if "Alcaraz C." in all_players else 0)
with c2: p2 = st.selectbox("Jugador 2", all_players, index=all_players.index("Sinner J.") if "Sinner J." in all_players else 1)

col_conf1, col_conf2 = st.columns(2)
with col_conf1: surface = st.selectbox("Superficie", ["Hard", "Clay", "Grass"])
with col_conf2: best_of = st.selectbox("Sets", [3, 5])

if st.button("🔮 Analizar Partido", type="primary", use_container_width=True):
    if p1 == p2:
        st.warning("Selecciona dos jugadores distintos.")
    else:
        d1 = get_last_stats(p1)
        d2 = get_last_stats(p2)
        
        if d1 and d2:
            # Preparar Input Vector
            row = {}
            # Features básicas
            row['Best of'] = best_of
            row['diff_elo'] = d1['elo'] - d2['elo']
            row['diff_rank'] = d2['rank'] - d1['rank']
            row['diff_form'] = d1['form'] - d2['form']
            row['diff_surf'] = d1['surf_win'] - d2['surf_win']
            
            row['player_elo'] = d1['elo']
            row['opponent_elo'] = d2['elo']
            row['player_surf_win'] = d1['surf_win']
            row['opponent_surf_win'] = d2['surf_win']
            row['player_ace_avg'] = d1['ace']
            row['opponent_ace_avg'] = d2['ace']
            row['days_rest'] = d1['rest']
            row['opponent_rest'] = d2['rest']
            row['player_bp_save_avg'] = d1['saved']
            row['opponent_bp_save_avg'] = d2['saved']
            
            # One Hot Surface
            for f in features:
                if 'Surface_' in f:
                    row[f] = 1 if f == f'Surface_{surface}' else 0

            # Crear DataFrame ordenado según el entrenamiento
            X_input = pd.DataFrame([row])
            for f in features:
                if f not in X_input.columns: X_input[f] = 0
            X_input = X_input[features]

            # Predicción
            prob_p1 = model_win.predict_proba(X_input)[0][1]
            pred_games = model_games.predict(X_input)[0]
            
            winner = p1 if prob_p1 >= 0.5 else p2
            prob_win = prob_p1 if prob_p1 >= 0.5 else 1 - prob_p1
            
            # --- VISUALIZACIÓN ---
            st.markdown(f"""
            <div class='main-card'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div style='text-align:center; width:40%;'>
                        <div class='player-header'>{p1}</div>
                        <div class='player-sub'>Elo {int(d1['elo'])} | Rank {int(d1['rank'])}</div>
                    </div>
                    <div class='vs'>VS</div>
                    <div style='text-align:center; width:40%;'>
                        <div class='player-header'>{p2}</div>
                        <div class='player-sub'>Elo {int(d2['elo'])} | Rank {int(d2['rank'])}</div>
                    </div>
                </div>
                <hr style='border-color: #334155; margin: 20px 0;'>
                <div style='text-align:center;'>
                    <div style='color:#94a3b8; font-size:14px;'>Probabilidad de Victoria</div>
                    <div class='win-prob'>{winner} {prob_win*100:.1f}%</div>
                    <div style='color:#cbd5e1; margin-top:5px;'>Cuota Real Justa: <b>{1/prob_win:.2f}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculadora de Valor
            with st.expander("💰 Calculadora de Valor (Apuestas)", expanded=True):
                col_odds, col_res = st.columns([1, 2])
                with col_odds:
                    odds_bookie = st.number_input(f"Cuota Casa para {winner}", value=1.85, step=0.05)
                with col_res:
                    implied_prob = 1 / odds_bookie
                    edge = prob_win - implied_prob
                    st.write(f"Prob. Casa: {implied_prob*100:.1f}% vs IA: {prob_win*100:.1f}%")
                    if edge > 0.03: # 3% de margen de seguridad
                        st.markdown(f"<div style='color:#4ade80; font-weight:bold;'>✅ VALOR DETECTADO (+{edge*100:.1f}%)</div>", unsafe_allow_html=True)
                        st.write("La cuota es más alta de lo que debería. Oportunidad matemática.")
                    else:
                        st.markdown(f"<div style='color:#f87171; font-weight:bold;'>❌ SIN VALOR ({edge*100:.1f}%)</div>", unsafe_allow_html=True)
                        st.write("La casa paga poco para el riesgo real.")

            # Estadísticas Comparativas
            st.write("### 📊 Comparativa Técnica")
            
            def stat_row(label, v1, v2, is_pct=True):
                fmt = "{:.1%}" if is_pct else "{:.0f}"
                c1, c2, c3 = st.columns([3, 6, 3])
                c1.markdown(f"<div style='text-align:right; font-weight:bold;'>{fmt.format(v1)}</div>", unsafe_allow_html=True)
                with c2:
                    # Normalizar para la barra (max relativo)
                    m = max(v1, v2) if max(v1, v2) > 0 else 1
                    v1_n = v1 / m
                    v2_n = v2 / m
                    st.progress(v1_n) # Streamlit progress es simple, solo izquierda a derecha
                    # Truco visual simple: Nombre del stat al centro
                    st.markdown(f"<div style='text-align:center; font-size:10px; margin-top:-15px; color:#ccc;'>{label}</div>", unsafe_allow_html=True)
                c3.markdown(f"<div style='text-align:left; font-weight:bold;'>{fmt.format(v2)}</div>", unsafe_allow_html=True)

            stat_row("Efectividad Superficie", d1['surf_win'], d2['surf_win'])
            stat_row("Forma Reciente (Win Rate)", d1['form'], d2['form'])
            stat_row("Presión Saque (Aces)", d1['ace'], d2['ace'])
            stat_row("Resistencia Mental (BP Saved)", d1['saved'], d2['saved'])
            
            st.info(f"🎾 **Juegos Totales Estimados:** {pred_games:.1f} juegos")
            
        else:
            st.error("Datos insuficientes para generar predicción fiable para estos jugadores.")