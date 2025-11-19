import streamlit as st
import pandas as pd
import joblib
import numpy as np
from scipy.stats import norm

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="NeuralTennis Pro", page_icon="🎾", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .main-card { background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; margin-bottom: 20px; }
    .player-header { font-size: 20px; font-weight: bold; color: #f8fafc; text-align: center; }
    .player-sub { font-size: 12px; color: #94a3b8; text-align: center; }
    .vs { font-size: 18px; font-weight: 900; color: #64748b; text-align: center; padding: 10px 0; }
    .win-prob { font-size: 42px; font-weight: 800; text-align: center; color: #4ade80; margin: 10px 0; }
    .bet-section { background: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #475569; margin-top: 10px; }
    .val-good { color: #4ade80; font-weight: bold; }
    .val-bad { color: #f87171; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- CARGA ---
@st.cache_resource
def load_system():
    try:
        m_win = joblib.load('modelo_ganador.joblib')
        m_games = joblib.load('modelo_juegos.joblib')
        feats = joblib.load('features.joblib')
        db = joblib.load('database_reciente.joblib')
        std_g = joblib.load('std_juegos.joblib') 
        return m_win, m_games, feats, db, std_g
    except: return None, None, None, None, None

model_win, model_games, features, db, std_games = load_system()

if db is None:
    st.error("❌ Ejecuta 'entrenar_ia.py' de nuevo para generar los archivos necesarios.")
    st.stop()

def get_last_stats(player_name):
    p_data = db[db['player_name'] == player_name].sort_values(by='Date').tail(1)
    if p_data.empty: return None
    row = p_data.iloc[0]
    return {
        'name': player_name, 'rank': row['player_rank'], 'elo': row['player_elo'],
        'form': row['player_form_last_5'], 'surf_win': row['player_surf_win'],
        'ace': row['player_ace_avg'], 'saved': row['player_bp_save_avg'], 'rest': 7
    }

# --- UI ---
st.title("🎾 NeuralTennis Pro")
all_players = sorted(db['player_name'].unique())

# === GESTIÓN DE ESTADO (LA SOLUCIÓN) ===
if 'analisis_activo' not in st.session_state:
    st.session_state.analisis_activo = False

# Si cambiamos de jugadores, reseteamos el análisis para no mostrar datos antiguos
def resetear_analisis():
    st.session_state.analisis_activo = False

c1, c2 = st.columns(2)
# Añadimos on_change para resetear si cambias de jugador
with c1: p1 = st.selectbox("J1", all_players, index=0, on_change=resetear_analisis)
with c2: p2 = st.selectbox("J2", all_players, index=1, on_change=resetear_analisis)

col_conf1, col_conf2 = st.columns(2)
with col_conf1: surface = st.selectbox("Superficie", ["Hard", "Clay", "Grass"], on_change=resetear_analisis)
with col_conf2: best_of = st.selectbox("Sets", [3, 5], on_change=resetear_analisis)

# BOTÓN: Al hacer click, activamos el estado persistente
if st.button("🔮 Analizar Mercados", type="primary", use_container_width=True):
    st.session_state.analisis_activo = True

# LÓGICA PRINCIPAL (Ahora depende del estado, no del botón directamente)
if st.session_state.analisis_activo:
    if p1 == p2: st.warning("Elige jugadores distintos.")
    else:
        d1, d2 = get_last_stats(p1), get_last_stats(p2)
        if d1 and d2:
            # 1. Preparar Datos
            row = {
                'Best of': best_of,
                'diff_elo': d1['elo'] - d2['elo'], 'diff_rank': d2['rank'] - d1['rank'],
                'diff_form': d1['form'] - d2['form'], 'diff_surf': d1['surf_win'] - d2['surf_win'],
                'player_elo': d1['elo'], 'opponent_elo': d2['elo'],
                'player_surf_win': d1['surf_win'], 'opponent_surf_win': d2['surf_win'],
                'player_ace_avg': d1['ace'], 'opponent_ace_avg': d2['ace'],
                'days_rest': d1['rest'], 'opponent_rest': d2['rest'],
                'player_bp_save_avg': d1['saved'], 'opponent_bp_save_avg': d2['saved']
            }
            for f in features: 
                if 'Surface_' in f: row[f] = 1 if f == f'Surface_{surface}' else 0
            
            X_in = pd.DataFrame([row])
            for f in features:
                if f not in X_in.columns: X_in[f] = 0
            X_in = X_in[features]

            # 2. Predicciones
            prob_p1 = model_win.predict_proba(X_in)[0][1]
            pred_games = model_games.predict(X_in)[0]
            
            winner = p1 if prob_p1 >= 0.5 else p2
            prob_win = prob_p1 if prob_p1 >= 0.5 else 1 - prob_p1

            # --- VISUALIZACIÓN ---
            st.markdown(f"""
            <div class='main-card'>
                <div style='display:flex; justify-content:space-between;'>
                    <div style='text-align:center;'><div class='player-header'>{p1}</div><small>Elo {int(d1['elo'])}</small></div>
                    <div class='vs'>VS</div>
                    <div style='text-align:center;'><div class='player-header'>{p2}</div><small>Elo {int(d2['elo'])}</small></div>
                </div>
                <div style='text-align:center; margin-top:10px;'>
                    <div class='win-prob'>{winner} {prob_win*100:.1f}%</div>
                    <small>Juegos Estimados: <b>{pred_games:.1f}</b> (±{std_games:.1f})</small>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # === CALCULADORAS ===
            t1, t2, t3 = st.tabs(["🏆 Ganador", "🔢 Over/Under", "🏁 Hándicap"])

            with t1:
                st.markdown("<div class='bet-section'>", unsafe_allow_html=True)
                c_odd, c_res = st.columns([1, 2])
                with c_odd: odd_val = st.number_input(f"Cuota {winner}", value=1.50, step=0.05)
                with c_res:
                    implied = 1/odd_val
                    edge = prob_win - implied
                    st.write(f"Prob. Real: **{prob_win*100:.1f}%** | Prob. Casa: **{implied*100:.1f}%**")
                    if edge > 0.02: st.markdown(f"<div class='val-good'>✅ VALOR: +{edge*100:.1f}%</div>", unsafe_allow_html=True)
                    else: st.markdown(f"<div class='val-bad'>❌ SIN VALOR ({edge*100:.1f}%)</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with t2:
                st.markdown("<div class='bet-section'>", unsafe_allow_html=True)
                st.info(f"🧠 La IA predice **{pred_games:.1f}** juegos. Calculando probabilidades...")
                
                col_line, col_odd_ou = st.columns(2)
                # IMPORTANTE: Como estamos dentro de session_state, cambiar estos inputs NO cerrará la vista
                with col_line: line_ou = st.number_input("Línea Juegos (ej: 22.5)", value=float(int(pred_games)), step=0.5)
                with col_odd_ou: odd_ou = st.number_input("Cuota Over", value=1.85, step=0.05)
                
                z_score = (line_ou - pred_games) / std_games
                prob_under = norm.cdf(z_score)
                prob_over = 1 - prob_under
                
                col_res_over, col_res_under = st.columns(2)
                with col_res_over:
                    st.write(f"**OVER {line_ou}**")
                    st.write(f"Prob: **{prob_over*100:.1f}%**")
                    fair_odd_over = 1/prob_over if prob_over > 0 else 99
                    edge_over = prob_over - (1/odd_ou)
                    if edge_over > 0.02: st.markdown(f"<span class='val-good'>✅ VALOR (Cuota > {fair_odd_over:.2f})</span>", unsafe_allow_html=True)
                    else: st.markdown(f"<span class='val-bad'>❌ NO ENTRAR</span>", unsafe_allow_html=True)

                with col_res_under:
                    st.write(f"**UNDER {line_ou}**")
                    st.write(f"Prob: **{prob_under*100:.1f}%**")
                    fair_odd_under = 1/prob_under if prob_under > 0 else 99
                    st.write(f"Cuota justa: {fair_odd_under:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

            with t3:
                st.markdown("<div class='bet-section'>", unsafe_allow_html=True)
                expected_diff = (prob_p1 - 0.5) * 8 
                st.write(f"Diferencia esperada: **{expected_diff:.1f}** juegos para {p1}")
                h_line = st.number_input(f"Hándicap {p1} (ej: -3.5)", value=-2.5, step=0.5)
                dist = expected_diff - h_line
                prob_cover = 1 / (1 + np.exp(-0.5 * dist))
                st.write(f"Probabilidad de cubrir {h_line}: **{prob_cover*100:.1f}%**")
                st.markdown("</div>", unsafe_allow_html=True)

        else: st.error("Faltan datos.")