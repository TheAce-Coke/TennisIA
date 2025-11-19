import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random

# --- CONFIGURACIÓN PRO ---
st.set_page_config(page_title="QuantTennis AI", page_icon="🎾", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card { background: #1e2329; padding: 15px; border-radius: 10px; border: 1px solid #2d333b; text-align: center; }
    .big-number { font-size: 24px; font-weight: bold; color: #58a6ff; }
    .sub-text { font-size: 12px; color: #8b949e; }
    .win-green { color: #3fb950; font-weight: bold; }
    .loss-red { color: #f85149; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- CARGA ---
@st.cache_resource
def load_data():
    try:
        model = joblib.load('modelo_calibrado.joblib')
        feats = joblib.load('features.joblib')
        db = joblib.load('db_players.joblib')
        return model, feats, db
    except: return None, None, None

model, features, db = load_data()

if not model:
    st.error("❌ Error: Ejecuta entrenar_ia.py primero.")
    st.stop()

# --- MOTOR DE SIMULACIÓN MONTE CARLO (TENIS REAL) ---
def sim_game(p_serve):
    # Simula un juego de tenis simple basado en prob de ganar punto al saque
    pts_server, pts_receiver = 0, 0
    while True:
        if random.random() < p_serve: pts_server += 1
        else: pts_receiver += 1
        
        if pts_server >= 4 and pts_server >= pts_receiver + 2: return 1, pts_server + pts_receiver # Gana server
        if pts_receiver >= 4 and pts_receiver >= pts_server + 2: return 0, pts_server + pts_receiver # Gana receiver

def sim_set(p1_serve_prob, p2_serve_prob):
    g1, g2 = 0, 0
    total_pts = 0
    # Tiebreak a los 6-6
    while g1 < 6 and g2 < 6:
        # Turno saque P1
        w, pts = sim_game(p1_serve_prob)
        if w: g1 += 1
        total_pts += pts
        if g1 == 6 and g2 < 5: return g1, g2, total_pts
        
        # Turno saque P2
        w, pts = sim_game(p2_serve_prob)
        if not w: g2 += 1 # Gana P2
        else: g1 += 1 # Break de P1 (gana receiver)
        total_pts += pts
        if g2 == 6 and g1 < 5: return g1, g2, total_pts

    # Tiebreak logic simplificada (quien llegue a 7 con dif de 2)
    # Asumimos prob promedio en tiebreak
    tb1, tb2 = 0, 0
    while True:
        if random.random() < 0.5: tb1 += 1 # Simplificado 50/50 en TB ajustado por skill
        else: tb2 += 1
        if (tb1 >= 7 and tb1 >= tb2+2): return 7, 6, total_pts + tb1 + tb2
        if (tb2 >= 7 and tb2 >= tb1+2): return 6, 7, total_pts + tb1 + tb2

def monte_carlo_match(p1_serve, p2_serve, best_of=3, sims=2000):
    results = []
    
    for _ in range(sims):
        sets_p1, sets_p2 = 0, 0
        games_p1, games_p2 = 0, 0
        match_games = 0
        
        needed = 2 if best_of == 3 else 3
        
        while sets_p1 < needed and sets_p2 < needed:
            s1, s2, pts = sim_set(p1_serve, p2_serve)
            games_p1 += s1
            games_p2 += s2
            match_games += (s1 + s2)
            
            if s1 > s2: sets_p1 += 1
            else: sets_p2 += 1
            
        results.append({
            'winner': 1 if sets_p1 > sets_p2 else 2,
            'total_games': match_games,
            'diff_games': games_p1 - games_p2, # Para handicap
            'score': f"{sets_p1}-{sets_p2}"
        })
        
    return pd.DataFrame(results)

# --- UI ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Configuración")
    players = sorted(db['player_name'].unique())
    p1_name = st.selectbox("Jugador 1", players, index=0)
    p2_name = st.selectbox("Jugador 2", players, index=1)
    surface = st.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    best_of = st.selectbox("Sets", [3, 5])
    
    analyze = st.button("🚀 EJECUTAR SIMULACIÓN", type="primary", use_container_width=True)

with col2:
    st.title("🎾 QuantTennis: Monte Carlo Engine")
    
    if analyze and p1_name != p2_name:
        # 1. Obtener Datos
        d1 = db[db['player_name'] == p1_name].iloc[0]
        d2 = db[db['player_name'] == p2_name].iloc[0]
        
        # 2. Feature Vector para el modelo base
        # (Necesitamos saber la probabilidad base de victoria para derivar el saque)
        row = {
            'Best of': best_of,
            'delta_elo': d1['player_elo'] - d2['player_elo'],
            'delta_form': d1['ewma_form'] - d2['ewma_form'],
            'delta_serve': d1['ewma_serve'] - d2['ewma_serve'],
            'delta_surf': d1['ewma_surface'] - d2['ewma_surface'],
            'player_elo': d1['player_elo'], 'opponent_elo': d2['player_elo'],
            'ewma_serve': d1['ewma_serve'], 'opp_ewma_serve': d2['ewma_serve'],
            'days_rest': 10, 'opp_days_rest': 10 # Default
        }
        for f in features: 
            if 'Surface_' in f: row[f] = 1 if f == f'Surface_{surface}' else 0
            
        X = pd.DataFrame([row])[features]
        
        # 3. Probabilidad General (Head-to-Head)
        prob_win_p1 = model.predict_proba(X)[0][1]
        
        # 4. INFERENCIA DE PROBABILIDAD DE SAQUE (Ingeniería Inversa)
        # Si P1 es muy favorito, su prob de saque será mayor que su media habitual
        # Ajustamos la media histórica del jugador según la calidad del rival
        base_serve_p1 = d1['ewma_serve'] # Ej: 0.65
        base_serve_p2 = d2['ewma_serve'] # Ej: 0.62
        
        # Factor de ajuste basado en la predicción del modelo ML
        # Si ML dice 80% victoria, subimos la prob de saque de P1
        bias = (prob_win_p1 - 0.50) * 0.15 # Ajuste fino
        
        sim_p1_serve = np.clip(base_serve_p1 + bias, 0.50, 0.85)
        sim_p2_serve = np.clip(base_serve_p2 - bias, 0.50, 0.85)
        
        st.write(f"**Parámetros de Simulación:** P1 Serve Win: {sim_p1_serve:.1%} | P2 Serve Win: {sim_p2_serve:.1%}")
        
        # 5. MONTE CARLO
        with st.spinner("🎲 Simulando 2,000 partidos punto a punto..."):
            sim_results = monte_carlo_match(sim_p1_serve, sim_p2_serve, best_of)
            
        # --- RESULTADOS ---
        win_pct = sim_results['winner'].value_counts(normalize=True).get(1, 0)
        avg_games = sim_results['total_games'].mean()
        
        # Header
        c_head1, c_head2, c_head3 = st.columns(3)
        c_head1.markdown(f"<div class='metric-card'><div class='sub-text'>{p1_name} Win%</div><div class='big-number'>{win_pct:.1%}</div></div>", unsafe_allow_html=True)
        c_head2.markdown(f"<div class='metric-card'><div class='sub-text'>Total Juegos (Media)</div><div class='big-number'>{avg_games:.1f}</div></div>", unsafe_allow_html=True)
        c_head3.markdown(f"<div class='metric-card'><div class='sub-text'>Cuota Justa</div><div class='big-number'>{1/win_pct if win_pct>0 else 0:.2f}</div></div>", unsafe_allow_html=True)
        
        # TABS DE APUESTAS
        tab_ou, tab_handicap, tab_sets = st.tabs(["🔢 Over/Under", "🏁 Hándicap", "📊 Marcador Exacto"])
        
        with tab_ou:
            st.write("#### Probabilidades Reales (Basadas en 2,000 simulaciones)")
            lines = range(18, 30) if best_of == 3 else range(30, 45)
            data_ou = []
            for line in lines:
                prob_over = (sim_results['total_games'] > line).mean()
                prob_under = 1 - prob_over
                # Solo mostramos líneas competitivas (entre 20% y 80%)
                if 0.15 < prob_over < 0.85:
                    data_ou.append({
                        "Línea": line,
                        "Over %": f"{prob_over:.1%}",
                        "Cuota Over": f"{1/prob_over:.2f}",
                        "Under %": f"{prob_under:.1%}",
                        "Cuota Under": f"{1/prob_under:.2f}"
                    })
            st.dataframe(pd.DataFrame(data_ou), hide_index=True, use_container_width=True)
            
        with tab_handicap:
            st.write(f"#### Hándicap {p1_name} (Juegos)")
            h_lines = [-5.5, -4.5, -3.5, -2.5, -1.5, 1.5, 2.5, 3.5, 4.5]
            data_h = []
            for h in h_lines:
                # Cubre hándicap si (JuegosP1 - JuegosP2) > Handicap
                # Ej: P1 gana por 4 juegos. Hándicap -3.5. 4 > 3.5 -> Gana
                # Ojo: En handicap negativo, necesitamos ganar por MÁS de eso.
                # Hándicap -3.5 significa diff_games > 3.5
                prob_cover = (sim_results['diff_games'] + h > 0).mean() 
                
                if 0.10 < prob_cover < 0.90:
                    data_h.append({
                        "Hándicap": h,
                        "Probabilidad": f"{prob_cover:.1%}",
                        "Cuota Justa": f"{1/prob_cover:.2f}"
                    })
            st.dataframe(pd.DataFrame(data_h), hide_index=True, use_container_width=True)
            
        with tab_sets:
            st.write("#### Marcador de Sets")
            counts = sim_results['score'].value_counts(normalize=True)
            st.bar_chart(counts)
            st.write(counts)

    elif analyze:
        st.warning("Selecciona dos jugadores distintos.")