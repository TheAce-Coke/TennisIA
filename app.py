import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(page_title="NeuralSports Quant", page_icon="🏆", layout="wide")

# --- CSS UNIFICADO Y MEJORADO ---
st.markdown("""
<style>
    /* Fondo Global */
    .stApp { background: radial-gradient(circle at 10% 20%, #0f172a 0%, #020617 90%); color: #e2e8f0; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0b1120; border-right: 1px solid #1e293b; }
    
    /* Tarjetas de Métricas */
    .metric-container {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-container:hover { transform: translateY(-2px); border-color: rgba(56, 189, 248, 0.3); }
    
    .metric-value { font-size: 26px; font-weight: 800; color: #38bdf8; }
    .metric-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-bottom: 5px; }
    
    /* Títulos */
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; background-color: #1e293b; border-radius: 8px; color: #cbd5e1; border: none;
    }
    .stTabs [aria-selected="true"] { background-color: #3b82f6 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- SELECTOR DE DEPORTE ---
with st.sidebar:
    st.title("NeuralSports AI")
    st.caption("Sistema de Predicción Monte Carlo")
    deporte = st.radio("Selecciona Deporte", ["🎾 Tenis ATP", "🏀 NBA Basket"], index=0)
    st.markdown("---")

# ==============================================================================
#                                   MÓDULO TENIS
# ==============================================================================
if deporte == "🎾 Tenis ATP":
    
    @st.cache_resource
    def load_tennis():
        try:
            m = joblib.load('modelo_calibrado.joblib')
            f = joblib.load('features.joblib')
            d = joblib.load('db_players.joblib')
            return m, f, d
        except: return None, None, None

    model, features, db = load_tennis()
    
    if db is None:
        st.error("⚠️ Faltan archivos de Tenis. Ejecuta 'actualizar_auto.py' primero.")
        st.stop()

    # FUNCIONES SIMULACIÓN TENIS
    def sim_point(prob): return 1 if random.random() < prob else 0
    
    def sim_game(prob):
        p1, p2 = 0, 0
        while True:
            if sim_point(prob): p1+=1 
            else: p2+=1
            if p1>=4 and p1>=p2+2: return 1, p1+p2
            if p2>=4 and p2>=p1+2: return 0, p1+p2
            
    def sim_set(p1_p, p2_p):
        g1, g2, pts = 0, 0, 0
        while g1<6 and g2<6:
            w, p = sim_game(p1_p); pts+=p
            if w: g1+=1
            w, p = sim_game(p2_p); pts+=p
            if not w: g2+=1
            else: g1+=1
            if g1==6 and g2<5: return g1, g2, pts
            if g2==6 and g1<5: return g1, g2, pts
        # Tiebreak
        tb1, tb2 = 0, 0
        while True:
            if (tb1+tb2)%2==0: 
                if sim_point(p1_p): tb1+=1
                else: tb2+=1
            else:
                if sim_point(p2_p): tb2+=1
                else: tb1+=1
            if tb1>=7 and tb1>=tb2+2: return 7, 6, pts+tb1+tb2
            if tb2>=7 and tb2>=tb1+2: return 6, 7, pts+tb1+tb2

    def run_monte_carlo_tennis(p1_prob, p2_prob, best_of, n=1500):
        res = []
        target = 2 if best_of==3 else 3
        for _ in range(n):
            s1, s2, tg, gp1, gp2 = 0, 0, 0, 0, 0
            while s1<target and s2<target:
                g1, g2, _ = sim_set(p1_prob, p2_prob)
                if g1>g2: s1+=1
                else: s2+=1
                tg += g1+g2; gp1+=g1; gp2+=g2
            res.append({'winner': 1 if s1>s2 else 2, 'total_games': tg, 'diff_games': gp1-gp2})
        return pd.DataFrame(res)

    # UI TENIS SIDEBAR
    players = sorted(db['player_name'].unique())
    
    # Índices por defecto inteligentes
    idx1 = players.index("Alcaraz C.") if "Alcaraz C." in players else 0
    idx2 = players.index("Sinner J.") if "Sinner J." in players else 1
    
    p1 = st.sidebar.selectbox("J1 (Servicio)", players, index=idx1)
    p2 = st.sidebar.selectbox("J2 (Resto)", players, index=idx2)
    surf = st.sidebar.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    bo = st.sidebar.radio("Sets", [3, 5], horizontal=True)
    
    analyze_btn = st.sidebar.button("⚡ EJECUTAR SIMULACIÓN", type="primary")

    if analyze_btn and p1 != p2:
        d1, d2 = db[db['player_name']==p1].iloc[0], db[db['player_name']==p2].iloc[0]
        
        # Lógica Quant (Log5 ajustado)
        tour_avg = 0.64
        adj = -0.04 if surf=='Clay' else (0.03 if surf=='Grass' else 0.01)
        
        # Fallback si faltan datos de resto
        ret1 = d1.get('ewma_return', 1-tour_avg)
        ret2 = d2.get('ewma_return', 1-tour_avg)
        
        sim_p1 = np.clip(d1['ewma_serve'] - (ret2 - (1-tour_avg)) + adj, 0.45, 0.85)
        sim_p2 = np.clip(d2['ewma_serve'] - (ret1 - (1-tour_avg)) + adj, 0.45, 0.85)
        
        with st.spinner(f"Simulando {p1} vs {p2}..."):
            sim_df = run_monte_carlo_tennis(sim_p1, sim_p2, bo)
            
        # --- RESULTADOS TENIS ---
        
        # 1. Determinar Ganador y Confianza
        p1_win_prob = sim_df['winner'].value_counts(normalize=True).get(1, 0)
        
        if p1_win_prob >= 0.5:
            pred_winner = p1
            final_prob = p1_win_prob
            loser_prob = 1 - p1_win_prob
            color_win = "#4ade80" # Verde
        else:
            pred_winner = p2
            final_prob = 1 - p1_win_prob
            loser_prob = p1_win_prob
            color_win = "#4ade80" # Verde
            
        # Etiquetas de confianza
        if final_prob >= 0.80: confidence = "🔥 Muy Alta"
        elif final_prob >= 0.65: confidence = "✅ Alta"
        elif final_prob >= 0.55: confidence = "⚖️ Reñido"
        else: 
            confidence = "🎲 Moneda al aire"
            color_win = "#facc15" # Amarillo warning

        st.markdown(f"## {p1} <span style='color:#64748b; font-size:18px'>vs</span> {p2}", unsafe_allow_html=True)
        
        # KPI ROW
        k1, k2, k3, k4 = st.columns(4)
        
        k1.markdown(f"""
        <div class='metric-container' style='border-color: {color_win}; box-shadow: 0 0 10px {color_win}20;'>
            <div class='metric-label'>Ganador Estimado</div>
            <div class='metric-value' style='font-size: 20px; color: {color_win};'>{pred_winner}</div>
            <div style='font-size: 12px; color: #cbd5e1; margin-top:5px;'>{confidence}</div>
        </div>
        """, unsafe_allow_html=True)
        
        k2.markdown(f"""
        <div class='metric-container'>
            <div class='metric-label'>Probabilidad</div>
            <div class='metric-value'>{final_prob:.1%}</div>
            <div style='font-size: 10px; color: #64748b;'>Rival: {loser_prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        avg_games = sim_df['total_games'].mean()
        k3.markdown(f"""
        <div class='metric-container'>
            <div class='metric-label'>Total Juegos</div>
            <div class='metric-value'>{avg_games:.1f}</div>
            <div style='font-size: 10px; color: #64748b;'>±{sim_df['total_games'].std():.1f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        fair_odd = 1/final_prob if final_prob > 0 else 99
        k4.markdown(f"""
        <div class='metric-container'>
            <div class='metric-label'>Cuota Justa</div>
            <div class='metric-value'>{fair_odd:.2f}</div>
            <div style='font-size: 10px; color: #64748b;'>Valor si cuota > {fair_odd:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # TABS DETALLADOS
        tab1, tab2, tab3 = st.tabs(["📊 Mercados", "📈 Distribución", "🎾 Stats Técnicas"])
        
        with tab1:
            c_ou, c_hc = st.columns(2)
            with c_ou:
                st.markdown("#### 🔢 Over / Under")
                lines = range(int(avg_games)-3, int(avg_games)+4)
                ou_data = []
                for l in lines:
                    over = (sim_df['total_games'] > l).mean()
                    if 0.15 < over < 0.85:
                        ou_data.append({"Línea": l, "Over %": over, "Cuota O": 1/over, "Under %": 1-over, "Cuota U": 1/(1-over)})
                
                df_ou = pd.DataFrame(ou_data)
                if not df_ou.empty:
                    st.dataframe(
                        df_ou.style.format({
                            "Over %": "{:.1%}", "Under %": "{:.1%}", "Cuota O": "{:.2f}", "Cuota U": "{:.2f}"
                        }).background_gradient(subset=['Over %'], cmap='RdYlGn'),
                        use_container_width=True, hide_index=True
                    )

            with c_hc:
                st.markdown(f"#### 🏁 Hándicap ({p1})")
                hc_lines = [-4.5, -3.5, -2.5, -1.5, 1.5, 2.5, 3.5, 4.5]
                hc_data = []
                for h in hc_lines:
                    cover = (sim_df['diff_games'] + h > 0).mean() # P1 gana handicap?
                    if 0.15 < cover < 0.85:
                        hc_data.append({"Hándicap": h, "Probabilidad": cover, "Cuota Real": 1/cover})
                
                df_hc = pd.DataFrame(hc_data)
                if not df_hc.empty:
                    st.dataframe(
                        df_hc.style.format({"Probabilidad": "{:.1%}", "Cuota Real": "{:.2f}"})
                        .background_gradient(subset=['Probabilidad'], cmap='Blues'),
                        use_container_width=True, hide_index=True
                    )

        with tab2:
            fig = px.histogram(sim_df, x="total_games", nbins=20, title="Frecuencia de Juegos Totales", color_discrete_sequence=['#38bdf8'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', bargap=0.1)
            fig.add_vline(x=avg_games, line_dash="dash", line_color="#f472b6", annotation_text="Media")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            st.markdown("#### 🧬 ADN del Partido (Inputs de Simulación)")
            cg1, cg2 = st.columns(2)
            
            def draw_gauge(val, title, color):
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = val * 100, title = {'text': title},
                    gauge = {'axis': {'range': [40, 90]}, 'bar': {'color': color}, 'bgcolor': "#1e293b", 'borderwidth': 0}
                ))
                fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                return fig

            with cg1: st.plotly_chart(draw_gauge(sim_p1, f"Saque Real {p1}", "#4ade80"), use_container_width=True)
            with cg2: st.plotly_chart(draw_gauge(sim_p2, f"Saque Real {p2}", "#f87171"), use_container_width=True)
            st.info(f"Valores calculados: Saque Histórico - Calidad Resto Rival + Ajuste Superficie ({surf})")

    elif not analyze_btn:
        st.info("👈 Selecciona jugadores en el menú lateral para comenzar.")

# ==============================================================================
#                                   MÓDULO NBA
# ==============================================================================
elif deporte == "🏀 NBA Basket":
    
    @st.cache_resource
    def load_nba():
        try:
            mw = joblib.load('nba_model_win.joblib')
            mp = joblib.load('nba_model_pts.joblib')
            f = joblib.load('nba_features.joblib')
            d = joblib.load('nba_db_teams.joblib')
            return mw, mp, f, d
        except: return None, None, None, None

    mw, mp, feats, db = load_nba()
    
    if db is None:
        st.error("⚠️ Faltan archivos NBA. Ejecuta 'actualizar_nba.py' y 'entrenar_ia_nba.py'.")
        st.stop()
        
    def run_monte_carlo_nba(t1_stats, t2_stats, n=2000):
        pace = (t1_stats['EWMA_PACE'] + t2_stats['EWMA_PACE']) / 2
        off1 = t1_stats['EWMA_OFF_RTG']
        off2 = t2_stats['EWMA_OFF_RTG']
        res = []
        for _ in range(n):
            noise1 = np.random.normal(0, 12) 
            noise2 = np.random.normal(0, 12)
            pts1 = ((pace/100) * off1) + 3 + noise1 # +3 Home Adv
            pts2 = ((pace/100) * off2) + noise2
            res.append({'winner': 1 if pts1 > pts2 else 2, 'total_pts': pts1 + pts2, 'diff': pts1 - pts2})
        return pd.DataFrame(res)

    teams = sorted(db['TEAM_NAME'].unique())
    t1 = st.sidebar.selectbox("Equipo Local (Casa)", teams, index=0)
    t2 = st.sidebar.selectbox("Equipo Visitante", teams, index=1)
    
    if st.sidebar.button("Analizar NBA", type="primary"):
        d1 = db[db['TEAM_NAME'] == t1].iloc[0]
        d2 = db[db['TEAM_NAME'] == t2].iloc[0]
        
        with st.spinner("Simulando partido en la cancha..."):
            sim_df = run_monte_carlo_nba(d1, d2)
            
        win_pct = sim_df['winner'].value_counts(normalize=True).get(1, 0)
        avg_pts = sim_df['total_pts'].mean()
        spread = sim_df['diff'].mean()
        
        st.title(f"{t1} vs {t2}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-container'><div class='metric-label'>Prob. Local</div><div class='metric-value'>{win_pct:.1%}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-container'><div class='metric-label'>Total Puntos</div><div class='metric-value'>{avg_pts:.1f}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-container'><div class='metric-label'>Spread Estimado</div><div class='metric-value'>{spread:+.1f}</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-container'><div class='metric-label'>Cuota Justa</div><div class='metric-value'>{1/win_pct if win_pct>0 else 99:.2f}</div></div>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📊 Mercados", "📈 Distribución"])
        
        with tab1:
            co, ch = st.columns(2)
            with co:
                st.markdown("#### 🔢 Over / Under")
                lines = range(int(avg_pts)-5, int(avg_pts)+6)
                ou_data = []
                for l in lines:
                    over = (sim_df['total_pts'] > l).mean()
                    ou_data.append({"Línea": l, "Over %": f"{over:.1%}", "Cuota O": f"{1/over:.2f}"})
                st.dataframe(pd.DataFrame(ou_data), hide_index=True, use_container_width=True)
                
            with ch:
                st.write("#### 🏁 Hándicap")
                hc_data = []
                h_lines = [-10.5, -7.5, -4.5, -1.5, 1.5, 4.5, 7.5]
                for h in h_lines:
                    cover = (sim_df['diff'] + h > 0).mean()
                    hc_data.append({"Hándicap Local": h, "Probabilidad": f"{cover:.1%}", "Cuota": f"{1/cover:.2f}"})
                st.dataframe(pd.DataFrame(hc_data), hide_index=True, use_container_width=True)
                
        with tab2:
             fig = px.histogram(sim_df, x="total_pts", nbins=30, title="Distribución de Puntos", color_discrete_sequence=['#f59e0b'])
             fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
             fig.add_vline(x=avg_pts, line_dash="dash", line_color="white", annotation_text="Media")
             st.plotly_chart(fig, use_container_width=True)

    elif not st.sidebar.button:
        st.info("👈 Selecciona equipos NBA para comenzar.")