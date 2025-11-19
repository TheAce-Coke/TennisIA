import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURACIÓN VISUAL PREMIUM ---
st.set_page_config(page_title="NeuralTennis Quant", page_icon="🎾", layout="wide")

# CSS Avanzado (Sin Canvas, puro CSS3)
st.markdown("""
<style>
    /* Fondo general */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #020617 90%);
        color: #e2e8f0;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid #1e293b;
    }

    /* Cards Metrics */
    .metric-container {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-top: 5px;
    }

    /* Headers */
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap;
        background-color: #1e293b; border-radius: 8px; color: #cbd5e1; border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CARGA DE DATOS ---
@st.cache_resource
def load_data():
    try:
        m = joblib.load('modelo_calibrado.joblib')
        f = joblib.load('features.joblib')
        d = joblib.load('db_players.joblib')
        return m, f, d
    except: return None, None, None

model, features, db = load_data()

if db is None:
    st.error("⚠️ Sistema detenido: Faltan archivos de datos...")
    st.stop()

# --- MOTOR SIMULACIÓN (MATH FIX) ---
def sim_point(prob_server):
    return 1 if random.random() < prob_server else 0

def sim_game(prob_server):
    # Simulación realista de un juego (0, 15, 30, 40, Deuce)
    p1, p2 = 0, 0
    while True:
        if sim_point(prob_server): p1 += 1
        else: p2 += 1
        if p1 >= 4 and p1 >= p2 + 2: return 1, p1+p2
        if p2 >= 4 and p2 >= p1 + 2: return 0, p1+p2

def sim_set(p1_prob, p2_prob):
    g1, g2 = 0, 0
    pts = 0
    # Tie Break Logic
    while g1 < 6 and g2 < 6:
        w, p = sim_game(p1_prob); pts += p
        if w: g1 += 1
        if g1==6 and g2<5: return g1, g2, pts # 6-4
        
        w, p = sim_game(p2_prob); pts += p
        if not w: g2 += 1
        else: g1 += 1 # Break
        if g2==6 and g1<5: return g1, g2, pts # 4-6
        
    # 6-6 -> Tie Break
    tb1, tb2 = 0, 0
    # En TB, la ventaja de saque se diluye ligeramente
    while True:
        # Turnos alternos simplificados
        if (tb1+tb2)%2 == 0: # Saque A
            if sim_point(p1_prob): tb1+=1
            else: tb2+=1
        else: # Saque B
            if sim_point(p2_prob): tb2+=1
            else: tb1+=1
            
        if (tb1>=7 and tb1>=tb2+2): return 7, 6, pts+tb1+tb2
        if (tb2>=7 and tb2>=tb1+2): return 6, 7, pts+tb1+tb2

def run_monte_carlo(p1_prob, p2_prob, best_of=3, n_sims=1500):
    data = []
    needed = 2 if best_of==3 else 3
    
    for _ in range(n_sims):
        s1, s2 = 0, 0
        tot_games = 0
        games_p1, games_p2 = 0, 0
        
        while s1 < needed and s2 < needed:
            g1, g2, _ = sim_set(p1_prob, p2_prob)
            s1 += 1 if g1 > g2 else 0
            s2 += 1 if g2 > g1 else 0
            tot_games += (g1 + g2)
            games_p1 += g1
            games_p2 += g2
            
        data.append({
            'winner': 1 if s1 > s2 else 2,
            'total_games': tot_games,
            'diff_games': games_p1 - games_p2,
            'sets_score': f"{s1}-{s2}"
        })
    return pd.DataFrame(data)

# --- UI SIDEBAR ---
with st.sidebar:
    st.markdown("### 🎛️ Control Center")
    
    players_list = sorted(db['player_name'].unique())
    # Índices por defecto seguros
    idx_1 = players_list.index("Alcaraz C.") if "Alcaraz C." in players_list else 0
    idx_2 = players_list.index("Sinner J.") if "Sinner J." in players_list else 1
    
    p1 = st.selectbox("Jugador 1 (Servicio)", players_list, index=idx_1)
    p2 = st.selectbox("Jugador 2 (Resto)", players_list, index=idx_2)
    
    st.markdown("---")
    surface = st.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    best_of = st.radio("Formato", [3, 5], horizontal=True)
    
    analyze_btn = st.button("⚡ EJECUTAR SIMULACIÓN", type="primary", use_container_width=True)
    
    st.markdown("""
    <div style='margin-top: 20px; padding: 10px; background: #1e293b; border-radius: 8px; font-size: 11px; color: #64748b;'>
    <b>Quant Engine v2.0</b><br>
    Utiliza Log5 Prob y Monte Carlo (1500 runs).
    Datos ajustados por superficie y fatiga.
    </div>
    """, unsafe_allow_html=True)

# --- MAIN AREA ---
if analyze_btn and p1 != p2:
    # 1. Extracción de Stats
    d1 = db[db['player_name'] == p1].iloc[0]
    d2 = db[db['player_name'] == p2].iloc[0]
    
    # 2. CALCULO MATEMÁTICO (EL CORAZÓN DEL SISTEMA)
    # P_Serve_Real = Stats_Saque_P1 - (Promedio_Tour - Stats_Resto_P2)
    # Si P2 resta muy bien (Stats_Resto alto), baja el saque de P1.
    TOUR_AVG_SERVE = 0.64 # Media ATP
    
    # Ajuste Superficie
    surf_adj = 0
    if surface == 'Clay': surf_adj = -0.04 # Se rompe más en tierra
    if surface == 'Grass': surf_adj = 0.03 # Se rompe menos en hierba
    if surface == 'Hard': surf_adj = 0.01
    
    # Probabilidad Base P1 al Saque (LOG5 Modificado)
    # P1 Serve Base + (P2 es mal restador?) + Superficie
    p1_serve_base = d1['ewma_serve']
    p2_return_quality = d2['ewma_return'] - (1 - TOUR_AVG_SERVE) # Que tan bueno es restando vs media
    
    sim_p1_serve = p1_serve_base - p2_return_quality + surf_adj
    
    # Lo mismo para P2
    p2_serve_base = d2['ewma_serve']
    p1_return_quality = d1['ewma_return'] - (1 - TOUR_AVG_SERVE)
    sim_p2_serve = p2_serve_base - p1_return_quality + surf_adj
    
    # Clips de seguridad (nadie saca al 95% ni al 30%)
    sim_p1_serve = np.clip(sim_p1_serve, 0.45, 0.82)
    sim_p2_serve = np.clip(sim_p2_serve, 0.45, 0.82)

    # 3. Ejecución Monte Carlo
    with st.spinner(f"Simulando enfrentamiento: {p1} ({sim_p1_serve:.1%}) vs {p2} ({sim_p2_serve:.1%})..."):
        sim_df = run_monte_carlo(sim_p1_serve, sim_p2_serve, best_of)

    # 4. Métricas Principales
    win_rate = sim_df['winner'].value_counts(normalize=True).get(1, 0)
    avg_games = sim_df['total_games'].mean()
    std_games = sim_df['total_games'].std()
    
    # --- HEADER VISUAL ---
    col_head1, col_head2 = st.columns([2, 1])
    with col_head1:
        st.markdown(f"<h2 style='margin-bottom:0;'>{p1} <span style='color:#64748b; font-size:18px;'>vs</span> {p2}</h2>", unsafe_allow_html=True)
        st.caption(f"{surface} | Best of {best_of} | Elo: {int(d1['player_elo'])} vs {int(d2['player_elo'])}")
        
    # KPI ROW
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='metric-container'><div class='metric-label'>Prob. Victoria</div><div class='metric-value'>{win_rate:.1%}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='metric-container'><div class='metric-label'>Cuota Justa</div><div class='metric-value'>{1/win_rate if win_rate>0 else 99:.2f}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='metric-container'><div class='metric-label'>Total Juegos</div><div class='metric-value'>{avg_games:.1f}</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='metric-container'><div class='metric-label'>Volatilidad</div><div class='metric-value'>±{std_games:.1f}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- TABS ANÁLISIS ---
    tab1, tab2, tab3 = st.tabs(["📊 Mercados Principales", "📈 Distribución de Juegos", "🎾 Estadísticas Técnicas"])
    
    with tab1:
        c_ou, c_hc = st.columns(2)
        
        with c_ou:
            st.markdown("#### 🔢 Over / Under")
            # Tabla dinámica de Over/Under
            lines = range(int(avg_games)-3, int(avg_games)+4)
            ou_data = []
            for l in lines:
                over = (sim_df['total_games'] > l).mean()
                if 0.2 < over < 0.8: # Filtro de relevancia
                    ou_data.append({
                        "Línea": l, 
                        "Over %": over, 
                        "Cuota O": 1/over, 
                        "Under %": 1-over,
                        "Cuota U": 1/(1-over)
                    })
            
            df_ou = pd.DataFrame(ou_data)
            if not df_ou.empty:
                st.dataframe(
                    df_ou.style.format({
                        "Over %": "{:.1%}", "Under %": "{:.1%}", 
                        "Cuota O": "{:.2f}", "Cuota U": "{:.2f}"
                    }).background_gradient(subset=['Over %'], cmap='RdYlGn'),
                    use_container_width=True, hide_index=True
                )
            else: st.info("Líneas muy extremas, sin datos relevantes.")

        with c_hc:
            st.markdown(f"#### 🏁 Hándicap ({p1})")
            hc_lines = [-4.5, -3.5, -2.5, -1.5, 1.5, 2.5, 3.5, 4.5]
            hc_data = []
            for h in hc_lines:
                cover = (sim_df['diff_games'] + h > 0).mean()
                if 0.15 < cover < 0.85:
                    hc_data.append({
                        "Hándicap": h,
                        "Probabilidad": cover,
                        "Cuota Real": 1/cover
                    })
            df_hc = pd.DataFrame(hc_data)
            if not df_hc.empty:
                st.dataframe(
                    df_hc.style.format({"Probabilidad": "{:.1%}", "Cuota Real": "{:.2f}"})
                    .background_gradient(subset=['Probabilidad'], cmap='Blues'),
                    use_container_width=True, hide_index=True
                )

    with tab2:
        # GRÁFICO PLOTLY (MUCHO MEJOR QUE ST.BAR_CHART)
        fig = px.histogram(sim_df, x="total_games", nbins=20, title="Distribución de Juegos Totales",
                           color_discrete_sequence=['#38bdf8'])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"), bargap=0.1,
            xaxis=dict(title="Total de Juegos"), yaxis=dict(title="Frecuencia (Simulaciones)")
        )
        # Línea media
        fig.add_vline(x=avg_games, line_dash="dash", line_color="#f472b6", annotation_text="Media")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### 🧬 ADN del Enfrentamiento (Inputs Simulación)")
        col_t1, col_t2 = st.columns(2)
        
        def draw_gauge(val, title, color):
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = val * 100, title = {'text': title},
                gauge = {'axis': {'range': [40, 90]}, 'bar': {'color': color},
                         'bgcolor': "#1e293b", 'borderwidth': 0}
            ))
            fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            return fig

        with col_t1:
            st.plotly_chart(draw_gauge(sim_p1_serve, f"Efectividad Saque {p1}", "#4ade80"), use_container_width=True)
        with col_t2:
            st.plotly_chart(draw_gauge(sim_p2_serve, f"Efectividad Saque {p2}", "#f87171"), use_container_width=True)
            
        st.info(f"""
        ℹ️ **Explicación Quant:** {p1} tiene un Saque Histórico de **{d1['ewma_serve']:.1%}**.
        {p2} permite al rival ganar el **{1-d2['ewma_return']:.1%}** de puntos al saque.
        El modelo ajusta estos valores por superficie ({surface}) y genera la probabilidad final usada en Monte Carlo.
        """)

elif not analyze_btn:
    st.markdown("""
    <div style='text-align:center; padding: 50px; color: #64748b;'>
    <h3>👋 Bienvenido a NeuralTennis Quant</h3>
    Selecciona jugadores en el menú lateral para iniciar el motor de simulación.
    </div>
    """, unsafe_allow_html=True)