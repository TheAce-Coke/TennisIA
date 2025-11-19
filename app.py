import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- CONFIGURACIÓN VISUAL AVANZADA ---
st.set_page_config(page_title="Tennis Neural Core", page_icon="🧠", layout="wide")

# Inyección de CSS para estilo "Cyberpunk/AI"
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .title-box {
        text-align: center;
        padding: 20px;
        border-bottom: 1px solid #30363d;
        margin-bottom: 30px;
    }
    .stat-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .big-number {
        font-size: 24px;
        font-weight: bold;
        color: #58a6ff;
        font-family: 'Courier New', monospace;
    }
    .metric-label {
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .prediction-box {
        background: linear-gradient(45deg, #1f6feb, #238636);
        padding: 2px;
        border-radius: 12px;
        margin: 20px 0;
    }
    .prediction-inner {
        background-color: #0d1117;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    div[data-testid="stProgressBar"] > div > div {
        background-color: #238636;
    }
</style>
""", unsafe_allow_html=True)

# --- CARGA DE MODELOS ---
@st.cache_resource
def cargar_cerebro():
    try:
        # Intentamos cargar V3 (Features unificadas)
        if os.path.exists('features.joblib'):
            feats = joblib.load('features.joblib')
        elif os.path.exists('features_ganador.joblib'):
            feats = joblib.load('features_ganador.joblib')
        else: return None

        if not os.path.exists('modelo_ganador.joblib'): return None
        
        m_win = joblib.load('modelo_ganador.joblib')
        m_games = joblib.load('modelo_juegos.joblib')
        
        df = pd.read_csv("atp_matches_procesados.csv")
        # Limpieza rápida al cargar
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
    st.error("⚠️ SYSTEM FAILURE: Archivos no encontrados en el repositorio.")
    st.stop()

model_win, model_games, feats, df = sistema

# --- LÓGICA INTELIGENTE DE DATOS ---
def imputar_stats_faltantes(perfil, ranking):
    """Si faltan datos, los inventa basándose en el Ranking (Lógica Pro)."""
    # Valores base según ranking (Top 50 vs Top 100 vs Resto)
    if ranking <= 50:
        base_ace = 0.08; base_1st = 0.74; base_bp = 0.62; base_form = 0.65
    elif ranking <= 150:
        base_ace = 0.06; base_1st = 0.68; base_bp = 0.58; base_form = 0.55
    else:
        base_ace = 0.04; base_1st = 0.62; base_bp = 0.54; base_form = 0.50

    # Corregimos si los datos son muy bajos (sospecha de error/falta de datos)
    if perfil.get('player_1st_won_avg', 0) < 0.40: perfil['player_1st_won_avg'] = base_1st
    if perfil.get('player_bp_save_avg', 0) < 0.30: perfil['player_bp_save_avg'] = base_bp
    if perfil.get('player_ace_avg', 0) == 0: perfil['player_ace_avg'] = base_ace
    if perfil.get('player_form', 0) == 0: perfil['player_form'] = base_form
    
    return perfil

def get_smart_profile(player_name):
    col_p = 'player_name' if 'player_name' in df.columns else 'Player_1'
    matches = df[df[col_p] == player_name].copy()
    
    if matches.empty: return None
    
    # Últimos 15 partidos válidos
    last = matches.tail(15)
    
    # Ranking más reciente
    current_rank = last.iloc[-1]['player_rank']
    if pd.isna(current_rank) or current_rank == 0: current_rank = 500
    
    # Calculamos medias ignorando ceros
    stats = {}
    cols = ['player_ace_avg', 'player_1st_won_avg', 'player_bp_save_avg', 'player_form', 'player_surf_win']
    
    for c in cols:
        if c in last.columns:
            val = last[last[c] > 0.01][c].mean() # Media sin ceros
            stats[c] = val if not pd.isna(val) else 0
        else:
            stats[c] = 0
            
    stats['player_rank'] = current_rank
    
    # APLICAMOS LA IMPUTACIÓN INTELIGENTE
    stats = imputar_stats_faltantes(stats, current_rank)
    
    return stats

# --- INTERFAZ ---

# 1. Header
st.markdown("<div class='title-box'><h1>🧠 Tennis Neural Core v5.0</h1></div>", unsafe_allow_html=True)

# 2. Sidebar (Control Panel)
with st.sidebar:
    st.header("⚙️ Configuración de Entrada")
    
    # Buscador
    col_p = 'player_name' if 'player_name' in df.columns else 'Player_1'
    all_players = sorted(list(set(df[col_p].unique()) | set(df['opponent_name' if 'opponent_name' in df.columns else 'Player_2'].unique())))
    
    p1 = st.selectbox("Jugador 1", all_players, index=None, placeholder="Buscar Jugador...")
    p2 = st.selectbox("Jugador 2", all_players, index=None, placeholder="Buscar Jugador...")
    
    # Auto-Ranking
    r1, r2 = 500, 500
    if p1: 
        prof1 = get_smart_profile(p1)
        if prof1: r1 = int(prof1['player_rank'])
    if p2: 
        prof2 = get_smart_profile(p2)
        if prof2: r2 = int(prof2['player_rank'])

    col_r1, col_r2 = st.columns(2)
    rank1 = col_r1.number_input("Rank J1", value=r1)
    rank2 = col_r2.number_input("Rank J2", value=r2)
    
    surf = st.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    bo = st.selectbox("Sets", [3, 5])
    
    run_btn = st.button("⚡ EJECUTAR ANÁLISIS", type="primary")

# 3. Main Display
if run_btn:
    if p1 and p2 and p1 != p2:
        prof1 = get_smart_profile(p1)
        prof2 = get_smart_profile(p2)
        
        if prof1 and prof2:
            # Override ranking manual
            prof1['player_rank'] = rank1
            prof2['player_rank'] = rank2
            
            # H2H
            h2h_data = df[(df[col_p] == p1) & (df['opponent_name' if 'opponent_name' in df.columns else 'Player_2'] == p2)]
            h2h_w = len(h2h_data[h2h_data['result']==1]) if 'result' in df.columns else len(h2h_data)
            
            # Preparar vector de entrada
            row = {
                'player_rank': rank1, 'opponent_rank': rank2, 'Best of': bo,
                'player_form': prof1['player_form'], 'opponent_form': prof2['player_form'],
                'h2h_wins': h2h_w, 'h2h_total': len(h2h_data),
                'player_surf_win': prof1.get('player_surf_win', 0.5), 
                'opponent_surf_win': prof2.get('player_surf_win', 0.5),
                'player_ace_avg': prof1['player_ace_avg'], 'opponent_ace_avg': prof2['player_ace_avg'],
                'player_1st_won_avg': prof1['player_1st_won_avg'], 'opponent_1st_won_avg': prof2['player_1st_won_avg'],
                'player_bp_save_avg': prof1['player_bp_save_avg'], 'opponent_bp_save_avg': prof2['player_bp_save_avg']
            }
            
            for f in feats:
                if 'Surface_' in f: row[f] = 1 if f == f'Surface_{surf}' else 0
            
            # Predicción
            try:
                X_in = pd.DataFrame([row])
                for c in feats: 
                    if c not in X_in.columns: X_in[c] = 0
                X_in = X_in[feats] # Orden estricto

                prob = model_win.predict_proba(X_in)[0][1]
                games = model_games.predict(X_in)[0]
                
                # --- VISUALIZACIÓN RESULTADOS ---
                
                # 1. Tarjeta Principal (Winner)
                st.markdown(f"""
                <div class='prediction-box'>
                    <div class='prediction-inner'>
                        <h2 style='margin:0; color:#fff;'>{p1} vs {p2}</h2>
                        <p style='color:#8b949e;'>{surf} | Best of {bo}</p>
                        <hr style='border-color:#30363d;'>
                        <div style='display:flex; justify-content:space-around; align-items:center;'>
                             <div>
                                <div class='metric-label'>Probabilidad {p1}</div>
                                <div class='big-number' style='color:{'#238636' if prob>0.5 else '#da3633'}'>{prob*100:.1f}%</div>
                             </div>
                             <div>
                                <div class='metric-label'>Juegos Estimados</div>
                                <div class='big-number' style='color:#a371f7'>{games:.1f}</div>
                             </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if prob > 0.55:
                    st.success(f"💎 **PICK RECOMENDADO:** {p1} (Cuota de Valor > {1/prob:.2f})")
                elif prob < 0.45:
                    st.success(f"💎 **PICK RECOMENDADO:** {p2} (Cuota de Valor > {1/(1-prob):.2f})")
                else:
                    st.warning("⚠️ Partido sin claro favorito (No Bet)")

                # 2. Métricas Comparativas (Cards)
                st.markdown("### 🔬 Análisis de Métricas Recuperadas")
                c1, c2, c3, c4 = st.columns(4)
                
                with c1:
                    st.markdown(f"<div class='stat-card'><div class='metric-label'>Ranking</div><div class='big-number' style='font-size:18px'>{rank1} vs {rank2}</div></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='stat-card'><div class='metric-label'>Forma</div><div class='big-number' style='font-size:18px'>{prof1['player_form']:.0%} vs {prof2['player_form']:.0%}</div></div>", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"<div class='stat-card'><div class='metric-label'>Saque (1st W)</div><div class='big-number' style='font-size:18px'>{prof1['player_1st_won_avg']:.0%} vs {prof2['player_1st_won_avg']:.0%}</div></div>", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"<div class='stat-card'><div class='metric-label'>Mental (BP)</div><div class='big-number' style='font-size:18px'>{prof1['player_bp_save_avg']:.0%} vs {prof2['player_bp_save_avg']:.0%}</div></div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error de cálculo: {e}")
        else:
            st.error("Datos insuficientes para generar perfil completo.")
    else:
        st.warning("Selecciona jugadores diferentes.")

# Footer
st.markdown("<div style='text-align:center; color:#444; margin-top:50px;'>Neural Core v5.0 | Powered by Gradient Boosting</div>", unsafe_allow_html=True)