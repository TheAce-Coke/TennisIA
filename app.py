import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tennis AI Pro", page_icon="🎾", layout="centered")

# --- CARGA DE DATOS ---
@st.cache_resource
def cargar_datos():
    # Intentamos cargar la versión V3 (Features unificadas)
    if os.path.exists('features.joblib'):
        feats = joblib.load('features.joblib')
    elif os.path.exists('features_ganador.joblib'):
        feats = joblib.load('features_ganador.joblib')
    else:
        return None

    if not os.path.exists('modelo_ganador.joblib'): return None
    
    m_win = joblib.load('modelo_ganador.joblib')
    m_games = joblib.load('modelo_juegos.joblib')
    
    # Cargar CSV
    try:
        df = pd.read_csv("atp_matches_procesados.csv")
        # Parches de compatibilidad de nombres de columnas
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
    st.error("❌ Error: No encuentro los archivos. Asegúrate de haber subido todo a GitHub.")
    st.stop()

model_win, model_games, feats, df = res

# --- MENÚ LATERAL ---
st.sidebar.title("Menú")
modo = st.sidebar.radio("Ir a:", ["🔮 Predictor PRO", "🕵️‍♂️ Detective de Nombres"])

# ==============================================================================
# MODO 1: DETECTIVE (BUSCADOR AVANZADO)
# ==============================================================================
if modo == "🕵️‍♂️ Detective de Nombres":
    st.title("🕵️‍♂️ Detective de Jugadores")
    st.info("Encuentra cómo está escrito el nombre de un jugador en la base de datos.")
    
    busqueda = st.text_input("Escribe nombre o apellido (ej: Passaro):")
    
    if busqueda:
        col_p1 = 'player_name' if 'player_name' in df.columns else 'Player_1'
        col_p2 = 'opponent_name' if 'opponent_name' in df.columns else 'Player_2'
        
        # Buscar en ambas columnas
        r1 = df[df[col_p1].astype(str).str.contains(busqueda, case=False, na=False)][col_p1].unique()
        r2 = df[df[col_p2].astype(str).str.contains(busqueda, case=False, na=False)][col_p2].unique()
        
        resultados = sorted(list(set(list(r1) + list(r2))))
        
        if resultados:
            st.success(f"✅ He encontrado {len(resultados)} variantes:")
            for r in resultados:
                st.code(r)
        else:
            st.warning("❌ No encontrado. Prueba con menos letras.")

# ==============================================================================
# MODO 2: PREDICTOR PRO
# ==============================================================================
elif modo == "🔮 Predictor PRO":
    st.title("🎾 Tennis AI Pro V3")
    
    def get_stats(player):
        col_jugador = 'player_name' if 'player_name' in df.columns else 'Player_1'
        row = df[df[col_jugador] == player]
        if row.empty: return None
        return row.iloc[-1]

    # --- SIDEBAR ---
    st.sidebar.header("Configuración del Partido")
    
    col_p1 = 'player_name' if 'player_name' in df.columns else 'Player_1'
    col_p2 = 'opponent_name' if 'opponent_name' in df.columns else 'Player_2'
    players = sorted(list(set(df[col_p1].unique()) | set(df[col_p2].unique())))
    
    p1 = st.sidebar.selectbox("Jugador 1", players, index=None, placeholder="Buscar...")
    p2 = st.sidebar.selectbox("Jugador 2", players, index=None, placeholder="Buscar...")
    surf = st.sidebar.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    bo = st.sidebar.selectbox("Sets", [3, 5])
    
    if st.sidebar.button("⚡ Analizar Partido"):
        if p1 and p2 and p1 != p2:
            s1, s2 = get_stats(p1), get_stats(p2)
            
            if s1 is not None and s2 is not None:
                # Calcular H2H
                h2h_data = df[(df[col_p1] == p1) & (df[col_p2] == p2)]
                h2h_w = len(h2h_data) # Simplificado
                if 'result' in df.columns: h2h_w = h2h_data['result'].sum()
                h2h_t = len(h2h_data)

                # Preparar Input (Con valores seguros por si faltan columnas)
                row = {
                    'player_rank': s1.get('player_rank', 500), 'opponent_rank': s2.get('player_rank', 500),
                    'Best of': bo,
                    'player_form': s1.get('player_form', 0.5), 'opponent_form': s2.get('player_form', 0.5),
                    'h2h_wins': h2h_w, 'h2h_total': h2h_t,
                    # Nuevas Métricas V3
                    'player_surf_win': s1.get('player_surf_win', 0.5), 'opponent_surf_win': s2.get('player_surf_win', 0.5),
                    'player_ace_avg': s1.get('player_ace_avg', 0.05), 'opponent_ace_avg': s2.get('player_ace_avg', 0.05),
                    'player_1st_won_avg': s1.get('player_1st_won_avg', 0.65), 'opponent_1st_won_avg': s2.get('player_1st_won_avg', 0.65),
                    'player_bp_save_avg': s1.get('player_bp_save_avg', 0.55), 'opponent_bp_save_avg': s2.get('player_bp_save_avg', 0.55)
                }
                
                # One Hot Encoding Superficie
                for f in feats:
                    if 'Surface_' in f: row[f] = 1 if f == f'Surface_{surf}' else 0
                
                # Predecir
                try:
                    # Asegurar orden de columnas
                    X_in = pd.DataFrame([row])
                    # Rellenar columnas faltantes con 0
                    for c in feats: 
                        if c not in X_in.columns: X_in[c] = 0
                    X_in = X_in[feats]

                    prob = model_win.predict_proba(X_in)[0][1]
                    games = model_games.predict(X_in)[0]
                    
                    # --- VISUALIZACIÓN ---
                    st.divider()
                    
                    # 1. Encabezado
                    c1, c2, c3 = st.columns([5, 2, 5])
                    c1.markdown(f"### {p1}")
                    c1.caption(f"Ranking: {int(s1.get('player_rank', 0))}")
                    c3.markdown(f"### {p2}")
                    c3.caption(f"Ranking: {int(s2.get('player_rank', 0))}")
                    
                    # 2. Barra de Probabilidad
                    st.progress(prob, text=f"Probabilidad {p1}: {prob*100:.1f}%")
                    
                    if prob > 0.5:
                        st.success(f"🏆 **{p1}** es favorito (Cuota justa: {1/prob:.2f})")
                    else:
                        st.success(f"🏆 **{p2}** es favorito (Cuota justa: {1/(1-prob):.2f})")
                    
                    # 3. ESTADÍSTICAS COMPARADAS (¡LO QUE FALTABA!)
                    st.markdown("### 📊 Duelo de Estilos")
                    m1, m2, m3 = st.columns(3)
                    
                    # Forma
                    delta_form = (s1.get('player_form', 0) - s2.get('player_form', 0)) * 100
                    m1.metric("Racha Reciente", f"{s1.get('player_form', 0):.0%}", f"{delta_form:.0f}% vs rival")
                    
                    # Saque (1st Won)
                    delta_srv = (s1.get('player_1st_won_avg', 0) - s2.get('player_1st_won_avg', 0)) * 100
                    m2.metric("Efectividad Saque", f"{s1.get('player_1st_won_avg', 0):.0%}", f"{delta_srv:.0f}% vs rival")
                    
                    # Mental (BP Saved)
                    delta_bp = (s1.get('player_bp_save_avg', 0) - s2.get('player_bp_save_avg', 0)) * 100
                    m3.metric("Resistencia (BP)", f"{s1.get('player_bp_save_avg', 0):.0%}", f"{delta_bp:.0f}% vs rival")

                    st.divider()
                    st.info(f"🎾 Se estiman **{games:.1f} juegos**.")
                    
                except Exception as e:
                    st.error(f"Error calculando: {e}")
                    
            else:
                st.error("No se encontraron datos recientes para uno de los jugadores.")
    else:
        st.info("Selecciona jugadores y pulsa Analizar.")