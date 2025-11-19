import os
import time
import sys

def ejecutar_paso(comando, descripcion):
    print(f"\n🚀 INICIANDO: {descripcion}...")
    print("-" * 50)
    start = time.time()
    ret = os.system(comando)
    end = time.time()
    
    if ret == 0:
        print(f"✅ {descripcion} COMPLETADO ({end-start:.2f}s)")
    else:
        print(f"❌ ERROR CRÍTICO en {descripcion}.")
        sys.exit(1) # Salir con código de error para que GitHub avise

print("""
===================================================
   ⚡ NEURALSPORTS: ACTUALIZADOR UNIFICADO ⚡
===================================================
Este script actualizará las bases de datos de:
   1. 🎾 Tenis ATP (Historia + Stats Avanzadas)
   2. 🏀 NBA (Regular Season + Playoffs)
   
Y re-entrenará las IAs correspondientes.
""")

# --- FASE 1: TENIS ---
if os.path.exists("actualizar_auto.py"):
    ejecutar_paso("python actualizar_auto.py", "Descarga Datos Tenis")
    ejecutar_paso("python crear_ia.py", "Procesado Elo Tenis")
    ejecutar_paso("python entrenar_ia.py", "Entrenamiento IA Tenis")
else:
    print("⚠️ Saltando Tenis (Falta actualizar_auto.py)")

# --- FASE 2: NBA ---
if os.path.exists("actualizar_nba.py"):
    ejecutar_paso("python actualizar_nba.py", "Descarga Datos NBA")
    ejecutar_paso("python crear_ia_nba.py", "Ingeniería de Datos NBA")
    ejecutar_paso("python entrenar_ia_nba.py", "Entrenamiento IA NBA")
else:
    print("⚠️ Saltando NBA (Falta actualizar_nba.py)")

print("\n" + "="*50)
print("       🎉 TODO ACTUALIZADO: APP LISTA 🎉")
print("="*50)

# --- FIX PARA GITHUB ACTIONS ---
# Solo pedimos input si NO estamos en un servidor de automatización
if "GITHUB_ACTIONS" not in os.environ:
    try:
        input("\nPresiona Enter para salir...")
    except EOFError:
        pass