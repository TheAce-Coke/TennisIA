import os
import sys
import time

def paso(comando, nombre):
    print(f"\n🚀 INICIANDO: {nombre}...")
    inicio = time.time()
    # Ejecuta el comando en la terminal del sistema
    codigo = os.system(comando)
    fin = time.time()
    
    if codigo == 0:
        print(f"✅ {nombre} COMPLETADO ({fin-inicio:.1f}s)")
    else:
        print(f"❌ ERROR CRÍTICO en {nombre}.")
        sys.exit()

print("===================================================")
print("      ⚡ ACTUALIZADOR MAESTRO (TENIS + NBA) ⚡")
print("===================================================")

# --- PARTE 1: TENIS ---
if os.path.exists("actualizar_auto.py"):
    paso("python actualizar_auto.py", "Descarga Datos Tenis")
    paso("python crear_ia.py", "Procesado Elo Tenis")
    paso("python entrenar_ia.py", "Entrenamiento IA Tenis")
else:
    print("⚠️ No encuentro actualizar_auto.py (Saltando Tenis)")

# --- PARTE 2: NBA ---
if os.path.exists("actualizar_nba.py"):
    paso("python actualizar_nba.py", "Descarga Datos NBA")
    paso("python crear_ia_nba.py", "Procesado Elo NBA")
    paso("python entrenar_ia_nba.py", "Entrenamiento IA NBA")
else:
    print("⚠️ No encuentro actualizar_nba.py (Saltando NBA)")

print("\n" + "="*50)
print("       🎉 TODO ACTUALIZADO: APP LISTA 🎉")
print("="*50)
input("Presiona Enter para salir...")