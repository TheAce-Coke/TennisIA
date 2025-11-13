import os
import zipfile
import sys
import time

# --- CONFIGURACIÓN ---
DATASET = "dissfya/atp-tennis-2000-2023daily-pull"
ARCHIVO_FINAL = "atp_tennis.csv" 

print("==================================================")
print("   🤖 ACTUALIZADOR AUTOMÁTICO (VERSIÓN FINAL) 🤖")
print("==================================================")

# 1. DESCARGAR DESDE KAGGLE
print("\n--- 1. Descargando datos frescos desde Kaggle... ---")
comando = f"kaggle datasets download -d {DATASET} --force"
codigo = os.system(comando)

if codigo != 0:
    print("❌ Error al descargar. Revisa tu kaggle.json")
    sys.exit()
else:
    print("✅ Descarga completada.")
    time.sleep(2) # Esperamos a que el disco termine de escribir

# 2. BUSCAR EL ZIP AUTOMÁTICAMENTE
print("\n--- 2. Buscando el archivo ZIP descargado... ---")
archivo_zip_encontrado = None
archivos_en_carpeta = os.listdir(".")

for archivo in archivos_en_carpeta:
    if archivo.endswith(".zip") and ("atp" in archivo.lower() or "tennis" in archivo.lower()):
        archivo_zip_encontrado = archivo
        break

if not archivo_zip_encontrado:
    print("❌ Error: No encuentro el .zip.")
    sys.exit()

print(f"✅ ZIP encontrado: {archivo_zip_encontrado}")

# 3. DESCOMPRIMIR Y PREPARAR
print("\n--- 3. Descomprimiendo... ---")
try:
    with zipfile.ZipFile(archivo_zip_encontrado, 'r') as zip_ref:
        # Extraemos todo
        zip_ref.extractall(".")
        archivos_extraidos = zip_ref.namelist()
        
        # Buscamos el CSV
        csv_encontrado = None
        for archivo in archivos_extraidos:
            if archivo.endswith(".csv"):
                csv_encontrado = archivo
                break
        
        if csv_encontrado:
            print(f"   -> Archivo extraído: {csv_encontrado}")
            
            # LÓGICA CORREGIDA: Solo renombramos si tienen nombres distintos
            if csv_encontrado != ARCHIVO_FINAL:
                if os.path.exists(ARCHIVO_FINAL):
                    os.remove(ARCHIVO_FINAL) # Borramos el viejo solo si vamos a poner uno con nombre distinto
                os.rename(csv_encontrado, ARCHIVO_FINAL)
                print(f"✅ Renombrado a: {ARCHIVO_FINAL}")
            else:
                print(f"✅ El archivo ya tiene el nombre correcto ({ARCHIVO_FINAL}).")

            zip_ref.close()
            os.remove(archivo_zip_encontrado) # Borramos el zip limpio
        else:
            print("❌ Error: El zip no tenía ningún .csv dentro.")
            sys.exit()

except Exception as e:
    print(f"❌ Error al procesar zip: {e}")
    sys.exit()

# 4. EJECUTAR PIPELINE DE IA
print("\n--- 4. Ejecutando crear_ia.py ---")
if os.system("python crear_ia.py") != 0:
    print("❌ Error en crear_ia.py")
    sys.exit()

print("\n--- 5. Ejecutando entrenar_ia.py ---")
if os.system("python entrenar_ia.py") != 0:
    print("❌ Error en entrenar_ia.py")
    sys.exit()

print("\n")
print("==================================================")
print("       🎉 ¡SISTEMA ACTUALIZADO AUTOMÁTICAMENTE! 🎉")
print("==================================================")