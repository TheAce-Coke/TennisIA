import os
import zipfile
import sys
import time

# --- INICIO: CONFIGURACIÓN PARA GITHUB ACTIONS ---
# Esta parte es nueva. Comprueba si estamos en la nube.
KAGGLE_SECRET = os.environ.get("KAGGLE_JSON")

if KAGGLE_SECRET:
    print("--- 1. Detectado entorno GitHub Actions. Configurando Kaggle... ---")
    # Si el secreto existe, lo escribimos en el archivo .json
    # que la librería 'kaggle' espera encontrar.
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
        f.write(KAGGLE_SECRET)
    
    # Damos permisos de lectura/escritura solo al propietario
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
    print("✅ Credenciales de Kaggle configuradas en la nube.")
else:
    print("--- 1. Detectado entorno Local (PC). Usando kaggle.json local. ---")
# --- FIN: CONFIGURACIÓN PARA GITHUB ACTIONS ---


# --- CONFIGURACIÓN ---
DATASET = "dissfya/atp-tennis-2000-2023daily-pull"
ARCHIVO_FINAL = "atp_tennis.csv" 

print("==================================================")
print("   🤖 ACTUALIZADOR AUTOMÁTICO (VERSIÓN NUBE/LOCAL) 🤖")
print("==================================================")

# 2. DESCARGAR DESDE KAGGLE
print("\n--- 2. Descargando datos frescos desde Kaggle... ---")
comando = f"kaggle datasets download -d {DATASET} --force"
codigo = os.system(comando)

if codigo != 0:
    print("❌ Error al descargar.")
    sys.exit()
else:
    print("✅ Descarga completada.")
    time.sleep(2) 

# 3. BUSCAR EL ZIP AUTOMÁTICAMENTE
print("\n--- 3. Buscando el archivo ZIP descargado... ---")
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

# 4. DESCOMPRIMIR Y PREPARAR
print("\n--- 4. Descomprimiendo... ---")
try:
    with zipfile.ZipFile(archivo_zip_encontrado, 'r') as zip_ref:
        zip_ref.extractall(".")
        archivos_extraidos = zip_ref.namelist()
        
        csv_encontrado = None
        for archivo in archivos_extraidos:
            if archivo.endswith(".csv"):
                csv_encontrado = archivo
                break
        
        if csv_encontrado:
            print(f"   -> Archivo extraído: {csv_encontrado}")
            if csv_encontrado != ARCHIVO_FINAL:
                if os.path.exists(ARCHIVO_FINAL):
                    os.remove(ARCHIVO_FINAL) 
                os.rename(csv_encontrado, ARCHIVO_FINAL)
                print(f"✅ Renombrado a: {ARCHIVO_FINAL}")
            else:
                print(f"✅ El archivo ya tiene el nombre correcto ({ARCHIVO_FINAL}).")

            zip_ref.close()
            os.remove(archivo_zip_encontrado)
        else:
            print("❌ Error: El zip no tenía ningún .csv dentro.")
            sys.exit()

except Exception as e:
    print(f"❌ Error al procesar zip: {e}")
    sys.exit()

# 5. EJECUTAR PIPELINE DE IA
print("\n--- 5. Ejecutando crear_ia.py ---")
if os.system("python crear_ia.py") != 0:
    print("❌ Error en crear_ia.py")
    sys.exit()

print("\n--- 6. Ejecutando entrenar_ia.py ---")
if os.system("python entrenar_ia.py") != 0:
    print("❌ Error en entrenar_ia.py")
    sys.exit()

print("\n")
print("==================================================")
print("       🎉 ¡SISTEMA ACTUALIZADO AUTOMÁTICAMENTE! 🎉")
print("==================================================")