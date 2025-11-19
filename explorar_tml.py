import pandas as pd
import requests
import io

# Apuntamos al archivo de 2024 que vimos en tu lista
URL_TEST = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/2024.csv"

print(f"--- Explorando TML (2024): {URL_TEST} ---")

try:
    r = requests.get(URL_TEST)
    
    if r.status_code != 200:
        print(f"❌ Error de descarga: {r.status_code}")
        exit()

    print("✅ Archivo descargado. Intentando leer columnas...")
    
    # Probamos leerlo. A veces usan comas, a veces punto y coma.
    csv_data = io.StringIO(r.text)
    
    try:
        # Intento 1: Coma normal
        df = pd.read_csv(csv_data, nrows=5)
        
        # Si sale todo en 1 columna, es que usaban otro separador
        if len(df.columns) < 2:
            csv_data.seek(0)
            df = pd.read_csv(csv_data, sep=';', nrows=5)
            
    except:
        # Intento 2: Forzar punto y coma
        csv_data.seek(0)
        df = pd.read_csv(csv_data, sep=';', nrows=5)

    print("\n👇 COPIA ESTA LISTA Y PÉGAMELA 👇")
    print("===================================")
    print(list(df.columns))
    print("===================================")

except Exception as e:
    print(f"Error grave: {e}")