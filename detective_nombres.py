import pandas as pd

ARCHIVO = "atp_tennis.csv"

print(f"--- BUSCANDO EN {ARCHIVO} ---")

try:
    # Cargamos solo las columnas de nombres para ir rápido
    df = pd.read_csv(ARCHIVO, usecols=['Player_1', 'Player_2'])
    
    print(f"Base de datos cargada: {len(df)} partidos.")
    
    while True:
        texto = input("\nEscribe parte del nombre (ej: Gima): ").strip().lower()
        if texto == "salir": break
        
        # Buscamos en ambas columnas
        # .str.contains(..., case=False) busca sin importar mayúsculas/minúsculas
        resultados_1 = df[df['Player_1'].str.lower().str.contains(texto, na=False)]['Player_1'].unique()
        resultados_2 = df[df['Player_2'].str.lower().str.contains(texto, na=False)]['Player_2'].unique()
        
        # Unimos y limpiamos
        todos = set(list(resultados_1) + list(resultados_2))
        
        if todos:
            print(f"\n✅ HE ENCONTRADO ESTOS NOMBRES QUE CONTIENEN '{texto}':")
            print("------------------------------------------------")
            for nombre in todos:
                print(f" -> {nombre}")
            print("------------------------------------------------")
            print("Prueba a buscar EXACTAMENTE ese nombre en tu web.")
        else:
            print(f"❌ No he encontrado nada con '{texto}'.")
            print("Si no sale aquí, es que Tennis My Life (la fuente de datos) aún no ha subido sus partidos.")

except FileNotFoundError:
    print("Error: No encuentro el archivo 'atp_tennis.csv'.")