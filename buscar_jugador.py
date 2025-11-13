import pandas as pd

ARCHIVO_DATOS = 'atp_tennis.csv'

try:
    # Cargamos solo la columna de ganadores y perdedores para ir rápido
    df = pd.read_csv(ARCHIVO_DATOS, usecols=['Player_1', 'Player_2'])
    
    # Unimos todos los nombres en una lista única
    nombres_unicos = pd.concat([df['Player_1'], df['Player_2']]).unique()
    
    # Convertimos a un DataFrame para buscar fácil
    df_nombres = pd.DataFrame(nombres_unicos, columns=['nombre'])
    # Ordenamos alfabéticamente
    df_nombres = df_nombres.sort_values(by='nombre')
    
    print(f"--- Base de datos cargada: {len(df_nombres)} jugadores encontrados ---")

    while True:
        busqueda = input("\nEscribe parte del nombre (o 'salir'): ").lower()
        if busqueda == 'salir': break
        
        # Filtramos los que contienen el texto escrito
        resultados = df_nombres[df_nombres['nombre'].str.lower().str.contains(busqueda)]
        
        if len(resultados) > 0:
            print("\nHe encontrado estos jugadores:")
            # Mostramos los primeros 10 resultados
            print(resultados['nombre'].head(10).to_string(index=False))
            if len(resultados) > 10:
                print(f"... y {len(resultados)-10} más.")
        else:
            print("❌ No he encontrado a nadie con ese nombre.")

except FileNotFoundError:
    print("Error: No encuentro 'atp_tennis.csv'. Ejecuta el actualizador primero.")