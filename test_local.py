# test_local.py

# Importamos las funciones principales de tu algoritmo
from alg_nsga2 import preparar_datos_para_algoritmo, alg_NSGA2
import json # Usaremos json para imprimir el resultado de forma legible

def ejecutar_prueba_local():
    """
    Esta función simula una petición de Postman, pero de forma local.
    """
    print("--- INICIANDO PRUEBA LOCAL DEL ALGORITMO ---")

    # 1. Prepara los datos de prueba (el mismo JSON que usamos en Postman)
    datos_de_prueba = {
      "nodes": [
        {"id": "1", "lat": 82, "lng": 76, "demanda": 0},
        {"id": "2", "lat": 96, "lng": 44, "demanda": 19},
        {"id": "3", "lat": 50, "lng": 5, "demanda": 21},
        {"id": "4", "lat": 49, "lng": 8, "demanda": 6},
        {"id": "5", "lat": 13, "lng": 7, "demanda": 19},
        {"id": "6", "lat": 29, "lng": 89, "demanda": 7},
        {"id": "7", "lat": 58, "lng": 30, "demanda": 12},
        {"id": "8", "lat": 84, "lng": 39, "demanda": 16},
        {"id": "9", "lat": 14, "lng": 24, "demanda": 6},
        {"id": "10", "lat": 2, "lng": 39, "demanda": 16}
      ],
      "vehicleCapacity": 100,
      "timeWindow": ["08:00:00", "18:00:00"]
    }
    print("Datos de entrada listos.")

    try:
        # 2. Llama a tus funciones existentes para preparar y ejecutar el algoritmo
        print("Preparando datos para el algoritmo...")
        numN, numC, capacidad, coordenadas, horario = preparar_datos_para_algoritmo(
            datos_de_prueba['nodes'],
            datos_de_prueba['vehicleCapacity'],
            datos_de_prueba['timeWindow']
        )

        print("Ejecutando alg_NSGA2... Esto puede tardar.")
        mejor_solucion = alg_NSGA2(numN, numC, coordenadas, horario, capacidad)
        print("--- ALGORITMO FINALIZADO CON ÉXITO ---")

        # 3. Muestra el resultado en la terminal
        print("\n--- MEJOR SOLUCIÓN ENCONTRADA ---")
        # Imprimimos la solución de una forma más bonita usando json.dumps
        # (Necesitamos un truco para que no falle con objetos complejos)
        print(json.dumps(mejor_solucion._asdict(), indent=4, default=str))

    except Exception as e:
        print(f"\n--- OCURRIÓ UN ERROR DURANTE LA EJECUCIÓN ---")
        print(f"Error: {e}")
        # Esto te dará más detalles sobre en qué línea falló
        import traceback
        traceback.print_exc()


# --- Esta es la parte que ejecuta la función de prueba ---
if __name__ == "__main__":
    ejecutar_prueba_local()