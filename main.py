# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import logging
import random

# ¡NUEVO! Importamos el middleware de CORS
from fastapi.middleware.cors import CORSMiddleware

from alg_nsga2 import preparar_datos_para_algoritmo, alg_NSGA2

logging.basicConfig(level=logging.INFO)

class Node(BaseModel):
    id: str
    lat: float
    lng: float
    demanda: int = 0

class OptimizeRequest(BaseModel):
    nodes: List[Node]
    vehicleCapacity: int
    timeWindow: Tuple[str, str]

app = FastAPI(
    title="API de Optimización de Rutas",
    description="Usa un algoritmo genético NSGA-II para optimizar rutas de recolección."
)

# --- ¡NUEVO! AÑADIMOS LA CONFIGURACIÓN DE CORS ---
# Definimos qué orígenes permitimos. El asterisco "*" significa "cualquiera".
# Esto es perfecto para desarrollo local.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"], # Permitir todas las cabeceras
)
# --- FIN DE LA CONFIGURACIÓN DE CORS ---


@app.get("/")
def read_root():
    return {"status": "API de optimización funcionando"}


@app.post("/optimize")
async def optimize_route(request_data: OptimizeRequest):
    # ... (el resto de tu función se queda exactamente igual)
    logging.info("Petición recibida en /optimize.")
    try:
        cuerpo = request_data.dict()
        nodos = cuerpo.get('nodes')
        capacidad_vehiculo = cuerpo.get('vehicleCapacity')
        ventana_tiempo = cuerpo.get('timeWindow')

        logging.info("Preparando datos para el algoritmo...")
        numN, numC, capacidad, coordenadas, horario = preparar_datos_para_algoritmo(
            nodos, capacidad_vehiculo, ventana_tiempo
        )

        logging.info("Ejecutando el algoritmo NSGA-II... Esto puede tardar.")
        mejor_solucion = alg_NSGA2(numN, numC, coordenadas, horario, capacidad)
        logging.info("Algoritmo finalizado.")

        pareto_front_json = []
        coordenadas_nodos_dict = {nodo['id']: (nodo['lat'], nodo['lng']) for nodo in nodos}
        poblacion_final = [mejor_solucion]

        for i, individuo in enumerate(poblacion_final):
            rutas_serializadas = []
            if hasattr(individuo, 'solucion') and individuo.solucion:
                for camion in individuo.solucion:
                    trayecto = camion[0]
                    rutas_serializadas.append([(p[0][0], p[0][1]) for p in trayecto])

            pareto_front_json.append({
                "id": f"sol-{i+1}",
                "label": chr(65 + i),
                "distancia": individuo.evaluacion_trayecto,
                "carga_total": individuo.evaluacion_carga,
                "tiempo": individuo.evaluacion_tiempo,
                "routes_serializadas": rutas_serializadas
            })

        respuesta = {
            "ejemplarId": f"DOCKER-VRP-{random.randint(1000, 9999)}",
            "paretoFront": pareto_front_json,
            "coordenadas_nodos": coordenadas_nodos_dict
        }
        
        logging.info("Respuesta generada exitosamente.")
        return respuesta

    except Exception as e:
        logging.error(f"Ha ocurrido un error durante la ejecución: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")