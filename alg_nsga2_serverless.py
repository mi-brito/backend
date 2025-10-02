# --- Imports ---
import numpy as np
import matplotlib
# --- Data Preparation ---
def preparar_datos_para_algoritmo(nodos, capacidad_vehiculo, ventana_tiempo):
    num_nodos = len(nodos)
    num_camiones = 2  # TODO: make configurable
    capacidad = capacidad_vehiculo
    coordenadas = [ [(n['lat'], n['lng']), n.get('demanda', 0)] for n in nodos ]
    horario = ventana_tiempo
    return num_nodos, num_camiones, capacidad, coordenadas, horario

# --- Utility Functions ---
def distancia_euclidiana(x, y):
    distancia_x = x[1] - x[0]
    distancia_y = y[1] - y[0]
    return math.sqrt(distancia_x**2 + distancia_y**2)

def dividir_horas(intervalos, horario):
    hora_inicio_dt = datetime.strptime(horario[0], "%H:%M:%S")
    hora_fin_dt = datetime.strptime(horario[1], "%H:%M:%S")
    duracion = hora_fin_dt - hora_inicio_dt
    intervalo_duracion = duracion / intervalos
    periodos = []
    inicio_intervalo = hora_inicio_dt
    for _ in range(intervalos):
        fin_intervalo = inicio_intervalo + intervalo_duracion
        periodos.append((inicio_intervalo.time(), fin_intervalo.time()))
        inicio_intervalo = fin_intervalo
    return periodos

def generar_hora_aleatoria(horario, hora_final):
    hora_inicio_dt = datetime.strptime(str(horario[0]).split(".")[0], "%H:%M:%S")
    hora_fin_dt = datetime.strptime(hora_final, "%H:%M:%S")
    diferencia = hora_fin_dt - hora_inicio_dt
    total_segundos = diferencia.total_seconds()

# --- Imports ---
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import math
import random
from datetime import datetime, time, timedelta
from collections import namedtuple

# --- Data Preparation ---
def preparar_datos_para_algoritmo(nodos, capacidad_vehiculo, ventana_tiempo):
    num_nodos = len(nodos)
    num_camiones = 2  # TODO: make configurable
    capacidad = capacidad_vehiculo
    coordenadas = [ [(n['lat'], n['lng']), n.get('demanda', 0)] for n in nodos ]
    horario = ventana_tiempo
    return num_nodos, num_camiones, capacidad, coordenadas, horario

# --- NSGA-II Core ---
    segundos_aleatorios = random.randint(0, int(total_segundos))
    hora_aleatoria = (hora_inicio_dt + timedelta(seconds=segundos_aleatorios)).time()
    return str(hora_aleatoria)

def limpiar_lista(lista):
    return [sublista for sublista in lista if sublista]

# --- Evaluation Functions ---
def evaluar_recorrido(trayecto):
    trafico = 0
    num_residuos = 0
    i = 1
    while i < len(trayecto):
        trafico += distancia_euclidiana(trayecto[i-1][0], trayecto[i][0])
        num_residuos += trayecto[i][1]
        i += 1
    return trafico, num_residuos

def evaluar_camion(camion, capacidad, penalizacion=1000):
    total_de_trafico, total_residuos = evaluar_recorrido(camion[0])
    evaluacion = total_de_trafico
    evaluacion_trafico = total_de_trafico
    evaluacion_carga = total_residuos
    if total_residuos > capacidad:
        evaluacion += evaluacion * penalizacion
        evaluacion_carga += evaluacion_carga * penalizacion
    horario = camion[1]
    hora_de_termino = camion[2]
    hora_dt = time.fromisoformat(hora_de_termino)
    evaluacion_tiempo = 1
    if hora_dt > horario[1]:
        evaluacion += evaluacion * penalizacion
        evaluacion_tiempo += evaluacion_tiempo * penalizacion
    return evaluacion, evaluacion_trafico, evaluacion_carga, evaluacion_tiempo

def evaluar_individuo(individuo, capacidad):
    eval_ind = 0
    eval_trafico = 0
    eval_carga = 0
    eval_tiempo = 0
    for i in individuo[0]:
        eval, eval_Tr, eval_C, eval_Ti = evaluar_camion(i, capacidad)
        eval_ind += eval
        eval_trafico += eval_Tr
        eval_carga += eval_C
        eval_tiempo += eval_Ti
    individuo = individuo._replace(evaluacion=eval_ind, evaluacion_trayecto=eval_trafico, evaluacion_carga=eval_carga, evaluacion_tiempo=eval_tiempo)
    return individuo

def evaluar_poblacion(poblacion, capacidad):
    for j, i in enumerate(poblacion):
        poblacion[j] = evaluar_individuo(i, capacidad)

# --- NSGA-II Core ---

Individuo = namedtuple('Individuo', ['solucion', 'evaluacion', 'evaluacion_trayecto', 'evaluacion_carga', 'evaluacion_tiempo', 'dominacion', 'distancia'])

def genera_poblacion(num_nodos, num_camiones, coordenadas, horario):
    poblacion = []
    deposito = coordenadas[0]
    for _ in range(100):
        coordenadas_agregadas = []
        solucion = []
        periodos = dividir_horas(num_camiones, horario)
        nodos = 1
        camion = 1
        while camion <= num_camiones:
            dimension = num_nodos-1
            datos_camion = []
            trayecto = []
            num_contenedores = 0
            trayecto.append(deposito)
            coordenadas_agregadas.append(deposito)
            if dimension > 0:
                num_contenedores = random.randint(1, dimension)
            contenedores = 1
            while contenedores <= num_contenedores:
                coordenada = random.choice(coordenadas)
                if coordenada not in coordenadas_agregadas:
                    trayecto.append(coordenada)
                    coordenadas_agregadas.append(coordenada)
                    nodos += 1
                contenedores += 1
            if camion == num_camiones and nodos < num_nodos:
                coordenadas_faltantes = [coordenada for coordenada in coordenadas if coordenada not in coordenadas_agregadas]
                for coordenada in coordenadas_faltantes:
                    trayecto.append(coordenada)
                    nodos += 1
            trayecto.append(deposito)
            datos_camion.append(trayecto)
            datos_camion.append(periodos[camion-1])
            hora_que_finalizo = generar_hora_aleatoria(periodos[camion-1], horario[1])
            datos_camion.append(hora_que_finalizo)
            solucion.append(datos_camion)
            dimension = dimension - contenedores
            camion += 1
        individuo = Individuo(solucion, 0, 0, 0, 0, [0,None], 0)
        poblacion.append(individuo)
    return poblacion

def alg_NSGA2(num_nodos, num_camiones, coordenadas, horario, capacidad):
    poblacion = genera_poblacion(int(num_nodos), int(num_camiones), coordenadas, horario)
    evaluar_poblacion(poblacion, int(capacidad))
    return poblacion

# --- Evaluation Functions ---
# TODO: migrate evaluar_individuo, evaluar_poblacion, etc.

# --- Graph Generation & Upload ---
def generar_graficas_y_subir(final_population, blob_service_client, container_name):
    # Generate optimization progress graph (mejor, peor, promedio)
    urls = {}
    try:
        blob_service_client.create_container(container_name)
    except Exception:
        pass  # Container may already exist

    # Collect stats from final_population
    y_mejor = [ind.evaluacion_trayecto for ind in final_population]
    y_peor = [ind.evaluacion_carga for ind in final_population]
    y_promedio = [ind.evaluacion_tiempo for ind in final_population]
    x_generacion = np.linspace(0, len(final_population), len(final_population))

    # Graph 1: Optimization progress
    plt.figure()
    plt.plot(x_generacion, y_mejor, label='Mejor')
    plt.plot(x_generacion, y_peor, label='Peor')
    plt.plot(x_generacion, y_promedio, label='Promedio')
    plt.title('NSGA-II con la optimización de rutas')
    plt.xlabel('Generación')
    plt.ylabel('Aptitud')
    plt.legend(loc='upper right')
    plt.xlim(1, len(final_population))
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format='png')
    plt.close()
    buffer1.seek(0)
    nombre_blob1 = 'optimizacion_progreso.png'
    blob_client1 = blob_service_client.get_blob_client(container=container_name, blob=nombre_blob1)
    blob_client1.upload_blob(buffer1, overwrite=True)
    urls['optimizacion_progreso'] = blob_client1.url

    # Graph 2: Final solutions (carga vs distancia)
    plt.figure()
    plt.scatter(y_peor, y_mejor, color='blue', marker='o')
    plt.title('Evaluación de la carga y distancia de las soluciones')
    plt.xlabel('Carga')
    plt.ylabel('Distancia')
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format='png')
    plt.close()
    buffer2.seek(0)
    nombre_blob2 = 'soluciones_finales.png'
    blob_client2 = blob_service_client.get_blob_client(container=container_name, blob=nombre_blob2)
    blob_client2.upload_blob(buffer2, overwrite=True)
    urls['soluciones_finales'] = blob_client2.url

    return urls

# --- Utility Functions ---
# TODO: migrate helper functions (distancia_euclidiana, mutacion, etc.)

# --- Main entrypoint for Azure Function ---
def run_nsga2(input_data):
    nodos = input_data.get('nodes')
    capacidad_vehiculo = input_data.get('vehicleCapacity')
    ventana_tiempo = input_data.get('timeWindow')
    num_nodos, num_camiones, capacidad, coordenadas, horario = preparar_datos_para_algoritmo(nodos, capacidad_vehiculo, ventana_tiempo)
    poblacion_final = alg_NSGA2(num_nodos, num_camiones, coordenadas, horario, capacidad)

    # Serialize population for frontend
    pareto_front_json = []
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

    coordenadas_nodos_dict = {nodo['id']: (nodo['lat'], nodo['lng']) for nodo in nodos}

    # If Azure Blob client is available, generate graphs and upload
    urls_imagenes = {}
    if 'blob_service_client' in input_data and 'container_name' in input_data:
        urls_imagenes = generar_graficas_y_subir(poblacion_final, input_data['blob_service_client'], input_data['container_name'])

    result = {
        "ejemplarId": f"AZURE-VRP-001",
        "paretoFront": pareto_front_json,
        "coordenadas_nodos": coordenadas_nodos_dict,
        "insightImages": urls_imagenes
    }
    return result
