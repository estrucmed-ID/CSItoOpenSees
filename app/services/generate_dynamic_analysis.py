"""
@author: Daniela Novoa, Orlando Arroyo, Frank Vidales
@owner: EstrucMed Ingeniería especializada S.A.S

-------------------------------------------------------------------------------
                           Análisis Dinámico Bidireccional
                        Ejecución paralela con registros sísmicos
                             Versión: ETABS17 v17.0.1

    Módulo para realizar análisis dinámicos no lineales bidireccionales del 
    modelo estructural en OpenSeesPy, utilizando acelerogramas del conjunto 
    de registros del FEMA. El análisis se ejecuta en paralelo para mejorar 
    la eficiencia computacional.

Características principales:

    • Aplicación simultánea de aceleraciones en las direcciones X y Y.
    • Uso de múltiples registros sísmicos con control de duración y paso de tiempo.
    • Ejecución paralela.
    • Control del uso de CPU con `threadpool_limits` para restringir el número 
      de hilos utilizados por librerías como pandas o numpy, evitando saturación 
      total de la CPU y posibles pérdidas de rendimiento por sobrecarga.

Ventajas:

    - Ahorro significativo de tiempo al procesar varios registros en paralelo.
    - Evita el uso excesivo de recursos computacionales.
    - Facilita la obtención de desplazamientos, derivas y fuerzas internas bajo 
      excitación sísmica realista.

Salidas principales:
    
    - Historias de desplazamiento, aceleración y fuerzas internas por nodo y elemento.
    - Derivas de entre piso por dirección.
    - Registros temporales para análisis de desempeño estructural.

Unidades: Sistema Internacional (SI) - metros (m), segundos (s)
Norma utilizada: NSR-10 (con base en criterios de selección y escalamiento FEMA P695)
-------------------------------------------------------------------------------
"""     

#%% ==> IMPORTAR LIBRERIAS 

# Ubicar los modulos personalizados usados en el script
import sys
import os
from tqdm import tqdm
import json
import multiprocessing
import time
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
import joblib as jb


service_path = os.getcwd()
modules_path = os.path.abspath(os.path.join(service_path, '..', '..', 'modules'))
main_path_default = os.path.abspath(os.path.join(service_path, '..', '..'))
database_path = os.path.abspath(os.path.join(service_path, '..', '..', 'data', 'inputs', 'records'))
outputs_path = os.path.abspath(os.path.join(service_path, '..', '..', 'outputs', 'dynamic_analysis_results'))
sys.path.append(modules_path)
sys.path.append(main_path_default)
sys.path.append(database_path)

# Importar módulos personalizados
import utilities as ut
import dynamic_analysis as dyn
import archetype as ach

#%% ==> COMENZAR A CONTAR TIEMPO DE EJECUCION
stime = time.time()

#%% ==> IMPORTAR DATA PROCESADA
folder_data = os.path.join(main_path_default, 'data', 'temporary', 'json')
data_file = ut.process_json_data(folder_data, 'model_information')

#%% ==> GENERAR EL MODELO INELASTICO (MODELO SIN PROCESAMIENTO DE DATA)
# Llamar a la clase principal
nonlinear_model = ach.NONLinearModelBuilder(data_file)
data_processfile = ut.process_json_data(folder_data, 'nonlinear_model_information')

#%% ==> ANALISIS DINAMICO
def run_dynamic_analysis(outputs_path, main_path=None):
    if main_path is None:
        main_path = main_path_default

    # --- Cargar los registros del FEMA
    json_path = os.path.join(database_path, 'fema_records_database.json')
    with open(json_path, 'r') as f:
        fema_records = json.load(f)

    # --- Filtrar registros en parejas
    indices = [2*i + 1 for i in range(len(fema_records) // 2)]
    records_filtered = [fema_records[str(i)] for i in indices]

    num_cores = multiprocessing.cpu_count() - 1

    with threadpool_limits(limits=1):  # Limitar hilos internos (pandas, numpy, etc..)
        resultados = Parallel(n_jobs=num_cores)(
            delayed(dyn.rundyn)(ind, data_file, nonlinear_model, data_processfile, fema_records)
            for ind in tqdm(range(len(records_filtered)), desc='Procesando análisis dinámico', unit='registro', ncols=75)
        )

    # Guardar resultados
    output_file = os.path.join(outputs_path, "model_linear_barev2")
    jb.dump(resultados, output_file)

#%% ==> EJECUTAR
if __name__ == "__main__":
    print('\nCSI TO OPENSEES - GENERADOR DE ANALISIS DINAMICO')
    print('\n▶️ Inicio de procesamiento')
    run_dynamic_analysis(outputs_path)
    etime = time.time()
    print(f"⏹️ Proceso completado en {(etime - stime)/60:.2f} minutos.")