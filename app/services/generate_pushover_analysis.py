"""
@author: Daniela Novoa, Orlando Arroyo, Frank Vidales
@owner: EstrucMed Ingeniería especializada S.A.S

-------------------------------------------------------------------------------
                           Análisis Pushover Bidireccional
                      Evaluación no lineal del modelo en OpenSeesPy
                             Versión: ETABS17 v17.0.1

    Módulo encargado de ejecutar el análisis estático no lineal (pushover) 
    del modelo estructural generado en OpenSeesPy. El análisis se realiza 
    de forma independiente en ambas direcciones principales: X y Y.

Este análisis permite evaluar la capacidad sísmica del sistema estructural, 
obtener la curva de capacidad fuerza-desplazamiento y verificar mecanismos 
de colapso bajo cargas laterales crecientes.

Se incluyen:

    • Definición del patrón de carga lateral tipo fuerza horizontal equivalente (FHE).
    • Aplicación incremental de la carga hasta alcanzar el colapso o la condición de control.
    • Registro de la curva de capacidad en X y Y para posterior evaluación.

Salidas principales:
    
    - Curvas de capacidad (fuerza base vs. desplazamiento de control)
    - Identificación del desplazamiento máximo alcanzado
    - Registro de esfuerzos y derivas durante el análisis

El análisis pushover se configura siguiendo los criterios de la norma NSR-10 
y recomendaciones internacionales para evaluación sísmica.

Unidades: Sistema Internacional (SI) - metros (m)
Norma utilizada: NSR-10
-------------------------------------------------------------------------------
"""     

#%% ==> IMPORTAR LIBRERIAS 

# Ubicar los modulos personalizados usados en el script
import sys
import os
from tqdm import tqdm
import time
from multiprocessing import Process

service_path = os.getcwd()
modules_path = os.path.abspath(os.path.join(service_path, '..', '..', 'modules'))
main_path_default = os.path.abspath(os.path.join(service_path, '..', '..'))
sys.path.append(modules_path)
sys.path.append(main_path_default)

# Importar módulos personalizados
import utilities as ut
import archetype as ach
import pushover as push 


#%% ==> COMENZAR A CONTAR TIEMPO DE EJECUCION
stime = time.time()

#%% ==> IMPORTAR DATA PROCESADA
folder_data = os.path.join(main_path_default, 'data', 'temporary', 'json')
data_file = ut.process_json_data(folder_data, 'model_information')

#%% ==> GENERAR EL MODELO INELASTICO (MODELO SIN PROCESAMIENTO DE DATA)
# Llamar a la clase principal
nonlinear_model = ach.NONLinearModelBuilder(data_file)
data_processfile = ut.process_json_data(folder_data, 'nonlinear_model_information')

#%% ===> ANALISIS PUSHOVER INDIVIDUAL 

def run_pushover_direction(direction, tag, idLoadPattern, main_path, data_processfile):
    nonlinear_model.genNONLinearModel(data_processfile)
    nonlinear_model.modal_analysis(nonlinear_model)
    push.pushoverclass(nonlinear_model, main_path, direction, tag, idLoadPattern)

#%% ==> EJECUTAR

if __name__ == "__main__":
    print('\nCSI TO OPENSEES - GENERADOR DE ANALISIS PUSHOVER')
    print('\n▶️ Inicio de procesamiento')
    
    p1 = Process(target=run_pushover_direction, args=('X', 2010, 3, main_path_default, data_processfile))
    p2 = Process(target=run_pushover_direction, args=('Y', 2011, 4, main_path_default, data_processfile))
    
    tqdm.write(" 🔄 Generando análisis pushover en dirección X ... ")
    p1.start()
    tqdm.write(" 🔄 Generando análisis pushover en dirección Y ... ")
    p2.start()
    
    
    p1.join()
    tqdm.write(" ✅​ Análisis completado en dirección X ") 
    p2.join()
    tqdm.write(" ✅​ Análisis completado en dirección Y")
    
    etime = time.time()
    print(f"⏹️ Proceso completado en {(etime - stime)/60:.2f} minutos.")
