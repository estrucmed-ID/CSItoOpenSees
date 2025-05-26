"""
@author: Daniela Novoa, Orlando Arroyo, Frank Vidales
@owner: EstrucMed Ingeniería especializada S.A.S

-------------------------------------------------------------------------------
      Verificación de Modelo Elástico - Comparación OpenSeesPy vs. ETABS
                             Versión: ETABS17 v17.0.1

    Módulo de verificación estructural que compara el modelo generado en 
    OpenSeesPy con el modelo original de ETABS. Esta rutina permite validar 
    la fidelidad del modelo elástico convertido antes de aplicar análisis 
    avanzados.

Se ejecutan tres tipos de verificación fundamentales:

    1. **Análisis Modal**  
       Compara los periodos de vibración y masas participativas modales entre 
       ambos modelos para asegurar una representación dinámica coherente.

    2. **Reacciones en la Base**  
       Evalúa las reacciones verticales en columnas, permitiendo validar 
       la correcta asignación de cargas gravitacionales en el modelo OpenSees.

    3. **Derivas de Entre Piso - Fuerza Horizontal Equivalente (FHE)**  
       Compara las derivas de entre piso entre ambos modelos bajo una 
       distribución de cargas tipo FHE, verificando la consistencia del 
       comportamiento lateral elástico.

Este módulo permite una validación rápida, precisa y automatizada del modelo 
elástico construido en OpenSeesPy a partir del modelo de ETABS.

Unidades: Sistema Internacional (SI) - metros (m)
Norma utilizada: NSR-10 (para la generación de carga lateral con FHE)
"""     

#%% ==> IMPORTAR LIBRERIAS 

# Ubicar los modulos personalizados usados en el script
import sys
import os
from tqdm import tqdm
import time  

service_path = os.getcwd()
modules_path = os.path.abspath(os.path.join(service_path, '..', '..', 'modules'))
output_path = os.path.abspath(os.path.join(service_path, '..', '..', 'outputs'))
main_path_default = os.path.abspath(os.path.join(service_path, '..', '..'))
sys.path.append(modules_path)
sys.path.append(main_path_default)

# Importar módulos personalizados
import utilities as ut
import archetype as ach
import modal_analysis as mdl
import reaction_comparison as rct
import drift_verification as fhe

#%% ==> PARAMETROS DE ENTRADA

project_name = 'PUERTO SERENO PLATAFORMA A'
load_case = '(0) 1CM + 0.25CV'
n_pint = 5
shell_craking = 1
ed_type = 'DMO'

#%% ==> COMENZAR A CONTAR TIEMPO DE EJECUCION
stime = time.time()

#%% ==> IMPORTAR DATA PROCESADA
folder_data = os.path.join(main_path_default, 'data', 'temporary', 'json')
data_file = ut.process_json_data(folder_data, 'model_information')

#%% ==> GENERAR EL MODELO ELASTICO (MODELO SIN PROCESAMIENTO DE DATA)
# Llamar a la clase principal
elastic_model = ach.ElasticModelBuilder(data_file)
data_processfile = ut.process_json_data(folder_data, 'elastic_model_information')
# Generar el modelo
elastic_model.genElasticModel(data_processfile)
# Analisis modal -->
modal_results = elastic_model.modal_analysis(elastic_model)

#%% ==> GENERAR EL ANALISIS MODAL
def run_modal_analysis(main_path = None):
    
    if main_path is None:
        main_path = main_path_default
        
    tqdm.write(" 🔄 Calculando y comparando resultados modales")
    modal_analysis_res = mdl.modal_analysis(elastic_model, main_path, modal_results)
    tqdm.write(" ✅​ Comparación completada ")
    
    return modal_analysis_res

#%% ==> CALCULAR REACCIONES EN LA BASE DE LAS COLUMNAS
def run_base_column_reactions(main_path = None):
    
    if main_path is None:
        main_path = main_path_default
    
    tqdm.write(" 🔄 Calculando y comparando reacciones en la base de las columnas")
    rct.basecolumnreactions_class(elastic_model, main_path, load_case)
    tqdm.write(" ✅​ Comparación completada ")

#%% ==> APLICAR EL METODO FHE
def run_fhe_method(modal_analysis_res, main_path = None):
    
    if main_path is None:
        main_path = main_path_default
    
    tqdm.write(" 🔄 Calculando y comparando derivas de entre piso con el método FHE")
    fhe.fhemethod_class(elastic_model, modal_analysis_res, main_path, data_processfile)
    tqdm.write(" ✅​ Comparación completada ")

#%% ==> FUNCION PRINCIPAL
def main():
    print('\nCSI TO OPENSEES - GENERADOR DE VERIFICACIONES')
    print('\n▶️ Inicio de procesamiento')
    modal_analysis_res = run_modal_analysis()
    run_base_column_reactions()
    run_fhe_method(modal_analysis_res)
    etime = time.time()
    print(f"⏹️ Proceso completado en {(etime - stime)/60:.2f} minutos.")

#%% ==> EJECUTAR
if __name__ == "__main__":
    main()