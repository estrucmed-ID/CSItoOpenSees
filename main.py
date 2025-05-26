"""
@author: Daniela Novoa, Orlando Arroyo, Frank Vidales
@owner: EstrucMed Ingeniería especializada S.A.S

-------------------------------------------------------------------------------
                      CSI Model Converter to OpenSeesPy
                          Versión: ETABS17 v17.0.1
                            
    Módulo principal para convertir modelos CSI a OpenSeesPy. Genera el 
        esquema estructural del modelo SCI para analisis posteriores

Este script contiene las primeras funciones:
    
    1. Carga de informacion del modelo CSI
    2. Convertidor CSI to OpenSees
    3. Consolidacion final de resultados
    
Unidades: Sistema Internacional (SI) - metros (m)
Norma utilizada: NSR-10.
-------------------------------------------------------------------------------

"""     
#%% ==> IMPORTAR LIBRERIAS 

# Ubicar los modulos personalizados usados en el script
import sys
import os
import time  
import pandas as pd

# Paths principales
main_path = os.getcwd()
modules_path = os.path.join(main_path, 'modules')
services_path = os.path.join(main_path, 'app', 'services')
outputs_path = os.path.join(main_path, 'outputs', 'opensees_models')

# Añadir rutas
sys.path.append(modules_path)
sys.path.append(services_path)

# Importar servicios
import utilities as ut
import archetype as ach

#%% ==> PARAMETROS DE ENTRADA

project_name = 'PUERTO SERENO PLATAFORMA A'
load_case = '(0) 1CM + 0.25CV'
n_pint = 5
shell_craking = 1
ed_type = 'DMO'

#%% ==> COMENZAR A CONTAR TIEMPO DE EJECUCION
stime = time.time()

#%% ==> IMPORTAR DATA
# Esto genera un json para no sobreescribir la data
folder_data = os.path.join(main_path, 'data', 'temporary', 'json')
ut.import_csi_data(main_path, 'input_model', folder_data)

#%% ==> INICIALIZAR DICCIONARIO PARA ALMACENAR RESULTADOS
results_dict = {}

#%% ==> GENERAR AQUETIPO ELASTICO

def create_and_save_elastic_model(folder_data):
    
    data_file = ut.process_json_data(folder_data, 'model_information')
    elastic_model = ach.ElasticModelBuilder(data_file)
    
    data_dict = elastic_model.genInitialElasticModel(main_path, model_type = 'EM', 
                load_case = load_case, npint = n_pint, shell_craking = shell_craking)
    
    ut.conver_to_json(data_dict,folder_data,'elastic_model_information')
    
    # Obtener resultados de analisis modal
    modal_results = elastic_model.modal_analysis(elastic_model)
    results_dict['modal_results_EM'] = elastic_model.modal_results_table(modal_results)
    # Obtener vista en 3D del modelo
    results_dict['fig_plot_model'] = elastic_model.plot_model()
    # Obtener la carga distribuida en las vigas
    results_dict['beam_loads']= data_dict.get('TABLE:  "DISTRIBUTED LOAD - BEAMS"')[['Story', 'Beam Label', 'Element Label', 'Floor Labels Matched',
                                                                                      'Wcl', 'Wcv', 'Wv', 'Distributed force']]
    results_dict['beam_loads'].rename(columns = {'Wcl':'WCL [kN]', 'Wcv':'WCV [kN]', 'Wv':'WV [kN]', 'Distributed force': 'WVT [kN]'}, inplace = True)
    results_dict['fig_forces'] = elastic_model.frame_responses_plot()
    # Obtener fuerzas de las columnas
    results_dict['column_forces_EM'] = elastic_model.column_forces_table(data_dict.get('TABLE:  "FRAME GENERATION"'), data_dict.get('load_case'))
    # Obtener la deformada 
    results_dict['fig_nodal_responses'], results_dict['fig_frame_responses'] = elastic_model.nodal_responses_plot()
    
#%% ==> GENERAR AQUETIPO INELASTICO

def create_and_save_nonlinear_model(folder_data):
    
    data_file = ut.process_json_data(folder_data, 'model_information')
    nonlinear_model = ach.NONLinearModelBuilder(data_file)
    
    data_dict = nonlinear_model.genInitialNONLinearModel(main_path, model_type = 'NLM', 
                load_case = load_case, npint = n_pint, shell_craking = shell_craking, ed_type = ed_type)

    ut.conver_to_json(data_dict,folder_data,'nonlinear_model_information')
    
    # Obtener resultados de analisis modal
    modal_results = nonlinear_model.modal_analysis(nonlinear_model)
    results_dict['modal_results_NLM'] = nonlinear_model.modal_results_table(modal_results)
    
    # Obtener fuerzas de las columnas
    results_dict['column_forces_NLM'] = nonlinear_model.column_forces_table(data_dict.get('TABLE:  "FRAME GENERATION"'), data_dict.get('load_case'))

def dataframe_modifications(df_em,df_nlm):
    df_em_renamed = df_em.rename(columns={
    'P [kN]': 'P_EM [kN]',
    'V2 [kN]': 'V2_EM [kN]',
    'V3 [kN]': 'V3_EM [kN]',
    'T [kN-m]': 'T_EM [kN-m]',
    'M2 [kN-m]': 'M2_EM [kN-m]',
    'M3 [kN-m]': 'M3_EM [kN-m]'
    })
    
    df_nlm_renamed = df_nlm.rename(columns={
        'P [kN]': 'P_NLM [kN]',
        'V2 [kN]': 'V2_NLM [kN]',
        'V3 [kN]': 'V3_NLM [kN]',
        'T [kN-m]': 'T_NLM [kN-m]',
        'M2 [kN-m]': 'M2_NLM [kN-m]',
        'M3 [kN-m]': 'M3_NLM [kN-m]'
    })
    
    df_merged = pd.merge(
        df_em_renamed,
        df_nlm_renamed,
        on=['Story', 'Object Label', 'Element Label', 'Load Case'],
        how='inner'  # puedes usar 'outer' si quieres conservar todo aunque no coincidan
    )
    
    def calc_diff_percent(a, b):
        return 100 * (b - a) / a if a != 0 else 0  # evita división por cero
    
    df_merged['%Δ P'] = (df_merged['P_NLM [kN]'] - df_merged['P_EM [kN]']) / df_merged['P_EM [kN]'] * 100
    df_merged['%Δ V2'] = (df_merged['V2_NLM [kN]'] - df_merged['V2_EM [kN]']) / df_merged['V2_EM [kN]'] * 100
    df_merged['%Δ V3'] = (df_merged['V3_NLM [kN]'] - df_merged['V3_EM [kN]']) / df_merged['V3_EM [kN]'] * 100
    df_merged['%Δ T']  = (df_merged['T_NLM [kN-m]'] - df_merged['T_EM [kN-m]']) / df_merged['T_EM [kN-m]'] * 100
    df_merged['%Δ M2'] = (df_merged['M2_NLM [kN-m]'] - df_merged['M2_EM [kN-m]']) / df_merged['M2_EM [kN-m]'] * 100
    df_merged['%Δ M3'] = (df_merged['M3_NLM [kN-m]'] - df_merged['M3_EM [kN-m]']) / df_merged['M3_EM [kN-m]'] * 100
    
    return df_merged

#%% ==> FLUJO PRINCIPAL
def main():
    print('\nCSI TO OPENSEES - GENERADOR DE ARQUETIPO LINEAL Y NO LINEAL')
    print('\n▶️ Inicio de procesamiento')
    create_and_save_elastic_model(folder_data)
    create_and_save_nonlinear_model(folder_data)
    etime = time.time()
    print(f"⏹️ Proceso completado en {(etime - stime)/60:.2f} minutos.")
    
#%% ==> EJECUTAR
folder_report = os.path.join(main_path, 'outputs', 'reports')
if __name__ == "__main__":
    main()
    results_dict['column_forces'] = dataframe_modifications(results_dict['column_forces_EM'], 
                                                            results_dict['column_forces_NLM'])
    
    ut.conver_to_json(results_dict,folder_report,'archetype_results_report')

