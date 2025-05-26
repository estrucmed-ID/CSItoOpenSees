"""
@author: Daniela Novoa, Orlando Arroyo, Frank Vidales
@owner: EstrucMed Ingeniería especializada S.A.S

-------------------------------------------------------------------------------
                         Módulo de Análisis Dinámico Individual
               Generación del análisis no lineal para un solo modelo
-------------------------------------------------------------------------------

Este módulo contiene la función encargada de ejecutar el análisis dinámico 
bidireccional de un solo modelo estructural en OpenSeesPy.

Es utilizado como unidad de ejecución dentro del servicio de análisis paralelo, 
y recibe como entrada el modelo ya definido junto con el par de acelerogramas 
correspondientes a X y Y.

Características:
    • Aplicación simultánea de registros sísmicos en ambas direcciones.
    • Definición del patrón de carga dinámico en función del tiempo.
    • Registro de historias de desplazamiento, aceleración y derivas por piso.
    • Compatible con análisis no lineales con integración explícita.

Este módulo puede ser encapsulado dentro de scripts de ejecución masiva de registros.

Unidades: Sistema Internacional (SI) - metros (m), segundos (s)

"""     
import openseespy.opensees as ops
import sys
import os
import numpy as np
import pandas as pd
import opseestools.analisis3D as an

modules_path = os.getcwd()
main_path = os.path.abspath(os.path.join(modules_path, '..'))
database_path = os.path.abspath(os.path.join(modules_path, '..', 'data', 'inputs', 'records'))
outputs_path = os.path.abspath(os.path.join(modules_path, '..', 'outputs', 'dynamic_analysis_results'))
sys.path.append(database_path)


#%% --- Función individual para cada análisis dinámico
def rundyn(ind, data_file, nonlinear_model, data_processfile, fema_records):
    
    nonlinear_model.genNONLinearModel(data_processfile)

    an.gravedad()
    ops.loadConst('-time', 0.0)

    # Obtener registros consecutivos para análisis bidireccional
    record_1 = fema_records[str(2 * ind)]
    record_2 = fema_records[str(2 * ind + 1)]
    
    record_paths = [record_1['file_path'],record_2['file_path']]
    dt = record_1['time_increment'] # Debe ser el mismo dt para ambos registros
    nsteps = min(record_1['steps_number'], record_2['steps_number']) 

    df_joints = nonlinear_model.OE_joints
    coordz = np.sort(pd.unique(df_joints['Global Z']))
    altur = coordz[1::]
    dnodes = [100000000 * i for i in range(len(altur))]
    
    nodes_base = df_joints[df_joints['Story']=='Base'].reset_index()
    nodes_control = [int(nodes_base['Element Label'][0])]
    nodes_control.extend(dnodes)

    # Análisis dinamico usando directamente valores de cada registro
    tiempo, techo, techo2, techoT, node_disp, _, node_acel, node_disp2, node_acel2, forces, driftX, driftY = an.dinamicoBD3(
        record_paths, dt, nsteps, dt, 9.81, 0.025, dnodes[-1], 1, nodes_control, [1], eletype='frame'
    )

    return tiempo, techo, techo2, node_acel, node_acel2, forces, driftX, driftY
