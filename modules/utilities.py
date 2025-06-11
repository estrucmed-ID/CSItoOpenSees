"""
@author: Daniela Novoa, Orlando Arroyo, Frank Vidales
@owner: EstrucMed Ingeniería especializada S.A.S


"""     

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
import shutil
from functools import wraps
import random
import json
import datetime


#%% ==> FUNCION CREAR CARPETA DE RESULTADOS

class create_folders:
    
    def __init__(self, main_path):
        self.main_path = main_path
        self.output_folder = os.path.abspath(os.path.join(main_path, 'outputs'))
        self.folder_names = [
            "modal_information",
            "columns_reactios",
            "FHE_analysis_results",
            "pushover_analysis_results",
            "pickles_information"
            ]
        
#%% ==> LEER EXCEL DEL MODELO CSI

class import_csi_info:
    
    def __init__(self, main_path):
        self.main_path = main_path
        self.data_folder = os.path.abspath(os.path.join(main_path, 'data', 'inputs', 'models'))
        self.tables_name = {
            'TABLE:  "OBJECTS AND ELEMENTS - JOINTS"':'Objects and Elements - Joints',
            'TABLE:  "OBJECTS AND ELEMENTS - SHELLS"':'Objects and Elements - Shells',
            'TABLE:  "OBJECTS AND ELEMENTS - FRAMES"':'Objects and Elements - Frames',
            'TABLE:  "JOINT ASSIGNMENTS - RESTRAINTS"':'Joint Assignments - Restraints',
            'TABLE:  "MASS SUMMARY BY DIAPHRAGM"':'Mass Summary by Diaphragm',
            'TABLE:  "MATERIAL PROPERTIES - CONCRETE"':'Material Properties - Concrete',
            'TABLE:  "SHELL SECTIONS - SLAB"':'Shell Sections - Slab',
            'TABLE:  "SHELL SECTIONS - WALL"':'Shell Sections - Wall',
            'TABLE:  "FRAME SECTIONS"':'Frame Sections',    
            'TABLE:  "PIER SECTION PROPERTIES"':'Pier Section Properties',  
            'TABLE:  "CONCRETE COLUMN REBAR DATA"':'Concrete Column Rebar Data',
            'TABLE:  "CONCRETE BEAM REBAR DATA"':'Concrete Beam Rebar Data',
            'TABLE:  "SHEAR WALL PIER SUMMARY - ACI 3"':'Shear Wall Pier Summary - ACI 3',
            'TABLE:  "SHELL ASSIGNMENTS - SECTIONS"':'Shell Assignments - Sections',
            'TABLE:  "SHELL ASSIGNMENTS - PIER SPANDR"':'Shell Assignments - Pier Spandr',
            'TABLE:  "FRAME ASSIGNMENTS - SECTIONS"':'Frame Assignments - Sections',
            'TABLE:  "FRAME ASSIGNMENTS - LOCAL AXES"':'Frame Assignments - Local Axes',
            'TABLE:  "TRIBUTARY AREA AND LLFR"':'Tributary Area and LLRF',
            'TABLE:  "SHELL LOADS - UNIFORM"':'Shell Loads - Uniform',
            'TABLE:  "FRAME LOADS - DISTRIBUTED"':'Frame Loads - Distributed',
            'TABLE:  "COLUMN FORCES"':'Column Forces',
            'TABLE:  "PIER FORCES"':'Pier Forces',
            'TABLE:  "RS FUNCTION - COLOMBIA NSR-10"':'RS Function - Colombia NSR-10',
            'TABLE:  "DIAPGRAGM CENTER OF MASS DISPLACEMENT"':'Diaphragm Center of Mass Displa',
            'TABLE:  "MODAL PARTICIPATING MASS RATIOS"':'Modal Participating Mass Ratios',
            'TABLE:  "FRAME ASSIGNMENTS - OFFSETS"':'Frame Assignments - Offsets'
            }
        
    def import_tables(self, input_name):
        
        excel_path = os.path.join(self.data_folder, f'{input_name}.xlsx')
        print('\n')
        
        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names
        
        data = {}
        
        indices = list(self.tables_name.keys())
        valores = list(self.tables_name.values())
        nombre_tablas = list(zip(indices, valores))
        
        for index in tqdm(range(len(nombre_tablas)),desc='Importando info CSI', unit='table', ncols = 75):
            if valores[index] in sheet_names:
                data[indices[index]] = pd.read_excel(excel_path,sheet_name=valores[index],
                                                   skiprows=1).drop(index=0)
                # df = pd.read_excel(excel_path, sheet_name=valores[index], skiprows=2)
                # data[indices[index]] = df.to_dict(orient='records')  # ✅ conversión segura
            else:
                data[indices[index]] = "Table Not Found"
        
        return data
    
#%% ==> OTRAS FUNCIONES UTILES

def modify_element_label_as_int(label):
    """Converts element label to an integer for consistency."""
    
    if '-' in label:
        if "00:00:00" in label:
            try:
                fecha = pd.to_datetime(label)
                new_label =  f"{fecha.day}-{fecha.month}"
                base, suffix = new_label.split('-')
                return int(f"{base}{suffix.zfill(5)}")  # Convert to integer, then modify suffix
            except ValueError:
                pass
        else:
            base, suffix = label.split('-')
            return int(f"{base}{suffix.zfill(5)}")  # Convert to integer, then modify suffix
        
    return int(float(label))  # Make sure that all the data is a integer

def offset_direction(A, B):
    import numpy as np
    # Definir vector del eje local X (eje axial)
    # siempre sera el vector unitario del nodo I al nodo J
    AB = B-A
    vect_offset = AB/np.linalg.norm(AB)
    
    return vect_offset

def moveODBInfo(origen,destino):
    
    # Mover todo el contenido
    for nombre_archivo in os.listdir(origen):
        origen_archivo = os.path.join(origen, nombre_archivo)
        destino_archivo = os.path.join(destino, nombre_archivo)
        shutil.move(origen_archivo, destino_archivo)
    
    # Cuando todo el contenido se ha movido, eliminar la carpeta vacía
    os.rmdir(origen)
    
    return tqdm.write("🔹 Comenzando análisis pushover...")

def mostrar_progreso(etiqueta):
    def decorador(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tqdm.write(f"🔹 {etiqueta}...")
            return func(*args, **kwargs)
        return wrapper
    return decorador

def generar_color_aleatorio():
    while True:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        if color.lower() not in ['#000000', '#75b72a', '#ffffff']:
            return color
        
def import_csi_data(main_path, input_name, folder_data):
    Import_csi_info = import_csi_info(main_path)
    data_file = Import_csi_info.import_tables(input_name)
    conver_to_json(data_file, folder_data, 'model_information')


def process_json_data(folder_data, file_name):
    output_path = os.path.join(folder_data, f'{file_name}.json')
    with open(output_path, 'r') as f:
        data_dict = json.load(f)
        
    restored_data = {
        k: pd.DataFrame(v) if isinstance(v, list) and all(isinstance(row, dict) for row in v) else v
        for k, v in data_dict.items()
    }
    
    return restored_data

def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    else:
        return obj
    
def conver_to_json(data,folder_data,file_name):
    
    # # Convertir a JSON (string) por cada DataFrame    
    # serializable_data = {k: safe_convert(v) for k, v in data.items()}
    
    # # Guardar como archivo .json
    # output_path = os.path.join(folder_data, f'{file_name}.json')
    
    # with open(output_path, 'w') as f:
    #     json.dump(serializable_data, f, indent=2, ensure_ascii=False) # default=convert_numpy)
        
    serializable_data = make_serializable(data)
    output_path = os.path.join(folder_data, f'{file_name}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
    # # Asegurar que los datos son serializables
    # safe_data = make_json_safe(data)
    
    # # Guardar como archivo .json
    # output_path = os.path.join(folder_data, f'{file_name}.json')
    
    # with open(output_path, 'w') as f:
    #     json.dump(safe_data, f, indent=2, ensure_ascii=False)
        
def safe_convert(v):
        if isinstance(v, pd.DataFrame):
            return make_json_safe(v.to_dict(orient='records'))
        elif isinstance(v, str) and v == "Table Not Found":
            return "Table Not Found"
        elif isinstance(v, go.Figure):
            return v.to_plotly_json()
        else:
            return make_json_safe(v)
            
        
def make_json_safe(obj, seen=None):

    # if seen is None:
    #     seen = set()
    # obj_id = id(obj)

    # if obj_id in seen:
    #     return None  # Evita referencias circulares

    # seen.add(obj_id)

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v, seen) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i, seen) for i in obj]
    elif isinstance(obj, tuple):
        return [make_json_safe(i, seen) for i in obj]
    elif isinstance(obj, set):
        return [make_json_safe(i, seen) for i in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return None
    elif isinstance(obj, (datetime.datetime, datetime.date, pd.Timestamp)):
        return str(obj)  # <-- Aquí resolvemos tu error actual
    else:
        return str(obj)  # fallback a string para cualquier otro tipo raro

def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    return obj

def cargar_figura_3d(path_json, figure:str):
    with open(path_json, 'r') as f:
        data = json.load(f)
    
    fig_dict = data[figure]   
    fig = go.Figure(fig_dict)  # convierte el diccionario en figura Plotly
    return fig

def fema_records_json(folder_records, folder_data):
    
    # Incrementos de tiempo por registro
    time_increment = [
        0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,
        0.005,0.005,0.01,0.01,0.01,0.01,0.005,0.005,
        0.05,0.05,0.02,0.02,0.0025,0.0025,0.005,0.005,
        0.005,0.005,0.02,0.02,0.005,0.005,0.01,0.01,
        0.02,0.02,0.005,0.005,0.005,0.005,0.01,0.01,
        0.005,0.005
    ]

    fema_records_json = {}
    txt_files = sorted([f for f in os.listdir(folder_records) if f.endswith('.txt')])
    
   
    
    for i, filename in enumerate(txt_files):
        file_path = os.path.join(folder_records, filename)
        
        # Leer los valores del archivo
        with open(file_path, 'r') as f:
            data = f.read()

        # Procesar: separar en números
        record_values = [float(val) for val in data.replace('\n', ' ').split() if val.strip()]
        
        # Solo metadatos, no cargamos los valores
        # Leemos los valores de cada registro para registrar los pasos del registro
        fema_records_json[i] = {
            'record_name': f'GM{i+1:02}',
            'file_path': file_path.replace('\\', '/'),  # ruta en formato seguro
            'time_increment': time_increment[i],
            'steps_number': len(record_values)
        }

    # Guardar como archivo .json
    output_path = os.path.join(folder_data, 'fema_records_database.json')
    with open(output_path, 'w') as f:
        json.dump(fema_records_json, f, indent=2)

# folder_records = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'temporary', 'fema_records'))
# folder_data = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'inputs', 'records'))
# fema_records_json(folder_records, folder_data)