"""
@author: Daniela Novoa, Orlando Arroyo, Frank Vidales
@owner: EstrucMed Ingeniería especializada S.A.S


"""     

#%% ==> IMPORTAR LIBRERIAS 

import os
import dash
import sys
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from datetime import datetime
import dash_table
import json
import plotly.graph_objects as go
import pandas as pd

# dash_path = os.getcwd()
dash_path = os.path.dirname(os.path.abspath(__file__))
main_path_default = os.path.abspath(os.path.join(dash_path, '..'))
report_path = os.path.abspath(os.path.join(main_path_default, 'outputs', 'reports'))
modules_path = os.path.join(main_path_default, 'modules')
# Añadir rutas
sys.path.append(modules_path)

#%% ==> CARGAR RESULTADOS DE TODOS LOS GENERADORES
    
path_json = os.path.join(report_path, 'archetype_results_report.json')

with open(path_json, 'r') as f:
    data = json.load(f)

#%% ==> DASHBOARD MAIN CLASS

class DashBoardReport:
    
    def __init__(self):
        
        # ---------------------------------------------------------------------
        # Leer data del main 
        # ---------------------------------------------------------------------
        
        self.fig = go.Figure(data['fig_plot_model'])
        self.df_modal_em = pd.DataFrame(data['modal_results_EM'])
        self.df_modal_nlm = pd.DataFrame(data['modal_results_NLM'])
        
        beam_loads = data['beam_loads']
        for row in beam_loads:
            for k, v in row.items():
                if isinstance(v, (dict, list)):
                    row[k] = str(v)
        self.df_cargas_vigas = pd.DataFrame(beam_loads)
        self.df_fuerzas = pd.DataFrame(data['column_forces'])
        self.fig_nodal_responses = go.Figure(data['fig_nodal_responses'])
        self.fig_frame_responses = go.Figure(data['fig_frame_responses'])
        self.fig_forces = go.Figure(data['fig_forces'])
        
        # -----> Generar fecha
        self.generate_date = datetime.today().strftime("%d/%m/%Y")
        
        # -----> Iniciar la app Dash 
        self.app = dash.Dash(
            
            __name__, 
            suppress_callback_exceptions=True,
            external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"]
            )
        
        self.header = self.header_fun()   # Header
        self.sidebar = self.sidebar_fun() # Navbar
        
        self.app_layout()                 # Main para el header y navbar
        self.render_page()                # Rutas del navbar
        
        # -----> Funciones especificas
        self.contenido_comparacion_modelos()  # Actualizar contenido del resumen
        
        if __name__ == '__main__':
            import webbrowser
            webbrowser.open("http://127.0.0.1:8051")
            self.app.run(debug=True, port=8051)
            
    def app_layout(self):
        
        self.app.layout = html.Div([
            dcc.Location(id='url', refresh=False),
            self.header,
            self.sidebar,
            html.Div(id='page-content', style={"backgroundColor": "white"})
        ], style={"backgroundColor": "#E7E6E6"})
        
        
    def render_page(self):
        @self.app.callback(
            Output('page-content', 'children'),
            Input('url', 'pathname')
            )
        
        def render_page_content(pathname):
            if pathname == "/":
                return LayoutHome.layout_home()
            elif pathname == "/resumen":
                return self.layout_resumen()
            # elif pathname == "/verificaciones":
            #     return layout_verificaciones()
            # elif pathname == "/resultados":
            #     return layout_resultados()
            # elif pathname == "/exportar":
            #     return layout_exportar()
            return html.Div("404 Página no encontrada")
    
    def layout_resumen(self):
        return html.Div([
            LayoutSummary.layout_resumen_descripcion_modelos(),
            LayoutSummary.layout_resumen_vista3D(self),
            LayoutSummary.layout_comparaciones(),
            LayoutSummary.layout_resumen_vista3D_forces(self),
            LayoutSummary.layout_resumen_vista3D_nodal_responses(self),
            LayoutSummary.layout_resumen_vista3D_frame_responses(self)
        ])
 
    def contenido_comparacion_modelos(self):
        @self.app.callback(
            Output("contenido-comparacion-modelos", "children"),
            Input("dropdown-comparacion-modelos", "value")
        )
        
        def actualizar_contenido(tipo):
            if tipo == "periodos":
                return LayoutSummary.contenido_periodos_modales(self)
            if tipo == "cargas":
                return LayoutSummary.contenido_cargas_vigas(self)
            if tipo == "fuerzas":
                return LayoutSummary.contenido_fuerzas_elementos(self)
     
        
    def header_fun(self):
        
        header = dbc.Container([
            dbc.Row([
                # Columna izquierda: logos (EstrucMed + CSItoOpenSees)
                dbc.Col([
                    html.Img(src="/assets/estrucmed.png", height="60px", style={"marginLeft": "25px"}),
                    html.Img(src="/assets/line.png", height="60px", style={"marginLeft": "30px"}),
                    html.Img(src="/assets/csi_to_opensees.png", height="60px", style={"marginLeft": "30px"})            
                ], width="auto", style={"display": "flex", "alignItems": "center"}),

                # Columna derecha: Nombre del proyecto + fecha de generacion
                dbc.Col([
                    html.Div([
                        html.H4(
                            "REPORTE DE RESULTADOS",
                            style={
                                "fontSize": "35px",       
                                "color": "white",         
                                "fontWeight": "bold",     
                                "marginBottom": "0px"     
                            }
                        ),
                        html.H6(
                            "VELVET-PLATAFORMA",
                            style={
                                "fontSize": "25px",
                                "color": "white",       
                                "fontWeight": "bold"     
                            }
                        ),
                        html.Div(f"Fecha de generación : {self.generate_date}", 
                                 style={"fontSize": "0.9em", 
                                        "color": "white", 
                                        "fontWeight": "bold"})
                    ], style={"textAlign": "center"})
                ], width=6),
            ], align="center", className="py-2")
            
        ], fluid=True, style={
                        "backgroundColor": "#75B72A",
                        "borderRadius": "0 0 0 35px",
                        "padding": "15px",
                        "position": "fixed",          
                        "top": "0",
                        "left": "0",
                        "right": "0",
                        "zIndex": "1000",             
                        "height": "150px"             
                    })
        return header
                              
    def sidebar_fun(self):
        
        sidebar = html.Div([
            dbc.Nav([
                dbc.NavLink(html.I(className="bi bi-house-door", style={"fontSize": "1.5rem"}), href="/", id="tab-home", active="exact", style={"marginTop": "10px", "marginBottom": "10px"}),
                dbc.NavLink(html.I(className="bi bi-diagram-3", style={"fontSize": "1.5rem"}), href="/resumen", id="tab-resumen", active="exact", style={"marginTop": "10px", "marginBottom": "10px"}),
                dbc.NavLink(html.I(className="bi bi-exclamation-octagon", style={"fontSize": "1.5rem"}), href="/verificaciones", id="tab-verificaciones", active="exact", style={"marginTop": "10px", "marginBottom": "10px"}),
                dbc.NavLink(html.I(className="bi bi-graph-up", style={"fontSize": "1.5rem"}), href="/resultados", id="tab-resultados", active="exact", style={"marginTop": "10px", "marginBottom": "10px"}),
                dbc.NavLink(html.I(className="bi bi-download", style={"fontSize": "1.5rem"}), href="/exportar", id="tab-exportar", active="exact", style={"marginTop": "10px", "marginBottom": "10px"}),
            ], vertical=True, pills=True, justified=True)
        ], style={
            "position": "fixed",
            "top": "125px",
            "left": "0",
            "bottom": "0",
            "width": "90px",
            "paddingTop": "180px",  # Para que no se cruce con el header
            "backgroundColor": "#E7E6E6",
            "textAlign": "center"
        })
        
        return sidebar

#%% ==> LAYOUT HOME CLASS
class LayoutHome(DashBoardReport):
    
    def layout_home():
        return dbc.Container([
            html.H2("Bienvenido al Reporte de Resultados", className="titulo-header"),
            
            html.H5("Reporte de la conversión del modelo estructural desde CSI a OpenSees.", 
                    className="mb-4 text-muted"),
            
            html.P(
                "El presente reporte documenta la comparación de resultados entre un modelo estructural tridimensional de "
                "pórticos resistentes a momento en concreto reforzado (PRMCR) desarrollado en ETABS y un modelo equivalente "
                "implementado en OpenSeesPy mediante el uso de la herramienta de conversión 'CSI to OpenSees'.",
                className="lead"
            ),
    
            html.P(
                "Esta plataforma, basada en Python, facilita la transferencia de datos entre ambos programas con el propósito "
                "de que los usuarios puedan realizar análisis avanzados en OpenSeesPy, aprovechando su flexibilidad para simulaciones "
                "no lineales y personalización en Python.",
                className="lead"
            ),  
            
            html.P(
                "El objetivo principal del reporte es verificar la fidelidad del modelo generado en OpenSeesPy respecto al "
                "modelo original en ETABS, asegurando que las diferencias observadas entre los resultados de ambos  "
                "modelos sean comprensibles y aceptables dentro del marco de las simplificaciones o ajustes realizados. "
                "Este análisis permitirá determinar si el modelo original de ETABS es adecuado o requiere modificaciones "
                "adicionales antes de continuar con simulaciones avanzadas en OpenSeesPy.",
                className="lead"
            ),  
            
            html.P(
                "Los resultados presentados en este documento incluyen: ",
                className="lead"
            ),  
            
            html.Ul([
                html.Li("📊  Sección 1: Resumen del modelo"),
                html.Li("✅  Sección 2: Verificaciones de periodos/masas modales, reacciones en la base de las columnas y derivas de entre piso"),
                html.Li("📈  Sección 3: Resultados de análisis pushover y dinámico"),
                html.Li("📄  Sección 4: Exportación del reporte"),
            ], style={"marginTop": "30px", "fontSize": "1.2em"}, className='lead'),
    
            html.Hr(),
    
            html.P("Puedes usar la barra lateral para moverte entre secciones.", className="lead")
        ], style={"padding": "60px", "backgroundColor": "white", 'marginTop':'150px'})

#%% ==> LAYOUT SUMMARY CLASS
class LayoutSummary(DashBoardReport):
    
    def layout_resumen_descripcion_modelos():
        return dbc.Container([
    
            html.H2("Resumen de los modelos generados", className="titulo-header"),
            
            html.P([
                "En esta sección del reporte se presenta una visión integral del modelo estructural generado. ",
                "El objetivo es que el usuario pueda verificar su correcta configuración, visualizar sus propiedades principales, ",
                "y comparar de forma clara los resultados entre los modelos Elástico (EM) e Inelástico (NLM)."
                
            ], className="lead"),
            
            html.Ol([
                html.Li([
                    html.B("Descripción general de los modelos:"), " se resumen las características principales del modelo elástico y no lineal, ",
                    "incluyendo sus configuraciones de materiales, elementos, transformación geométrica, aplicación de cargas y definición de masas."
                ]),
                html.Li([
                    html.B("Vista 3D del modelo:"), " visualización espacial del edificio para revisar que todos los elementos hayan sido modelados correctamente. ",
                    "Permite identificar errores como losas faltantes, elementos sueltos o geometría mal conectada."
                ]),
                html.Li([
                    html.B("Comparación entre modelos EM y NLM:"), " tablas detalladas comparan resultados clave entre ambos modelos: ",
                    "periodos modales, masas participativas, cargas distribuidas en vigas y fuerzas internas en columnas. ",
                    "En cada caso se analizan las diferencias entre modelos incluso cuando las cargas son equivalentes."
                ]),
                html.Li([
                    html.B("Visualización de cargas asignadas:"), " gráfico en 3D que muestra las cargas aplicadas a vigas y columnas, ",
                    "permitiendo verificar consistencia con el modelo original de ETABS."
                ]),
                html.Li([
                    html.B("Deformada 3D del modelo:"), " animación o vista estática que representa la deformada del modelo bajo el análisis gravitacional. ",
                    "Permite validar compatibilidad estructural, continuidad de diafragmas y deformaciones esperadas."
                ]),
                html.Li([
                    html.B("Respuesta interna de elementos (Momento Mz):"), " mapa 3D de esfuerzos internos en las vigas y columnas, particularmente el momento flector My. ",
                    "Esta vista permite comprobar la distribución de fuerzas bajo cargas gravitacionales."
                ])
            ], className="lead"),
                        
            html.Br(),
            
            html.H4("Descripción general de los modelos", 
                    className="mb-4 text-muted"),
    
            dbc.Row([
    
                # Modelo Elástico
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Modelo Elástico", className="mb-4 text-muted"),
                        dbc.CardBody([
                            html.P("El modelo elástico está diseñado para representar el comportamiento estructural lineal de la edificación. Su configuración incluye:", className="lead"),
    
                            html.Ul([
                                html.Li("Factor de agrietamiento por defecto: 1 (ajustable por el usuario)."),
                                html.Li("Secciones de losas modeladas con ElasticMembranePlateSection, módulo de elasticidad 4700·√f'c."),
                                html.Li("Vigas y columnas con materiales elásticos (`uniaxialMaterial Elastic`)."),
                                html.Li("Elementos `elasticBeamColumn` con integración Lobatto (5 puntos por defecto)."),
                                html.Li("Transformación geométrica: Lineal en vigas, PDelta en columnas."),
                                html.Li("Offsets en columnas definidos por la altura de la viga."),
                                html.Li("Asignación de masas en X, Y e inercia en Z (otros grados con masa cero)."),
                                html.Li("Aplicación de diafragmas rígidos en cada entrepiso."),
                                html.Li("Cargas gravitacionales asignadas con combinación: 1CM + 0.25CV."),
                                html.Li("Restricciones nodales importadas directamente desde ETABS.")
                            ], className="lead")
                        ])
                    ])
                ], width=6),
    
                # Modelo Inelástico
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Modelo Inelástico", className="mb-4 text-muted"),
                        dbc.CardBody([
                            html.P("El modelo inelástico incorpora materiales no lineales y secciones de fibras para capturar el comportamiento realista bajo cargas sísmicas o de daño acumulado. Sus características son:", className="lead"),
    
                            html.Ul([
                                html.Li("Factor de agrietamiento por defecto: 1 (ajustable por el usuario)."),
                                html.Li("Secciones de losas con ElasticMembranePlateSection (como en elástico)."),
                                html.Li("Uso de materiales no lineales: concreto confinado, sin confinar y acero."),
                                html.Li("Elementos `forceBeamColumn` con secciones de fibras e integración Lobatto (5 puntos)."),
                                html.Li("Transformación geométrica: Lineal en vigas, PDelta en columnas."),
                                html.Li("Offsets en columnas definidos por la altura de la viga."),
                                html.Li("Asignación de masas en X, Y e inercia en Z (otros grados con masa cero)."),
                                html.Li("Aplicación de diafragmas rígidos en cada entrepiso."),
                                html.Li("Cargas gravitacionales asignadas con combinación: 1CM + 0.25CV."),
                                html.Li("Restricciones nodales importadas directamente desde ETABS."),
                                html.Br()
                            ], className="lead")
                        ])
                    ])
                ], width=6)
    
            ], className="g-4"),  # g-4 = espacio horizontal entre columnas
            html.Hr(),
    
        ],  style={"padding": "60px", 'marginTop':'150px'})
    
    
    def layout_resumen_vista3D(self):
        
        return dbc.Container([
            
            html.H4("Vista 3D del modelo", className="mb-4 text-muted", style = {'padding':'0px 0px 0px 50px'}),
            dcc.Graph(figure = self.fig, style={'height': '700px'}),
            html.Hr(),
            html.Br(),
            html.Br()
            
        ])
    
    
    def layout_comparaciones():
        return dbc.Container([
    
            html.H4("Comparación entre Modelos Elástico e Inelástico", 
                    className="mb-4 text-muted"),
    
            # Dropdown selector
            dbc.Row([
                dbc.Col([
                    html.Label("Selecciona el tipo de comparación:", className="lead"),
                    dcc.Dropdown(
                        id="dropdown-comparacion-modelos",
                        options=[
                            {"label": "Periodos modales", "value": "periodos"},
                            {"label": "Cargas en vigas", "value": "cargas"},
                            {"label": "Fuerzas en elementos", "value": "fuerzas"}
                        ],
                        value="periodos",
                        clearable=False
                    )
                ], width=6)
            ], className="lead"),
            html.Br(),
    
            # Contenedor que cambia con base en la opción seleccionada
            html.Div(id="contenido-comparacion-modelos")
    
        ], style = {'padding':'0px 0px 0px 50px'}) #style={"padding": "40px"}
    
    def contenido_periodos_modales(self):
        df_modal_em = self.df_modal_em.round(3) 
        df_modal_nlm = self.df_modal_nlm.round(3) 

        return dbc.Container([
            html.H5("Comparación de Periodos Modales", className="container-title"),
    
            # Descripción del comando eigen
            html.Div([
                html.H6("Método de cálculo modal", className="container-subtitle"),
                html.P([
                    "El análisis modal se realiza usando el comando ",
                    html.Code("eigen"), ", el cual resuelve el sistema de autovalores para obtener los periodos, frecuencias naturales y formas modales del modelo estructural.",
                    " La comparación entre modelos elástico (EM) e inelástico (NLM) permite evaluar la influencia de la rigidez y la masa en la respuesta modal."
                ], className="lead")
            ]),
    
            # Tablas en paralelo
            dbc.Row([
                dbc.Col([
                    html.H6("Modelo Elástico (EM)", className="container-subtitle2", style={'textAlign':'center'}),
                    dash_table.DataTable(
                        data=df_modal_em.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in df_modal_em.columns],
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "center"},
                    )
                ], width=6, style={"paddingRight": "30px"}),
    
                dbc.Col([
                    html.H6("Modelo Inelástico (NLM)", className="container-subtitle2", style={'textAlign':'center'}),
                    dash_table.DataTable(
                        data = df_modal_nlm.to_dict('records'),
                        columns = [{"name": i, "id": i} for i in df_modal_nlm.columns],
                        style_table = {"overflowX": "auto"},
                        style_cell = {"textAlign": "center"},
                    )
                ], width=6)
            ])
        ], fluid=True)
    
    def contenido_cargas_vigas(self):
        
        df_cargas_vigas = self.df_cargas_vigas
        
        df_cargas_vigas['WCL [kN]'] = df_cargas_vigas['WCL [kN]'].round(3)
        df_cargas_vigas['WCV [kN]'] = df_cargas_vigas['WCV [kN]'].round(3)
        df_cargas_vigas['WV [kN]'] = df_cargas_vigas['WV [kN]'].round(3)
        df_cargas_vigas['WVT [kN]'] = df_cargas_vigas['WVT [kN]'].round(3)
        
        return dbc.Container([
    
            html.H5("Asignación de cargas en vigas a partir de losas", className="container-title"),
    
            html.P("El procedimiento para calcular las cargas distribuidas en las vigas a partir de la carga aplicada en las losas del modelo ETABS se desarrolló en varias etapas:", className="lead"),
    
            html.H6("1. Identificación de vigas compatibles con los vértices de las losas", className="container-subtitle2"),
            html.P([
                "Se analizaron todas las vigas presentes en el modelo para encontrar aquellas cuyos nodos 'i' y 'j' coincidan con dos vértices consecutivos de cada losa. ",
                "Dado que ETABS fragmenta las vigas al interrumpirse por un 'joint' (columna o cruce de viga), una viga como 'B36' puede aparecer como 'B36-1', 'B36-2', etc. ",
                "Esto dificulta la identificación directa, por lo que se implementaron estrategias para reconstruir la conexión completa."
            ], className="lead"),
    
            html.H6("2. Cálculo del área tributaria por viga", className="container-subtitle2"),
            html.Ul([
                html.Li("Se calculó el área y perímetro de cada losa."),
                html.Li("Se determinó si la losa es de una o dos direcciones, considerando el modelado en ETABS o usando relaciones geométricas."),
                html.Li([
                    "En losas bidireccionales, se aplicó el método de área tributaria irregular:",
                    html.Br(),
                    html.Code("ATv = (AL / PL) × Lv"),
                    html.Br(),
                    "donde AL es el área de la losa, PL su perímetro, y Lv la longitud del segmento de viga."
                ]),
                html.Li("En losas unidireccionales, se asignó el 50% de la franja perpendicular, según el eje de dirección de carga.")
            ], className="lead"),
    
            html.H6("3. Asignación de carga a vigas fragmentadas", className="container-subtitle2"),
            html.P([
                "Se buscaron coincidencias entre los vértices de cada losa y los nodos de las vigas fragmentadas. ",
                "Cuando una coincidencia era parcial (por ejemplo, por interrupción), se identificó la viga completa mediante su nodo inicial y final, y se sumó el área tributaria correspondiente."
            ], className="lead"),
    
            html.H6("4. Cálculo de la carga distribuida total", className="container-subtitle2"),
            html.P("Se calcularon tres componentes por viga:", className="lead"),
            html.Ul([
                html.Li([
                    html.B("Carga transmitida desde la losa:"),
                    html.Br(),
                    html.Code('WCL = (ATv / Lv) × (FCM × (CM + 24·eL) + FCV × CV)'),
                    html.Br(),
                    "donde FCM y FCV son factores de carga muerta y viva, respectivamente, CM y CV es la carga sobre impuesta y viva de la losa, respectivamente, y eL el espesor de la losa"
                ]),
                html.Li([
                    html.B("Peso propio de la viga:"),
                    html.Br(),
                    html.Code("WCV = FCM × (24 × atv)"),
                    html.Br(),
                    "donde atv es el área transversal de la viga"
                ]),
                html.Li([
                    html.B("Carga total en la viga:"),
                    html.Br(),
                    html.Code("WVT = WCL + WCV + WV"),
                    " (si existe una carga WV desde ETABS)"
                    
                ])
            ], className="lead"),
    
            # html.H6("5. Carga muerta equivalente en columnas (para pesos de losas no soportadas por vigas)", className="container-subtitle2"),
            # html.Code("PC = 24 × (Lc × atc)", className="lead"),
    
            html.H5("Cargas distribuidas calculadas en vigas", className="container-subtitle2", style={'textAlign':'center'}),
    
            dash_table.DataTable(
                data=df_cargas_vigas.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df_cargas_vigas.columns],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center"},
            ),
            
            html.Hr(),
            
    
        ], fluid=True)
    
    def contenido_fuerzas_elementos(self):
        df_fuerzas = self.df_fuerzas
        
        def round_values(data_frame):
            column_list = ['P_EM [kN]','V2_EM [kN]','V3_EM [kN]','T_EM [kN-m]','M2_EM [kN-m]','M3_EM [kN-m]']
            column_list.extend(['P_NLM [kN]','V2_NLM [kN]','V3_NLM [kN]','T_NLM [kN-m]','M2_NLM [kN-m]','M3_NLM [kN-m]'])
            column_list.extend(['%Δ P','%Δ V2','%Δ V3','%Δ T','%Δ M2','%Δ M3'])
            
            for key in column_list:
                data_frame[key] = data_frame[key].round(2)
        
        round_values(df_fuerzas)
      
        return dbc.Container([
            
            html.H5("Fuerzas internas en elementos (Columnas)", className="container-title"),

            html.P([
                "Las fuerzas internas mostradas a continuación corresponden a una combinación de carga por defecto: ",
                html.B("1.0CM + 0.25CV"), ". Esta combinación representa el caso gravitacional más exigente dentro del rango de servicio del modelo."
            ], className="lead"),
            
            html.P([
                "Aunque las cargas aplicadas son las mismas para ambos modelos (elástico e inelástico), ",
                "las diferencias en la modelación estructural generan discrepancias en las fuerzas internas. ",
                "Esto se debe principalmente a la representación más detallada del comportamiento material en el modelo no lineal, ",
                "que afecta la rigidez, la redistribución de cargas y el flujo de esfuerzos en la estructura."
            ], className="lead"),
            
            html.P([
                "Es esperable que los esfuerzos axiales, momentos y cortantes presenten variaciones entre ambos modelos, ",
                "incluso cuando la masa y las cargas distribuidas sean equivalentes."
            ], className="lead"),
                
            html.H5("Fuerzas en las columnas por piso", className="container-subtitle2", style={'textAlign':'center'}),
            dash_table.DataTable(
                data=df_fuerzas.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df_fuerzas.columns],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center"},
            ),
            
            html.Hr(),
            
            
        ], fluid=True)
    
    def layout_resumen_vista3D_forces(self):
        
        return dbc.Container([
            html.Br(),
            html.Br(),
            
            html.H4("Cargas asignadas al modelo", className="mb-4 text-muted", style = {'padding':'0px 0px 0px 50px'}),
            dcc.Graph(figure = self.fig_forces, style={'height': '700px'}),
            html.Hr(),
            html.Br(),
            html.Br()
            
        ])
    
    def layout_resumen_vista3D_nodal_responses(self):
        
        return dbc.Container([
            html.Br(),
            html.Br(),
            
            html.H4("Deformada 3D del modelo", className="mb-4 text-muted", style = {'padding':'0px 0px 0px 50px'}),
            dcc.Graph(figure = self.fig_nodal_responses, style={'height': '700px'}),
            html.Hr(),
            html.Br(),
            html.Br()
            
        ])
    
    def layout_resumen_vista3D_frame_responses(self):
        
        return dbc.Container([
            
            html.H4("Respuesta de los elementos", className="mb-4 text-muted", style = {'padding':'0px 0px 0px 50px'}),
            dcc.Graph(figure = self.fig_frame_responses, style={'height': '700px'}),
            html.Hr(),
            html.Br(),
            html.Br()
            
        ])

#%% ==> EJECUTAR
if __name__ == "__main__":
    DashBoardReport()

