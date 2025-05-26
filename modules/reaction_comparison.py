"""
@author: Daniela Novoa, Orlando Arroyo, Frank Vidales
@owner: EstrucMed Ingeniería especializada S.A.S

-------------------------------------------------------------------------------
                    Módulo de Comparación de Reacciones en la Base
                  Resultados + Visualización Interactiva (Dashboard)
-------------------------------------------------------------------------------

Este módulo compara las reacciones verticales en la base de las columnas 
entre el modelo ETABS y el modelo OpenSeesPy bajo cargas gravitacionales.

Incluye:
    • Procesamiento de resultados de carga puntual por columna.
    • Comparación gráfica de reacciones por elemento o por piso.
    • Identificación de diferencias críticas mediante umbrales definidos.
    • Reporte interactivo con Plotly + tabla y visualización con `opstool`.

Objetivo:
Verificar que la carga distribuida desde losas y vigas esté correctamente 
transmitida hacia los elementos verticales.

Salidas:
    - Porcentaje de error por columna.
    - Mapa de diferencias por planta.
    - Dashboard visual con resumen de alertas.

Unidades: Sistema Internacional (SI)
"""

import pandas as pd
import openseespy.opensees as ops
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import opseestools.analisis3D as an

class basecolumnreactions_class:
    
    def __init__(self, generate_model, main_path, load_case, threshold=10, num_columns= None):
        
        self.genframes = generate_model.GN_frames
        self.columnforces = generate_model.column_forces
        self.load_case  = load_case
        self.threshold = threshold
        self.ncolumns = num_columns
        
        self.folder_path = os.path.abspath(os.path.join(main_path, 'outputs', 'verification_results'))
        self.folder_temp_path = os.path.abspath(os.path.join(main_path, 'data', 'temporary'))
        
        self.opsforces = []
        self.csiforces = []
        self.data_basecolumn = []
        
        an.gravedad()
        ops.loadConst('-time', 0.0)
        
        self._mainfuntion()
        self._plot_BRverification()
    
    def _mainfuntion(self):
    
        # Story 1 Column Data --->
        Base_Data = self.genframes[(self.genframes['Design Type'] == 'Column') & (self.genframes['Story'] == 'Story1')]
        
        # Columns label--->
        Label_List = Base_Data['Element Label'].astype(int).tolist()
        
        # Column Forces --->
        Column_Response = [ops.eleResponse(label,'globalForce') for label in Label_List]
        
        # Estraer y asignar iformación de Column_Response --->
        Responses_np = np.array(Column_Response) 
        VBasal_x = -np.around(Responses_np[:,0],3)
        VBasal_y = -np.around(Responses_np[:,1],3)
        Axial_Force = -np.around(Responses_np[:,2],3)
        Momento_x = np.around(Responses_np[:,3],3)
        Momento_y = np.around(Responses_np[:,4],3)
        Torcion = -np.around(Responses_np[:,5],3)
        
        # Create DataFrame --->
        df_Forces_OpenSees = pd.DataFrame({
            'Unique Name':Label_List,
            'P(kN)':Axial_Force,
            'V2(kN)':VBasal_x,
            'V3(kN)':VBasal_y,
            'T(kN-m)':Torcion,
            'M2(kN-m)':Momento_x,
            'M3(kN-m)':Momento_y
        })
        
        df_ColumnForcesETABS = self.columnforces
        
        df_ColumnForcesETABS = df_ColumnForcesETABS[df_ColumnForcesETABS['Story'] == 'Story1']
        df_ColumnForcesETABS = df_ColumnForcesETABS[df_ColumnForcesETABS['Load Case/Combo'] == self.load_case]
        df_ColumnForcesETABS = df_ColumnForcesETABS[df_ColumnForcesETABS['Station'] == 0]
        
        # Export to Excel
        
        df_Forces_OpenSees.to_excel(os.path.join(os.path.join(self.folder_path, 'xlsx'), 'opensees_columnforces.xlsx'), index=False)
        df_ColumnForcesETABS.to_excel(os.path.join(os.path.join(self.folder_path, 'xlsx'), 'csi_columnforces.xlsx'), index=False)
        
        self.opsforces = df_Forces_OpenSees
        self.csiforces = df_ColumnForcesETABS
        
        # Incluir en el dataframe de opensses el valor de la fuerza axial de ETABS para comparar
        df_ColumnForcesOP = pd.merge(self.opsforces, 
                                     self.csiforces[['Unique Name','P','Column']],
                                     on='Unique Name', how='inner')
    
        # Calculo de la diferencia porcentual
        df_ColumnForcesOP['DFP'] = np.abs(df_ColumnForcesOP['P']/df_ColumnForcesOP['P(kN)']-1)*100
        # Ordenar diferencia porcentual de mayor a menor
        df_ColumnForcesOPSorted = df_ColumnForcesOP.sort_values(by='DFP', ascending=False)
        # Obtener indice de las primeras 30 columnas con los mayores valores de diferencia porcentual
        index_MaxDFPColumn = [df_ColumnForcesOPSorted.index[i] for i in range(min(30,len(df_ColumnForcesOPSorted)))]
        # Ordenar de menor a mayor el indice de las 30 columnas
        index_MaxDFPColumn.sort()
    
        # Obtener las diferencias porcentuales de las 30 columnas
        diferencias = [df_ColumnForcesOPSorted['DFP'][i] for i in index_MaxDFPColumn]
        # Obtener el label de las 30 columnas
        columnas = [df_ColumnForcesOPSorted['Column'][i] for i in index_MaxDFPColumn]
        # Obtener coordenadas X y Y de las columnas
        df_ColumnForcesOPSorted.rename(columns={'Unique Name':'Element Label'},inplace = True)
    
        df_Frames = self.genframes
        df_ColumnForcesOP.rename(columns={'Unique Name':'Element Label'},inplace=True)
        ubicacion_columnas= pd.merge(df_ColumnForcesOP,df_Frames[['Element Label','CoordI','t3','t2','Angle','Object Label']],
                                     on = 'Element Label', how = 'inner')
    
        ubicacion_columnas['CoordX'] = [val[0] for val in ubicacion_columnas['CoordI']]
        ubicacion_columnas['CoordY'] = [val[1] for val in ubicacion_columnas['CoordI']]
        ubicacion_columnas = ubicacion_columnas[['Element Label','CoordX','CoordY','t3','t2','Angle','DFP','Object Label']]
    
        ubicacion_vigas = df_Frames[df_Frames['Story'] == 'Story1']
        ubicacion_vigas = ubicacion_vigas[ubicacion_vigas['Design Type'] == 'Beam']
        ubicacion_vigas['CoordXI'] = [val[0] for val in ubicacion_vigas['CoordI']]
        ubicacion_vigas['CoordYI'] = [val[1] for val in ubicacion_vigas['CoordI']]
        ubicacion_vigas['CoordXJ'] = [val[0] for val in ubicacion_vigas['CoordJ']]
        ubicacion_vigas['CoordYJ'] = [val[1] for val in ubicacion_vigas['CoordJ']]
        ubicacion_vigas = ubicacion_vigas[['Element Label','CoordXI','CoordYI','CoordXJ','CoordYJ']]
    
    
        # Colores para cada tipo de dato (ETABS y OpenSees)
        colors = ['#D3DDC5' if value <= self.threshold else '#75B72A' for value in diferencias]
    
        # Obtener fuerzas para las columnas seleccionadas
        etabs_forces = [abs(df_ColumnForcesOPSorted['P'].iloc[i]) for i in index_MaxDFPColumn]
        opensees_forces = [abs(df_ColumnForcesOPSorted['P(kN)'].iloc[i]) for i in index_MaxDFPColumn]
        
        # Diccionario completo
        data_basecolumn = {
            'columnas': columnas,
            'etabs': etabs_forces,
            'opensees': opensees_forces,
            'diferencias': diferencias,
            'colors': colors,
            'threshold': self.threshold,
            'ubicacion_columnas': ubicacion_columnas,
            'ubicacion_vigas': ubicacion_vigas
        }
            
        self.data_basecolumn = data_basecolumn

    def _plot_BRverification(self):
        
        # Leer imagen y convertirla a base64
        logo_path = os.path.join(self.folder_temp_path, 'logo_estrucmed.png')  # Ajusta el nombre del archivo
        with open(logo_path, 'rb') as image_file:
            logo_base64 = base64.b64encode(image_file.read()).decode()
                
        # --- Inicializar Figura
        fig = make_subplots(
            rows=1, cols=2,
            shared_yaxes=False,
            shared_xaxes=True,
            column_widths=[0.5, 0.5],
            horizontal_spacing=0.1,
            specs=[[{"secondary_y": True}, {"secondary_y": True}]],
            subplot_titles=(
                "<b>Comparación reacciones en la base</b>", "<b>Vista en planta (primer piso)</b>"
            )
        )
        
        # Ajustar posición de títulos
        for annotation in fig['layout']['annotations']:
            annotation['yshift'] = 30
        
        # Graficas diferencias porcentuales
        self._plot_maxDFP(fig)
        # Graficar vista en planta de la base
        self._plot_columnas_vigas(fig)
        
        # Layout final
        fig.update_layout(
            # hovermode='x unified',
            title={
                'text': "<br><b>VERIFICACIONES DEL MODELO OPENSEES</b><br>"+
                        "<span style='font-size:20px; color:#3A3A3A'>Comparación de reacciones en la base de las columnas</span><br><br>",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {
                    'size': 27,
                    'color': '#75B72A',
                    'family': 'Arial'
                }
            },
            images=[
                dict(
                    source=f"data:image/png;base64,{logo_base64}",
                    xref="paper", yref="paper",
                    x=0.01, y=1.5,
                    sizex=0.15, sizey=0.15,
                    xanchor="left", yanchor="top",
                    opacity=0.8,
                    layer="above"
                )
            ],
            legend=dict(
                orientation='h',
                x=0.45,
                y=-0.15,
                xanchor='center',
                yanchor='top',
                traceorder='normal',
                itemwidth=50
            ),
            height=750,
            # margin=dict(t=250),
            margin=dict(
                l=190,   # left
                r=40,   # right
                t=250,   # top
                b=60    # bottom
            ),
            plot_bgcolor='white',
            
            xaxis=dict(
                title = 'Columnas',
                tickfont=dict(size=10),
                ticks = 'outside',
                ticklen = 5,
                tickwidth = 2,
                gridcolor = 'lightgray',
                gridwidth = 0.5,
                zeroline = False,
                showline = True,
                linecolor = 'black',
                linewidth = 0.5,
                mirror = True,
            ),
            
            yaxis=dict(
                title = 'Fuerza axial (kN)',
                tickfont=dict(size=10),
                ticks = 'outside',
                ticklen = 5,
                tickwidth = 2,
                gridcolor = 'lightgray',
                gridwidth = 0.5,
                zeroline = False,
                showline = True,
                linecolor = 'black',
                linewidth = 0.5,
                mirror = True
            ),
            yaxis2=dict(
                title = 'Diferencia porcentual (%)',
                tickfont=dict(size=10),
                ticks = 'outside',
                ticklen = 5,
                tickwidth = 2,
            ),
            
            xaxis2 = dict(
                title = 'Coordenadas X (m)',
                tickfont=dict(size=10),
                ticks = 'outside',
                ticklen = 5,
                tickwidth = 2,
                gridcolor = 'lightgray',
                gridwidth = 0.5,
                zeroline = False,
                showline = True,
                linecolor = 'black',
                linewidth = 0.5
                ),
            
            yaxis3 = dict(
                title = 'Coordenadas Y (m)',
                tickfont=dict(size=10),
                ticks = 'outside',
                ticklen = 5,
                tickwidth = 2,
                gridcolor = 'lightgray',
                gridwidth = 0.5,
                zeroline = False,
                showline = True,
                linecolor = 'black',
                linewidth = 0.5
                )
        )
        
        
        # Exportar y abrir en navegador
        output_file = os.path.join(os.path.join(self.folder_path, 'html'), 'reaction_comparison.html')
        fig.write_html(output_file)
        
    def _plot_maxDFP(self, fig):
        
        # Extraer datos del diccionario
        columnas = self.data_basecolumn['columnas']
        etabs = self.data_basecolumn['etabs']
        opensees = self.data_basecolumn['opensees']
        diferencias = self.data_basecolumn['diferencias']
        
        df = pd.DataFrame({
            'Columna': columnas,
            'ETABS': etabs,
            'OpenSees': opensees,
            'Diferencia': diferencias
        })
        
        # Barras de ETABS
        fig.add_trace(go.Bar(
            x=df['Columna'],
            y=df['ETABS'],
            name='ETABS',
            marker_color = '#D1D1D1',
            hovertext = [f'Fuerza axial ETABS: {v:.2f} kN' for v in df['ETABS']],
            hoverinfo = "text",
            opacity = 0.8,
            showlegend = True
        ), row = 1, col = 1)
        
        # Barras de OpenSees
        fig.add_trace(go.Bar(
            x=df['Columna'],
            y=df['OpenSees'],
            name='OpenSees',
            marker_color='#75B72A',
            hovertext = [f'Fuerza axial OpenSees: {v:.2f} kN' for v in df['OpenSees']],
            hoverinfo = "text",
            opacity = 0.8,
            showlegend = True
        ), row = 1, col = 1)
        
        # Línea de diferencia porcentual
        fig.add_trace(go.Scatter(
            x=df['Columna'],
            y=df['Diferencia'],
            name='Diferencia (%)',
            mode='lines+markers+text',
            hovertext = [f'Diferencia porcentual: {v:.2f}%' for v in df['Diferencia']],
            hoverinfo = "text",
            marker=dict(color='black', size = 8),
            line=dict(color='#0D3512', width=1.5, dash='dot'),
            showlegend = True
        ), row=1, col=1, secondary_y=True)
        
        # Umbral
        fig.add_trace(go.Scatter(
            x=df['Columna'],
            y=[10]*len(df['Columna']),
            name='Umbral',
            mode='lines',
            line=dict(color='#C2463C', width=3.2, dash='dash'),
            showlegend = True,
            hoverinfo = 'skip'
        ), row=1, col=1, secondary_y=True)
    
    def _plot_columnas_vigas(self, fig):
        
        df_columnas = self.data_basecolumn['ubicacion_columnas']
        df_vigas = self.data_basecolumn['ubicacion_vigas']
        
        # Dibujar vigas como líneas
        for index, row in df_vigas.iterrows():
            xplot, yplot = self.mallado(row['CoordXI'], row['CoordXJ'], 
                                        row['CoordYI'], row['CoordYJ'],
                                        10, 10, vector = True)
            customdata = [row['Element Label']]*10
            fig.add_trace(go.Scatter(
                x=xplot,
                y=yplot,
                mode='lines',
                line=dict(color='black', width=2.0),
                name='Viga',
                hovertemplate="Viga %{customdata} <extra></extra>",
                hoverlabel=dict(bgcolor='rgba(250,250,250,1.0)'),
                opacity=1.0,
                showlegend=False,
                customdata = customdata
                            
            ), row = 1, col = 2)
        
        # Dibujar columnas como polígonos (rotated rectangles)
        for index, row in df_columnas.iterrows():
            color = '#C2463C' if row['DFP'] > self.threshold else '#81C587'

            # Coordenadas del rectángulo sin rotar (centrado en 0,0)
            dx = row['t2'] / 2
            dy = row['t3'] / 2
            coords = np.array([
                [-dx, -dy],
                [ dx, -dy],
                [ dx,  dy],
                [-dx,  dy],
                [-dx, -dy]
            ])

            # Rotar y trasladar
            theta = np.radians(row['Angle']+90)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            rotated_coords = coords @ rotation_matrix.T
            x_coords = rotated_coords[:, 0] + row['CoordX']
            y_coords = rotated_coords[:, 1] + row['CoordY']
            
            customdata = [row['Object Label']]*len(x_coords)
            
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                fill='toself',
                line=dict(color='black',width=1.0),
                fillcolor=color,
                name="Columna",
                hovertemplate="Columna %{customdata} <extra></extra>",
                hoverlabel=dict(bgcolor='rgba(250,250,250,1.0)'),
                showlegend=False,
                opacity=0.8,
                marker=dict(size=1, color='black'),
                customdata=customdata
            ), row = 1, col = 2)
            
        # Agregar leyenda manual
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(size=10, color='#81C587'),
                                 legendgroup='df_low',
                                 name=f'DF ≤ {self.threshold:.2f}'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(size=10, color='#C2463C'),
                                 legendgroup='df_high',
                                 name=f'DF > {self.threshold:.2f}'))
        
        fig.update_xaxes(
            range = [min(df_vigas['CoordXI'])-10.0,max(df_vigas['CoordXI'])+10],
            row = 1, col = 2
        )
        
        fig.update_yaxes(
            range = [min(df_vigas['CoordYI'])-2.0,max(df_vigas['CoordYI'])+2.0],
            row = 1, col = 2
        )

    
    def mallado(self, xmin, xmax, ymin, ymax, nx, ny, vector = False):
        
        if vector == False:
            xgrid = np.linspace(xmin,xmax,nx)
            ygrid = np.linspace(ymin,ymax,ny)
            xpoint, ypoint = np.meshgrid(xgrid,ygrid)
            xflatten = xpoint.flatten()
            yflatten = ypoint.flatten()
        else:
            xflatten = np.linspace(xmin,xmax,nx)
            yflatten = np.linspace(ymin,ymax,nx)
        
        return xflatten, yflatten
    
    