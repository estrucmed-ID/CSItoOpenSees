"""
@author: Daniela Novoa, Orlando Arroyo, Frank Vidales
@owner: EstrucMed Ingeniería especializada S.A.S

-------------------------------------------------------------------------------
                Módulo de Comparación Modal (ETABS vs OpenSees)
              Resultados + Visualización Interactiva (Dashboard)
-------------------------------------------------------------------------------

Este módulo realiza la comparación de resultados modales entre el modelo de ETABS 
y el modelo equivalente generado en OpenSeesPy.

Incluye:
    • Comparación de periodos de vibración por modo.
    • Comparación de masas participativas en X, Y y rotación.
    • Generación de reportes interactivos con tablas y gráficos usando Plotly.
    • Visualización de modos animados con la herramienta `opstool`.

Este módulo permite validar el comportamiento dinámico global del modelo 
convertido y comunicar los resultados en un formato claro, visual e interactivo.

Salidas:
    - Tablas comparativas con porcentajes de diferencia.
    - Gráficos de barras y radar comparativos.
    - Dashboard completo embebido para visualización y exportación.

Unidades: Sistema Internacional (SI)

"""     

import pandas as pd
import openseespy.opensees as ops
import sys
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64


utils_path = os.getcwd()
sys.path.append(utils_path)


class modal_analysis:

    def __init__(self, model_data, main_path, modal_results, umbral_diferencia = 10.0, umbral_masa = 5.0):

        self.joints = model_data.OE_joints
        self.joints.rename(columns={'Joint J':'Element Label'}, inplace = True)
        self.diaphragm = model_data.masses
        self.modal_ETABS_OR = model_data.modal
        self.center_of_mass_nodes = model_data.center_of_mass_nodes
        self.Nmodes = len(model_data.altur) * 3
        
        self.folder_path = os.path.abspath(os.path.join(main_path, 'outputs', 'opensees_models'))
        self.folder_path_vr = os.path.abspath(os.path.join(main_path, 'outputs', 'verification_results'))
        self.folder_temp_path = os.path.abspath(os.path.join(main_path, 'data', 'temporary'))
        
        self.umbral_diferencia = umbral_diferencia
        self.umbral_masa = umbral_masa

        self.modal_displacements = []
        self.modal_results = modal_results
        self.modal_OPSEES = []

        self._ProcessModalInfo()

    def _ProcessModalInfo(self):
        
        self._get_modal_results() # Obtener resultados OpenSees y ETABS

        self.modal_OPSEESNew, self.modal_ETABSNew  = self._Modal_Matching_OpenSees_ETABS(self.modal_OPSEES, self.modal_ETABS)

        self.results_dic = {
            'XComparison': self._modal_comparison('X'),
            'YComparison': self._modal_comparison('Y'),
            'RXComparison': self._modal_comparison('RX'),
            'RYComparison': self._modal_comparison('RY'),
            'RZComparison': self._modal_comparison('RZ'),
        }

        self._plot_comparison(self.results_dic,'XComparison')

    def _Modal_Matching_OpenSees_ETABS(self, modal_OPSEES, modal_ETABS):

        # Agregar condiciones modales
        modal_OPSEES['Condition'] = self._Compare_Modal_Results(modal_OPSEES)
        modal_ETABS['Condition'] = self._Compare_Modal_Results(modal_ETABS)
        
        modal_ETABS_copy = modal_ETABS.copy()

        # Match basado en condición
        periodos_comparacion = []
        modos_comparacion = []
        for cond_OP in modal_OPSEES['Condition']:
            idx = modal_ETABS_copy[modal_ETABS_copy['Condition'] == cond_OP].index
            if not idx.empty:
                index_match = idx[0]
                periodos_comparacion.append(modal_ETABS_copy.at[index_match, 'Periodo (s)'])
                modos_comparacion.append(modal_ETABS_copy.at[index_match, 'Modo'])
                modal_ETABS_copy.drop(index=index_match, inplace=True)
                modal_ETABS_copy.reset_index(drop=True, inplace=True)
            else:
                periodos_comparacion.append(np.nan)
                modos_comparacion.append(np.nan)

        modal_OPSEES['Periodo ETABS (s)'] = periodos_comparacion
        modal_OPSEES['Modo ETABS'] = modos_comparacion

        # Calcular diferencia porcentual de períodos
        modal_OPSEES['Diferencia Periodo (%)'] = np.abs(modal_OPSEES['Periodo (s)'] - modal_OPSEES['Periodo ETABS (s)']) / modal_OPSEES['Periodo ETABS (s)'] * 100

        modal_ETABS2 = modal_OPSEES.copy()
        modal_ETABS2 = modal_ETABS2[['Modo ETABS']]
        modal_ETABS2.rename(columns = {'Modo ETABS':'Modo'}, inplace = True)
        modal_ETABS2 = pd.merge(modal_ETABS2, modal_ETABS[['Modo', 'Periodo (s)', 'Masa Participativa X (%)',
                                                           'Masa Participativa Y (%)', 'Masa Participativa RX (%)',
                                                           'Masa Participativa RY (%)', 'Masa Participativa RZ (%)']],
                                                           on = 'Modo', how = 'inner')
        
        modal_ETABS2['Modo'] = range(1,len(modal_ETABS2)+1)

        return modal_OPSEES, modal_ETABS2

    def _modal_comparison(self, direction):

        # Modos dominantes que tienen una masa participatiba mayor a 5%
        modos_dominantes = self._identificar_modos_dominantes(self.modal_ETABSNew, direction)
        results = []

        for modo in self.modal_ETABSNew['Modo']:
            row_etabs = self.modal_ETABSNew[self.modal_ETABSNew['Modo'] == modo]
            row_ops = self.modal_OPSEESNew[self.modal_OPSEESNew['Modo'] == modo]

            if not row_etabs.empty and not row_ops.empty:
                periodo_etabs = row_etabs['Periodo (s)'].values[0]
                periodo_ops = row_ops['Periodo (s)'].values[0]
                masa_etabs = np.around(row_etabs[f'Masa Participativa {direction} (%)'].values[0],2)
                masa_ops = np.around(row_ops[f'Masa Participativa {direction} (%)'].values[0],2)

                diferencia_periodo = abs(periodo_ops - periodo_etabs) / periodo_etabs * 100 if periodo_etabs != 0 else 0
                diferencia_masa = abs(masa_ops - masa_etabs) / masa_etabs * 100 if masa_etabs != 0 else 0

                results.append({
                'Modo': modo,
                'Periodo ETABS (s)': periodo_etabs,
                'Periodo OpenSees (s)': periodo_ops,
                'Masa ETABS (%)': masa_etabs,
                'Masa OpenSees (%)': masa_ops,
                'Diferencia Periodo (%)': diferencia_periodo,
                'Diferencia Masa (%)': diferencia_masa,
                'Dominante': modo in modos_dominantes,
                'Supera Umbral - periodo': diferencia_periodo > self.umbral_diferencia,
                'Supera Umbral - masa': diferencia_masa > self.umbral_diferencia
                })
        
        dic_results = {
            'results':results,
            'modos_dominantes':modos_dominantes
        }

        return dic_results
    
    def _Compare_Modal_Results(self, df_Modal):

        condition = []
        for _, row in df_Modal.iterrows():
            masas = {
                'MX': row['Masa Participativa X (%)'],
                'MY': row['Masa Participativa Y (%)'],
                'MRX': row['Masa Participativa RX (%)'],
                'MRY': row['Masa Participativa RY (%)'],
                'MRZ': row['Masa Participativa RZ (%)']
            }
            # Tomamos la dirección dominante
            direccion_dominante = max(masas, key=masas.get)
            condition.append(direccion_dominante)
        return condition
    
    def _plot_comparison(self, results_dic, direccion_actual):
        
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
                "<b>Comparación de Periodos Modales</b>", "<b>Comparación de Masas Participativas (%)</b>"
            )
        )
    
        # Ajustar posición de títulos
        for annotation in fig['layout']['annotations']:
            annotation['yshift'] = 30
            
        buttons = []
        directions = list(results_dic.keys())
        for i, direccion_actual in enumerate(directions):
            
            dir_data = pd.DataFrame(results_dic[direccion_actual]['results'])
            
            min_modo, max_modo = min(dir_data['Modo'])-1, max(dir_data['Modo'])+1
        
            # --- Función auxiliar para agregar barras
            def add_bars(fig, col, etabs_data, ops_data, label_etabs, label_ops, visible):
                fig.add_trace(go.Bar(
                    x=dir_data['Modo'],
                    y=etabs_data,
                    name='ETABS' if col == 1 else None,
                    marker_color='#D1D1D1',
                    hovertext=[f'{label_etabs}: {v:.2f}' for v in etabs_data],
                    hoverinfo="text",
                    opacity=0.8,
                    showlegend=(col == 1),
                    visible = visible
                ), row=1, col=col)
                
                fig.add_trace(go.Bar(
                    x=dir_data['Modo'],
                    y=ops_data,
                    name='OpenSees' if col == 1 else None,
                    marker_color='#75B72A',
                    hovertext=[f'{label_ops}: {v:.2f}' for v in ops_data],
                    hoverinfo="text",
                    opacity=0.8,
                    showlegend=(col == 1),
                    visible = visible
                ), row=1, col=col)
        
            # --- Función auxiliar para agregar diferencias
            def add_differences(fig, col, diff_data, showlegend, visible):
                fig.add_trace(go.Scatter(
                    x=dir_data['Modo'],
                    y=diff_data,
                    name='Diferencia (%)' if col == 1 else None,
                    mode='markers+lines',
                    marker=dict(
                        size=[11 if domin else 8 for domin in dir_data['Dominante']],
                        color=['#E1B12C' if domin else '#0D3512' for domin in dir_data['Dominante']],
                        symbol=['star' if domin else 'circle' for domin in dir_data['Dominante']]
                    ),
                    line=dict(color='#0D3512', width=1.5, dash='dot'),
                    showlegend=showlegend,
                    hovertemplate="<b>Diferencia:</b> %{y:.2f}%<extra></extra>",
                    hoverlabel=dict(bgcolor='rgba(250,250,250,1.0)'),
                    visible = visible
                ), row=1, col=col, secondary_y=True)
        
            # --- Función auxiliar para agregar umbral
            def add_umbral(fig, col, showlegend, visible):
                fig.add_trace(go.Scatter(
                    x=[min_modo, max_modo],
                    y=[self.umbral_diferencia]*2,
                    name='Umbral' if col == 1 else None,
                    mode='lines',
                    line=dict(color='#C2463C', width=3.2, dash='dash'),
                    showlegend=showlegend,
                    hovertemplate="<b>Umbral:</b> %{y:.0f}%<extra></extra>",
                    hoverlabel=dict(bgcolor='rgba(250,250,250,1.0)'),
                    visible = visible
                ), row=1, col=col, secondary_y=True)
        
            # --- Agregar periodos (columna 1)
            add_bars(fig, col=1, etabs_data=dir_data['Periodo ETABS (s)'], ops_data=dir_data['Periodo OpenSees (s)'],
                     label_etabs="Periodo ETABS (s)", label_ops="Periodo OpenSees (s)", visible=(i==0))
            add_differences(fig, col=1, diff_data=dir_data['Diferencia Periodo (%)'], showlegend=True, visible=(i==0))
            add_umbral(fig, col=1, showlegend=True, visible=(i==0))
            
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=15, symbol='star', color='#E1B12C'),
                name='Modo Dominante',
                showlegend=True,
                visible = (i==0)
            ))
            
        
            # --- Agregar masas (columna 2)
            add_bars(fig, col=2, etabs_data=dir_data['Masa ETABS (%)'], ops_data=dir_data['Masa OpenSees (%)'],
                     label_etabs="Masa ETABS (%)", label_ops="Masa OpenSees (%)", visible=(i==0))
            add_differences(fig, col=2, diff_data=dir_data['Diferencia Masa (%)'], showlegend=False, visible=(i==0))
            add_umbral(fig, col=2, showlegend=False, visible=(i==0))
            
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=15, symbol='star', color='#E1B12C'),
                name='Modo Dominante',
                showlegend=False,
                visible = (i==0)
            ))
            
            # Ejes x
            fig.update_xaxes(range=[min_modo, max_modo], row=1, col=1, showticklabels=False)
            fig.update_xaxes(range=[min_modo, max_modo], row=1, col=2, showticklabels=False)
            
            
            # --- Botón para esta dirección ---
            visibility = [False] * 50
            visibility[i*10:(i*10)+10] = [True] * 10
            
            label_text = direccion_actual.replace('Comparison', '')
            
            buttons.append(dict(
                label= f'Rotación {label_text[-1]}' if 'R' in label_text else f'Dirección {label_text}',
                method='update',
                args=[{'visible': visibility}]
            ))
        
        
        # Ejes
        fig.update_xaxes(showticklabels=False)
    
        # Layout final
        fig.update_layout(
            hovermode='x unified',
            title={
                'text': "<br><b>VERIFICACIONES DEL MODELO OPENSEES</b><br>"+
                        "<span style='font-size:20px; color:#3A3A3A'>Comparación de periodos modales y masas participativas</span><br><br>",
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
                x=0.5,
                y=-0.15,
                xanchor='center',
                yanchor='top',
                traceorder='normal',
                itemwidth=50
            ),
            height=750,
            margin=dict(t=250),
            updatemenus = [
                dict(
                    active = 0,
                    buttons = buttons
                ) 
            ],
            plot_bgcolor='white',
            
            xaxis=dict(
                title = 'Modos de vibración',
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
            
            xaxis2=dict(
                title = 'Modos de vibración',
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
            
            yaxis=dict(
                title = 'Periodo (s)',
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
            
            yaxis3=dict(
                title = 'Masa participativa (%)',
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
            
            yaxis4=dict(
                title = 'Diferencia porcentual (%)',
                tickfont=dict(size=10),
                ticks = 'outside',
                ticklen = 5,
                tickwidth = 2,
            ),
        )
    
        # Exportar
        output_file = os.path.join(os.path.join(self.folder_path_vr, 'html'), 'modal_analysis_verification.html')
        fig.write_html(output_file)

    def _identificar_modos_dominantes(self, df, direccion):

        col_masa = f'Masa Participativa {direccion} (%)'
        modos = df[df[col_masa] >= self.umbral_masa]['Modo'].tolist()
        return modos
    
    def _get_modal_results(self):

        self.modal_displacements = pd.DataFrame(self._get_modal_displacements())

        self.modal_OPSEES = pd.DataFrame({
            'Modo': np.arange(1, self.Nmodes+1),
            'Periodo (s)': self.modal_results['eigenPeriod'],
            'Masa Participativa X (%)': self.modal_results['partiMassRatiosMX'],
            'Masa Participativa Y (%)': self.modal_results['partiMassRatiosMY'],
            'Masa Participativa RX (%)': self.modal_results['partiMassRatiosRMX'],
            'Masa Participativa RY (%)': self.modal_results['partiMassRatiosRMY'],
            'Masa Participativa RZ (%)': self.modal_results['partiMassRatiosRMZ'],
            'Desplazamiento CM X (m)': self.modal_displacements['Desplazamiento CM X'],
            'Desplazamiento CM Y (m)': self.modal_displacements['Desplazamiento CM Y'],
            'Desplazamiento CM Z (m)': self.modal_displacements['Desplazamiento CM Z']
        })

        modal_ETABSdic = {}
        modal_ETABSdic['Masa Participativa X (%)'] = self.modal_ETABS_OR['UX'] * 100
        modal_ETABSdic['Masa Participativa Y (%)'] = self.modal_ETABS_OR['UY'] * 100
        modal_ETABSdic['Masa Participativa RX (%)'] = self.modal_ETABS_OR['RX'] * 100
        modal_ETABSdic['Masa Participativa RY (%)'] = self.modal_ETABS_OR['RY'] * 100
        modal_ETABSdic['Masa Participativa RZ (%)'] = self.modal_ETABS_OR['RZ'] * 100
        modal_ETABSdic['Periodo (s)'] = self.modal_ETABS_OR['Period'] 
        modal_ETABSdic['Modo'] = self.modal_ETABS_OR['Mode'] 

        self.modal_ETABS = pd.DataFrame(modal_ETABSdic)
        

    def _get_modal_displacements(self):

        # Inicializar almacenamiento
        disp_center_mass = {dof: [] for dof in range(1,7)}  # Desplazamientos modales en el centro de masa

        for i_mode in range(self.Nmodes):
            # Desplazamientos modales en el centro de masa
            for dof in range(1,4):
                disp_cm_mode = 0.0
                for node in self.center_of_mass_nodes:
                    try:
                        disp_cm_mode += ops.nodeEigenvector(node, i_mode+1, dof)
                    except:
                        pass
                disp_center_mass[dof].append(disp_cm_mode/len(self.center_of_mass_nodes))

        # Crear DataFrame de desplazamientos modales en el centro de masa
        df_modal_disp = {
            'Modo': np.arange(1, self.Nmodes+1),
            'Desplazamiento CM X': disp_center_mass[1],
            'Desplazamiento CM Y': disp_center_mass[2],
            'Desplazamiento CM Z': disp_center_mass[3],
        }

        return df_modal_disp
    