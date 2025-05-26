"""
@author: Daniela Novoa, Orlando Arroyo, Frank Vidales
@owner: EstrucMed Ingeniería especializada S.A.S

-------------------------------------------------------------------------------
                   Módulo de Comparación de Derivas por FHE (ETABS vs OpenSees)
                      Resultados + Visualización Interactiva (Dashboard)
-------------------------------------------------------------------------------

Este módulo compara las derivas de entre piso obtenidas mediante el 
método de Fuerza Horizontal Equivalente (FHE) entre ETABS y OpenSeesPy.

Incluye:
    • Aplicación de cargas tipo FHE y ejecución del análisis estático lineal.
    • Comparación de derivas piso a piso en ambas direcciones.
    • Visualización gráfica de la diferencia porcentual entre modelos.
    • Generación de un dashboard interactivo con Plotly y animaciones.

Objetivo:
Evaluar la fidelidad del modelo OpenSeesPy frente a deformaciones globales 
esperadas por carga lateral convencional.

Salidas:
    - Gráficos de perfiles de deriva.
    - Tablas por piso con desviaciones.
    - Animaciones del comportamiento lateral con `opstool`.

Unidades: Sistema Internacional (SI)

"""     

import pandas as pd
import openseespy.opensees as ops
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import opstool as opst
import opstool.vis.plotly as opsvis
from tqdm import tqdm

import opseestools.analisis3D as an

class fhemethod_class:

    def __init__(self, model_data, modal_analysis, main_path, data_processfile):

        self.model_data = model_data
        self.temporal_path = os.path.abspath(os.path.join(main_path, 'data', 'temporary'))
        self.ver_path = os.path.abspath(os.path.join(main_path, 'outputs', 'verification_results'))
        self.modal_analysis = modal_analysis
        self.data_processfile = data_processfile
        
        self._main_FHEmethod()

    def _main_FHEmethod(self):

        data_FHE, fig_animation1, fig_animation2 = self._FHE_factors()
            # 1). Generar excel con resultados FHE en ambas direcciones
        df_Driftsx = pd.DataFrame({
            'Height' : data_FHE['Heights'],
            'ETABS FHEx' : data_FHE['ETABS_driftsx'],
            'OpenSees FHEx' : data_FHE['OPs_driftsx']
        })
        
        df_Driftsy = pd.DataFrame({
            'Height' : data_FHE['Heights'],
            'ETABS FHEy' : data_FHE['ETABS_driftsy'],
            'OpenSees FHEy' : data_FHE['OPs_driftsy']
        })
        
        dataframes = {
            'FHEx Results' : df_Driftsx,
            'FHEy Results' : df_Driftsy
            }
        

        excel_path = os.path.join(os.path.join(self.ver_path, 'xlsx'), 'fhe_results.xlsx')

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        self._plotLayout(data_FHE, fig_animation1, 'x')
        self._plotLayout(data_FHE, fig_animation2, 'y')
                    

    def _FHE_factors(self):

        df_DCMetabs = self.model_data.displacements
        df_DiafMasses = self.model_data.masses
        df_ModalOP = self.modal_analysis.modal_OPSEESNew
        self.df_ResponseFunctions = self.model_data.functions

        DCMetabs_grouped = {name: group.reset_index(drop=True) for name, group in df_DCMetabs.groupby('Story')}
    
        df_FHEetabs = {'Base':[0,0,0]}
        for story, df_FHEstory in DCMetabs_grouped.items():
            df1 = df_FHEstory[df_FHEstory['Load Case/Combo'].str.contains('FHEx', na=False)]
            df2 = df_FHEstory[df_FHEstory['Load Case/Combo'].str.contains('FHEy', na=False)]
            df_FHEetabs[story] = [max(df1['UX']),max(df2['UY']),max(df1['Z'])]
        
        ETBS_results = {'Base':[0,0,0]}
        stories = list(df_FHEetabs.keys())
        for i in range(1,len(stories)):
            lower = stories[i - 1]  # Piso inferior
            upper = stories[i]      # Piso superior
            # Altura entre piso
            h = df_FHEetabs[upper][2] - df_FHEetabs[lower][2]
            # Derivas en X e Y
            drift_x = (df_FHEetabs[upper][0] - df_FHEetabs[lower][0]) / h
            drift_y = (df_FHEetabs[upper][1] - df_FHEetabs[lower][1]) / h
            
            ETBS_results[upper] = [drift_x*100, drift_y*100, df_FHEetabs[upper][2]]
        
        # 2). Obtener valor se Sa y k de OpenSees
        Tx_OpS = df_ModalOP['Periodo (s)'][df_ModalOP['Masa Participativa X (%)'] == max(df_ModalOP['Masa Participativa X (%)'])].values[0]
        Ty_OpS = df_ModalOP['Periodo (s)'][df_ModalOP['Masa Participativa Y (%)'] == max(df_ModalOP['Masa Participativa Y (%)'])].values[0]
        
        Sax, kx = self._Calculate_Factors(Tx_OpS)
        Say, ky = self._Calculate_Factors(Ty_OpS)
        
        altur = np.sort(pd.unique(df_DCMetabs['Z']))
        dnodes = [100000000*i for i in range(len(altur))]
        masas = df_DiafMasses['Mass X'].to_numpy()
        masas = masas/1000
        masas = list(masas)
        
        driftx_OP = self._FHE_analysis(masas,altur,dnodes,Sax,kx,2001,'X')
        stories = list(ETBS_results.keys())
        driftx_ET = [ETBS_results[story][0] for story in stories]
        drifty_ET = [ETBS_results[story][1] for story in stories]
        
        ratiox = self._ratio(driftx_OP, driftx_ET)

        tqdm.write("🔹 Aplicando el método FHE en dirección X...")
        self.model_data.genElasticModel(self.data_processfile)
        driftx_OP2 = self._FHE_analysis(masas,altur,dnodes,Sax,kx,2002,'X',mean_ratio = ratiox, odb_tag = 1)
        
        fig_animation1 = opsvis.plot_nodal_responses_animation(
            odb_tag=1,
            framerate=None,
            scale=1.0,
            show_defo=True,
            resp_type='disp',
            resp_dof=('UX', 'UY', 'UZ'),
            show_bc=True,
            bc_scale=1.0,
            show_mp_constraint=False,
            show_undeformed=False,
            style='surface',
            show_outline=False
        )
        
        tqdm.write("🔹 Aplicando el método FHE en dirección Y...")
        self.model_data.genElasticModel(self.data_processfile)
        drifty_OP = self._FHE_analysis(masas,altur,dnodes,Say,ky,2003,'Y')
        ratioy = self._ratio(drifty_OP, drifty_ET)

        self.model_data.genElasticModel(self.data_processfile)
        drifty_OP2 = self._FHE_analysis(masas,altur,dnodes,Say,ky,2004,'Y',mean_ratio = ratioy, odb_tag = 2)
        
        fig_animation2 = opsvis.plot_nodal_responses_animation(
            odb_tag=2,
            framerate=None,
            scale=1.0,
            show_defo=True,
            resp_type='disp',
            resp_dof=('UX', 'UY', 'UZ'),
            show_bc=True,
            bc_scale=1.0,
            show_mp_constraint=False,
            show_undeformed=False,
            style='surface',
            show_outline=False
        )
        
        
        coordz = [0]
        coordz += list(altur)

        # Extraer datos para los gráficos
        stories = list(ETBS_results.keys())
        heights = [ETBS_results[story][2] for story in stories]

        data1_x = [ETBS_results[story][0] for story in stories]
        data1_y = [ETBS_results[story][1] for story in stories]

        data2_x = driftx_OP2
        data2_y = drifty_OP2
        
        data_FHE = {
            'Heights' : heights,
            'ETABS_driftsx' : data1_x,
            'ETABS_driftsy' : data1_y,
            'OPs_driftsx' : data2_x,
            'OPs_driftsy' : data2_y,
            'Sax' : Sax,
            'Say' : Say,
            'kx' : kx,
            'ky' : ky,
            'Stories' : stories
            }
        
        return data_FHE, fig_animation1, fig_animation2


    def _Calculate_Factors(self,T):
        
        index_safunction = self.df_ResponseFunctions.index.tolist()
        
        Av = self.df_ResponseFunctions['Av'][index_safunction[0]]
        Fv = self.df_ResponseFunctions['Fv'][index_safunction[0]]
        Group = self.df_ResponseFunctions['Group of Use'][index_safunction[0]]
        
        if Group == 'Group 1':
            Group = 1
        elif Group == 'Group 2':
            Group = 1.1
        elif Group == 'Group 3':
            Group = 1.25
        elif Group == 'Group 4':
            Group = 1.5
        
        Sa = (1.2*Av*Fv*Group/T) # Sobre 10 para incursion elastica
        
        # Calculate k coeficient
        if T < 0.5:
            k = 1
        elif T>0.5 and T<2.5:
            k = 0.75+0.5*T
        else:
            k = 2
        
        return k,Sa
    
    def _FHE_analysis(self,mass,altur,dnodes,Sa,k,tagPattern,direction,mean_ratio=1, odb_tag = None):

        ops.wipeAnalysis()
        Mest = np.sum(mass)
        Vbasal = Mest*Sa*9.81/mean_ratio
        fi = np.array([mass[i]*altur[i]**k for i in range(len(dnodes))])
        Cv = fi/sum(fi)
        Fi = Vbasal*Cv

        ops.timeSeries('Linear', tagPattern)
        ops.pattern('Plain',tagPattern,tagPattern)
        
        for i,n in enumerate(dnodes):
            if direction == 'X':
                ops.load(n,Fi[i],0,0,0,0,0)
            else:
                ops.load(n,0,Fi[i],0,0,0,0)
        
        if odb_tag is not None:
        
            # Configurar análisis estático
            ops.system('BandGeneral')
            ops.constraints('Transformation')
            ops.numberer('RCM')
            ops.test('NormDispIncr', 1.0e-12, 10, 3)
            ops.algorithm('Newton')
            ops.integrator('LoadControl', 0.01)  # 🔵 Paso más pequeño para capturar mejor deformaciones
            ops.analysis('Static')
            
            ODB = opst.post.CreateODB(odb_tag=odb_tag) # 🔥 Crear ODB
            
            num_steps = 100
            for _ in range(num_steps):
                ok = ops.analyze(1)
                if ok != 0:
                    print('⚠️ Error en análisis en un paso de FHE')
                    break
                ODB.fetch_response_step()  # 🔥 Guardar el paso actual        
            
            ODB.save_response()  # 🔥 Guardar ODB al final
        
        else:
            
            an.gravedad() 

        disp = [0]
        if direction == 'X': 
            disp += [ops.nodeDisp(n,1) for n in dnodes]
        else:
            disp += [ops.nodeDisp(n,2) for n in dnodes]
        
        coordz = [0]
        coordz += list(altur)
        
        drift = [0]
        for i in range(1,len(coordz)):
            # Altura entre piso
            h = coordz[i] - coordz[i-1]
            # Derivas de entre piso
            drift.append(((disp[i] - disp[i-1]) / h) * 100)

        return drift

    def _ratio(self, drift_OP, drift_ET):

        ratio = np.array(drift_OP[1:]) / np.array(drift_ET[1:])
        # filtrar los valores que no son NaN
        valid_values = ratio[~np.isnan(ratio)]
        # media de los valores
        mean_ratio = np.min(valid_values)

        return mean_ratio
    
    def _plotLayout(self, data_FHE, fig_animation1, key1):
        
        key = f'drifts{key1}'
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "xy"}, {"type": "table"}],   # Fila 1: Gráfico + Tabla
                [{"type": "scene", "colspan": 2}, None]  # Fila 2: Animación ocupando dos columnas
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5],  # Opcional, para controlar proporciones
            subplot_titles=("Derivas máximas de entre piso", "Tabla de derivas máximas", "Deformación de la estructura")
        )
        
        # --- Fila 1, columna 1: Grafico comparaciones ETABS vs OpenSees
        
        customdata = list(zip(data_FHE[f'ETABS_{key}'], data_FHE[f'OPs_{key}']))
        
        self._plotDrifts(
            data_FHE[f'ETABS_{key}'], data_FHE['Heights'], 
            data_FHE[f'OPs_{key}'], data_FHE['Heights'], 
            customdata, fig)

        data_etabslist = [f'{val:.2f}%' for val in data_FHE[f'ETABS_{key}']]
        data_opslist = [f'{val:.2f}%' for val in data_FHE[f'OPs_{key}']]
        
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["<b>Piso</b>", "<b>Altura (m)</b>", f"<b>ETABS FHE{key1}</b>", f"<b>OpenSees FHE{key1}</b>"],
                    fill_color='#75B72A',  # Color de fondo del header
                    font=dict(color='black', size=18, family='Arial'),
                    align='center',
                    height=50
                ),
                cells=dict(
                    values=[data_FHE['Stories'], data_FHE['Heights'], data_etabslist, data_opslist],
                    fill_color='#F2F2F2',  # Color de fondo de las celdas
                    font=dict(color='black', size=14, family='Arial'),  # Texto normal
                    height=50,
                    align='center'
                )
            ),
            row=1, col=2
        )
        
        # Incluir animacion
        
        fig_animation = fig_animation1

        # Agregar los datos de la animación
        for trace in fig_animation.data:
            trace.showlegend = False 
            fig.add_trace(trace, row=2, col=1)
            
        if fig_animation.frames:
            # Ajustar los frames antes de agregarlos
            adjusted_frames = []
            for frame in fig_animation.frames:
                adjusted_frame = go.Frame(
                    data=frame.data,
                    name=frame.name,
                    traces=list(range(len(fig.data) - len(frame.data), len(fig.data)))
                    # Solo afectan a las últimas trazas que son de la animación
                )
                adjusted_frames.append(adjusted_frame)
            
            fig.frames = adjusted_frames
            
        # for trace in fig.data:
        #     if isinstance(trace, go.Scatter3d):
        #         print(trace.marker.colorbar)
                
        # Read image and convert to base64
        logo_path = os.path.join(self.temporal_path, 'logo_estrucmed.png')
        with open(logo_path, 'rb') as image_file:
            logo_base64 = base64.b64encode(image_file.read()).decode()
                
        # Configure layout
        fig.update_layout(
            height=1300,
            margin=dict(
                l=190,   # left
                r=190,   # right
                t=250,   # top
                b=60    # bottom
            ),
            plot_bgcolor='white',
            hovermode='x unified',
            title={
                'text': "<br><b>VERIFICACIONES DEL MODELO OPENSEES</b><br>"+
                        f"<span style='font-size:20px; color:#3A3A3A'>Comparación derivas máximas de entre piso en dirección {key1}</span><br><br>",
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
            images=[dict(
                source=f"data:image/png;base64,{logo_base64}",
                xref="paper", yref="paper",
                x=0.01, y=1.20,
                sizex=0.15, sizey=0.15,
                xanchor="left", yanchor="top",
                opacity=0.8,
                layer="above"
            )],
            legend=dict(
                orientation='h',
                x=0.22,
                y=0.49,
                xanchor='center',
                yanchor='top',
                traceorder='normal',
                itemwidth=50
            ),
            scene=dict(
                aspectmode='data'
            ),
            updatemenus=fig_animation.layout.updatemenus,  # Botones de animación
            sliders=[{
                **fig_animation.layout.sliders[0].to_plotly_json(),  
                'x': 0.1,        # Posición horizontal inicial (0 a 1)
                'y': 0,          # Posición vertical (0 es muy abajo, 0.05 mejor)
                'len': 0.8,      # Longitud (0 a 1)
                'pad': {"t": 50}, # Espacio desde arriba
                'ticklen': 15  # Grosor de la barra
            }],
            xaxis=dict(
                title='Deriva (%)',
                tickfont=dict(size=10),
                ticks='outside',
                ticklen=5,
                tickwidth=2,
                gridcolor='lightgray',
                gridwidth=0.5,
                zeroline=False,
                showline=True,
                linecolor='black',
                linewidth=0.5,
                mirror=True,
            ),
            yaxis=dict(
                title='Altura (m)',
                tickfont=dict(size=10),
                ticks='outside',
                ticklen=5,
                tickwidth=2,
                gridcolor='lightgray',
                gridwidth=0.5,
                zeroline=False,
                showline=True,
                linecolor='black',
                linewidth=0.5,
                mirror=True
            )
        )
                        
        # Export and open in browser
        
        output_file = os.path.join(os.path.join(self.ver_path, 'html'), f'drift_comparison_FHE{key1}.html')
        fig.write_html(output_file)
   
    def _plotDrifts(self, x_etabs, y_etabs, x_ops, y_ops, customdata, fig):
        # Add ETABS curve
        fig.add_trace(go.Scatter(
            x=x_etabs,
            y=y_etabs,
            mode='lines+markers',
            name='ETABS',
            marker=dict(color='#494949'),
            line=dict(color='#494949'),
            text=[
                    f"<span style='font-size:14px; color:black;'>"
                    f"<b>Deriva ETABS:</b> {row[0]:.2f}%"
                    f"</span>"
                    for row in customdata
                ],
            hoverinfo='text'
        ), row=1, col=1)
    
        # Add OpenSees curve
        fig.add_trace(go.Scatter(
            x=x_ops,
            y=y_ops,
            mode='lines+markers',
            name='OpenSees',
            marker=dict(color='#75B72A'),
            line=dict(color='#75B72A'),
            text=[
                    f"<span style='font-size:14px; color:black;'>"
                    f"<b>Deriva OpenSees:</b> {row[1]:.2f}%"  # Fixed text to show OpenSees value
                    f"</span>"
                    for row in customdata
                ],
            hoverinfo='text'
        ), row=1, col=1)
        
        return fig

    
            
