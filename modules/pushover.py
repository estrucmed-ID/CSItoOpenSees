"""
@author: Daniela Novoa, Orlando Arroyo, Frank Vidales
@owner: EstrucMed Ingeniería especializada S.A.S

-------------------------------------------------------------------------------
                    Módulo de Análisis Pushover Unidireccional
                Resultados + Visualización Interactiva (Dashboard)
-------------------------------------------------------------------------------

Este módulo ejecuta el análisis pushover en una dirección especificada 
(X o Y) sobre el modelo OpenSeesPy y genera el reporte correspondiente.

Incluye:
    • Generación de patrón lateral tipo FHE.
    • Control del desplazamiento mediante punto objetivo (nodo de control).
    • Registro de curva de capacidad: fuerza base vs desplazamiento.
    • Evaluación del mecanismo de colapso y deformaciones plásticas.

Además, se genera un **dashboard visual** que incluye:
    - Curva de capacidad interactiva (Plotly).
    - Tabla resumen de deformaciones y esfuerzos.
    - Animación de la deformación progresiva usando `opstool`.

Este módulo es clave para evaluar la capacidad sísmica del sistema 
estructural en análisis no lineal estático.

Unidades: Sistema Internacional (SI)
Norma base: NSR-10 (o FEMA 440/FEMA P-795 si se especifica)

"""     

import pandas as pd
import openseespy.opensees as ops
import sys
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import opstool as opst
import opstool.vis.plotly as opsvis
from tqdm import tqdm
import opseestools.analisis3D as an

utils_path = os.getcwd()
sys.path.append(utils_path)

# Importar módulos personalizados
import utilities as ut

class pushoverclass:
        
    point_style = {
        'capacidad_maxima':        {'name': 'Capacidad máxima',                    'marker-color': '#000000', 'marker-size': 12, 'marker-symbol': 'square'      },
        'capacidad80':             {'name': 'Capacidad al 80%',                    'marker-color': '#000000', 'marker-size': 12, 'marker-symbol': 'diamond'     },
        'fluencia_acero':          {'name': 'Primera fluencia del acero',          'marker-color': '#75B72A', 'marker-size': 12, 'marker-symbol': 'triangle-up' },
        'agrietamiento_concreto':  {'name': 'Primer agriertamiento del concreto',  'marker-color': '#75B72A', 'marker-size': 12, 'marker-symbol': 'star'        }
        }
    
    def __init__(self, model_data, main_path, direction, tag_pattern, odb_tag):
        
        self.model_data = model_data
        self.temporal_path = os.path.abspath(os.path.join(main_path, 'data', 'temporary'))
        self.push_path = os.path.abspath(os.path.join(main_path, 'outputs', 'pushover_results'))
        self.dir = direction
        self.odb_Tag = odb_tag
        
        an.gravedad()
        ops.loadConst('-time', 0.0)
        
        self.frames = self.model_data.GN_frames
        
        self.techo, self.vbasal, self.drift = self._main_PSanalysis(tag_pattern, direction, odb_tag)
    
        all_resp = opst.post.get_element_responses(odb_tag=odb_tag, ele_type="Frame")
        self.impPoints = self._pushover_analysisPoints(all_resp, self.techo, self.vbasal)
        
        # Procesar la informacion
        self._proccessInfo()
        
        # Obtener figura html
        self._plotLayout()
                
        
    def _main_PSanalysis(self, tag_pattern, direction, odb_tag):
        
        
        df_DCMetabs = self.model_data.displacements
        df_DiafMasses = self.model_data.masses
        df_joints = self.model_data.OE_joints
        
        altur = np.sort(pd.unique(df_DCMetabs['Z']))
        dnodes = [100000000*i for i in range(len(altur))]
        masas = df_DiafMasses['Mass X'].to_numpy()
        masas = masas/1000
        masas = list(masas)
                        
        coordz = [0]
        coordz.extend(altur)
                
        k = 1.0 # el k de toda la vida para la FHE
        Mest = np.sum(masas)
    
        # Rutina para calcular mh^k para cada piso
        fi = []
        for i in range(len(dnodes)):
            fi.append(masas[i]*coordz[i+1]**k)
        fi = np.array(fi)
    
        fact = fi/np.sum(fi) # se calcula el factor de corte para cada piso según la norma
                
        # Rutina para asignar las cargas
        ops.timeSeries('Linear', tag_pattern)
        ops.pattern('Plain',tag_pattern,tag_pattern)
        
        
        # Asignación de cargas con patrón triangular
        for i,n in enumerate(dnodes):
            if direction == 'X':
                ops.load(n,fact[i],0,0,0,0,0)
            elif direction == 'Y':
                ops.load(n,0,fact[i],0,0,0,0)
    
        # Asignación de cargas con patrón modal
        eigforce = np.zeros(len(dnodes))
        for j, n in enumerate(dnodes):
            dir_ = 1
            if direction == 'Y':
                dir_ = 2
            eigforce[j] = ops.nodeEigenvector(n, dir_, 1) 
            
        norm2 = np.sum(eigforce)
        eigforces  = eigforce/norm2
        
        for j, n in enumerate(dnodes):
            if direction == 'X':
                ops.load(n, eigforces[j], 0, 0, 0, 0, 0)
            elif direction == 'Y':
                ops.load(n, 0, eigforces[j], 0, 0, 0, 0)
        
        
        if direction == 'X':
            IDctrlDOF = 1 
        elif direction == 'Y':
            IDctrlDOF = 2 
            
        nodes_base = df_joints[df_joints['Story']=='Base'].reset_index()
        nodes_control = [int(nodes_base['Element Label'][0])]
        nodes_control.extend(dnodes)
    
        self.displacement, self.capacity, drifts = self._pushoveroutine(0.03*altur[-1], 0.001, dnodes[-1], IDctrlDOF, nodes_control, odb_tag)
                    
        dtecho = self.displacement*100/altur[-1]
        vbasal_norm = self.capacity/(Mest*9.81)
        
        self.alturas = altur
        
        return dtecho, vbasal_norm, drifts
            
    def _pushoveroutine(self, Dmax, Dincr, IDctrlNode, IDctrlDOF, nodes_control, odb_tag,
                        error = 1e-3, dincr_min = 1e-4, reduction_factor = 0.5, 
                        vbasal_limit = 0.65, drift_limit = 0.1):
        
        maxNumIter = 10 
        # 1). Cálculo de la tolerancia dado el Error y número de grados de libertad
        # propuesto po Michael H. Scott 
        Tol = error*np.sqrt(len(ops.getNodeTags())*6)

        # 2). Configuración básica del análisis
        ops.wipeAnalysis()
        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('BandGeneral')
        ops.test('NormUnbalance', Tol, maxNumIter)
        ops.algorithm('Newton')    
        ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr)
        ops.analysis('Static')
        
        # 3). Definir otras opciones de análisis    
        algoritmo = {1:'KrylovNewton', 2: 'SecantNewton' , 4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}

        # 4). Inicializar variables
        
        dtecho = [ops.nodeDisp(IDctrlNode,IDctrlDOF)] # Deriva de techo (lista)
        Vbasal = [ops.getTime()] # Vbasal (lista)
        drift = [] # Guardar derivas de piso (lista)
        
        Vmax = 0 # Contador de Vbasal máximo
        Dincr_red = Dincr # Paso reducido inicial
        step = 0 # Contador aumento de pasos
        
        # 5). Rutina de análisis
        # Genera un bucle infinito hasta que el modelo deje de converger de manera definitiva

        # ODB = opst.post.CreateODB(odb_tag=odb_tag, fiber_ele_tags="ALL")
        ODB = opst.post.CreateODB(odb_tag=odb_tag, elastic_frame_sec_points=5)

        while True:
            ok = ops.analyze(1)
            # 5.1). Probar convergencia con paso reducido (Intentar con pasos aún más pequeños)
            while ok != 0 and Dincr_red > dincr_min:
                Dincr_red *= reduction_factor # paso reducido
                ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr_red)
                
                ok = ops.analyze(1)
                
                # 5.1.1). Si converge con paso reducido, continua con paso original
                if ok == 0:
                    # Si converge, restaura el paso original y continua el análisis
                    ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr)
                    Dincr_red = Dincr
                    break
            
            # 5.2). Si no converge con paso reducido, intentar con otros algoritmos
            if ok != 0:                
                for j in algoritmo:
                    if j < 4:
                        ops.algorithm(algoritmo[j], '-initial')
        
                    else:
                        ops.algorithm(algoritmo[j])
                    
                    # El test se hace 50 veces más
                    ops.test('NormUnbalance', Tol, maxNumIter*50)
                    ok = ops.analyze(1)
                    # 5.2.1). Si converge con otro algoritmo, continua con algoritmo original
                    if ok == 0:
                        # Si converge vuelve a las opciones iniciales de análisis
                        ops.test('NormUnbalance', Tol, maxNumIter)
                        ops.algorithm('Newton')
                        ops.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr)
                        break
                    
            # 5.3). Si no hay convergencia de ningun modo, el análisis falló
            if ok != 0:
                tqdm.write(f'🔶​ Desplazamiento alcanzado: {round(ops.nodeDisp(IDctrlNode,IDctrlDOF),4)} m')
                break
            
            current_drift = []
            for node_i, node_tag in enumerate(nodes_control):
                if node_i != 0:
                    current_drift.append((ops.nodeDisp(node_tag, IDctrlDOF) - ops.nodeDisp(nodes_control[node_i - 1], IDctrlDOF)) / (ops.nodeCoord(node_tag, 3) - ops.nodeCoord(nodes_control[node_i - 1], 3)))
                
                
            ODB.fetch_response_step()

            dtecho.append(ops.nodeDisp(IDctrlNode,IDctrlDOF))
            Vbasal.append(ops.getTime())
            drift.append(current_drift)
            
            # 5.4). Actualizar Vmax: siempre y cuando encuentre un valor mas grande
            if Vbasal[-1] > Vmax:
                Vmax = Vbasal[-1]
                
            # 5.5). Comprobar si Vbasal se reduce al 75% de Vmax (75% por defecto)
            if Vbasal[-1] <= vbasal_limit * Vmax:
                print(f'Capacidad basal reducida al {int(vbasal_limit*100)}% de la máxima ({Vmax:.2f} kN). Terminando análisis...')
                break
            
            # 5.6). Comprobar que una deriva de entre piso llegó a 6% (6% por defecto)
            if max(drift[-1]) >= drift_limit: 
                piso_deriva = np.where(drift[-1] == max(drift[-1]))[0][0]
                print(f'La deriva del piso {piso_deriva} alcanzó un valor mayor o igual a {int(drift_limit*100)}%. Terminando análisis...')
                break
            
            # 5.7). Finalizar cuando el desplazamiento es mayor que el objetivo
            if ops.nodeDisp(IDctrlNode,IDctrlDOF) > Dmax:
                break
            
            step += 1 

        ODB.save_response()

        techo = np.array(dtecho)
        V = np.array(Vbasal)
        
        return techo, V, np.array(drift)
    
    def _pushover_analysisPoints(self, section_resp, techo, V,  
                                        steel_yield_strain=0.0021, conc_crack_strain=0.00013):
        """
        Analiza el pushover basado en las deformaciones de sección (no fibras).
        Detecta capacidad máxima, 80%, primer agrietamiento y primera fluencia.
    
        Args:
            section_resp (dict): Diccionario con las respuestas de sección (Deformations).
            techo (array): Desplazamientos del techo en cada paso.
            V (array): Cortante basal en cada paso.
            steel_yield_strain (float): Deformación que indica fluencia.
            conc_crack_strain (float): Deformación que indica agrietamiento de concreto.
        
        Returns:
            dict: Resultados de Vmax, V80, Dcrack, Dyield.
        """
            
        # Capacidad máxima
        Vmax_idx = np.argmax(V)
        Vmax = V[Vmax_idx]
        Dmax = techo[Vmax_idx]
    
        # Capacidad al 80%
        eighty_percent = 0.8 * Vmax
        idx_80 = np.where(V >= eighty_percent)[0][-1]
        D80 = techo[idx_80]
        idx_80 = np.where(techo == D80)[0][0]
        V80 = V[idx_80]
        
        # Deformaciones de sección
        deformations = section_resp["sectionDeformations"]  # (time, eleTag, secPoint, components)
        # components típicamente son: [ε11, ε22, γ12, κ11, κ22, κ12] (varía un poco dependiendo del elemento)
    
        # Vamos a suponer que la deformación relevante para agrietamiento es ε11 o ε22
        # Por ahora tomo ε11 (componente 0) para vigas/columnas
    
        strains_e1 = deformations[..., 0]  # Extraer ε11 (tensión normal eje 1)
    
        # Considerar sólo extremos (secPoints 0 y último)
        strains_extremos = strains_e1[:, :, [0, -1]]  # (time, eleTag, 2)
    
        # Aplanar para facilitar análisis
        strains_flat = strains_extremos.values.reshape(strains_extremos.shape[0], -1)  # (time, eleTag*2)
    
        # Detectar primer agrietamiento
        idx_crack = None
        for i in range(strains_flat.shape[0]):
            positive_strains = strains_flat[i, strains_flat[i, :] > 0]  # Sólo tracción
            if (positive_strains >= conc_crack_strain).any():
                idx_crack = i
                break
            
        if idx_crack is not None:
            Dcrack = techo[idx_crack]
            Vcrack = V[idx_crack]
        else:
            Dcrack = Vcrack = None
    
        # Detectar primer fluencia
        idx_yield = None
        for i in range(strains_flat.shape[0]):
            if (strains_flat[i, :] >= steel_yield_strain).any():
                idx_yield = i
                break
    
        if idx_yield is not None:
            Dyield = techo[idx_yield]
            Vyield = V[idx_yield]
        else:
            Dyield = Vyield = None
    
        # Resultados
        result = {
            # key : (capacidad normalizada, deriva de techo)
            'capacidad_maxima': (Vmax, Dmax),
            'capacidad80': (V80, D80),
            'fluencia_acero': (Vyield, Dyield),
            'agrietamiento_concreto': (Vcrack, Dcrack)
        }
        
        return result
    
    def _proccessInfo(self):
        
        # Derivas 1% por piso y maximas
        idx_drift, max_drifts = [], []
        for i in range(np.size(self.drift,axis=1)):
            drift_story = self.drift[:,i]*100
            max_drifts.append(max(drift_story))
            for idx, drf in enumerate(drift_story):
                if drf >= 1:
                    idx_drift.append(idx)
                    break
                
        if len(idx_drift) > 0:
            for i, idx in enumerate(idx_drift):
                if idx is not None:
                    self.impPoints[f'1deriva_piso{i+1}'] = (self.vbasal[idx+1], self.techo[idx+1])
                else:
                    self.impPoints[f'1deriva_piso{i+1}'] = None
                    
        for i in range(len(max_drifts)):
            key = f'1deriva_piso{i+1}'
            self.point_style[key] = {
                'name': f'1% de deriva en el piso {i+1}',
                'marker-color': ut.generar_color_aleatorio(),
                'marker-size': 9,
                'marker-symbol': 'circle'
            }
                    
        self.fig_animation = opsvis.plot_nodal_responses_animation(
                                odb_tag=self.odb_Tag,
                                resp_type="disp",
                                resp_dof=["UX", "UY"],
                                framerate=30,
                                scale=3.0
                            )
        
        self.maxDrifts = max_drifts
        self.idx_drift = idx_drift
        
    
    def _plotLayout(self):
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],   # Fila 1: Gráfico + Grafico
                [{"type": "scene", "colspan": 2}, None]  # Fila 2: Animación ocupando dos columnas
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.105,
            row_heights=[0.5, 0.5],  # Opcional, para controlar proporciones
            subplot_titles=("Curva de capacidad de la estructura", "Perfil de derivas máximas de entre piso", "Deformación de la estructura por método Pushover")
        )
        
        customdata = list(zip(self.techo, self.vbasal))
        
        # Graficar curva de capacidad inicial
        fig.add_trace(go.Scatter(
            x = self.techo,
            y = self.vbasal,
            mode = 'lines',
            name = 'Curva de capacidad',
            line = dict(color='black'),
            text = [
                    f"<span style='font-size:14px; color:black;'>"
                    f"<b>Deriva de techo = </b> {row[0]:.2f}% <br>"
                    f"<b>Capacidad basal normalizada = </b> {row[1]:.2f} " 
                    f"</span>"
                    for row in customdata
                ],
            hoverinfo='text'
        ), row=1, col=1)
        
        self.customdata = pd.DataFrame(self.impPoints)
        
        # Graficar los demas puntos
        self.pushover_points(fig, 'capacidad_maxima')
        self.pushover_points(fig, 'capacidad80')
        self.pushover_points(fig, 'fluencia_acero')
        self.pushover_points(fig, 'agrietamiento_concreto')
        
        # Graficar puntos derivas
        for i in range(len(self.idx_drift)):
            self.pushover_points(fig, f'1deriva_piso{i+1}')
        
        # Graficar perfil de derivas maximas
        
        fig.add_trace(go.Scatter(
            x = self.maxDrifts,
            y = self.alturas,
            mode = 'lines+markers',
            line = dict(color = 'black', width = 2.0),
            marker = dict(color = 'black', 
                          size = 9, 
                          symbol = 'circle'),
            hovertemplate = (
                    "<span style='font-size:14px; color:gray;'>"
                    "<b>Deriva máxima</b> = %{x:.2f}% <br>"
                    "<b>Altura</b> = %{y:.2f} m <br>"
                    "</span><extra></extra>"
                ),
            hoverlabel=dict(bgcolor='rgba(250,250,250,1.0)'),
            showlegend = False
        ), row=1, col=2)
        
        # Agregar animacion
        fig_animation = self.fig_animation
 
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
        
            
        
        # Read image and convert to base64
        logo_path = os.path.join(self.temporal_path, 'logo_estrucmed.png')
        with open(logo_path, 'rb') as image_file:
            logo_base64 = base64.b64encode(image_file.read()).decode()
        
        # Configure layout
        fig.update_layout(
            height=1500,
            margin=dict(
                l=190,   # left
                r=190,   # right
                t=250,   # top
                b=60    # bottom
            ),
            plot_bgcolor='white',
            hovermode='x unified',
            title={
                'text': "<br><b>ANALISIS DEL MODELO OPENSEES</b><br>"+
                        f"<span style='font-size:20px; color:#3A3A3A'>Resultados análisis pushover en dirección {self.dir}</span><br><br>",
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
                x=0.01, y=1.15,
                sizex=0.15, sizey=0.15,
                xanchor="left", yanchor="top",
                opacity=0.8,
                layer="above"
            )],
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
            # legend=dict(
            #     orientation='h',
            #     x=0.22,
            #     y=0.44,
            #     xanchor='center',
            #     yanchor='top',
            #     traceorder='normal',
            #     itemwidth=30
            # ),
            xaxis=dict(
                title='Deriva de techo (%)',
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
                title='Capacidad normalizada (vbasal/W)',
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
            ),
            xaxis2=dict(
                title='Máxima deriva entre piso (%)',
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
            yaxis2=dict(
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
            ),
        )
                        
        # Export 
        output_file = os.path.join(os.path.join(self.push_path, 'html'), f'pushover_analysis_{self.dir}direction.html')
        fig.write_html(output_file)
        
    def pushover_points(self, fig, key):
        
        customdata = self.customdata
        
        style = self.point_style[key]
        
        fig.add_trace(go.Scatter(
            x = [self.impPoints[key][1]],
            y = [self.impPoints[key][0]],
            mode = 'markers',
            name = style['name'],
            marker = dict(color = style['marker-color'], 
                          size = style['marker-size'], 
                          symbol = style['marker-symbol']),
            text=[
                    f"<span style='font-size:14px; color:gray;'>"
                    f"<b>{style['name']} </b> <br>"
                    f"Capacidad basal normalizada = {customdata[key][0]:.2f} <br>"
                    f"Deriva de techo = {customdata[key][1]:.2f}%"
                ],
            hoverinfo='text',
            showlegend = False
        ), row=1, col=1)
    
