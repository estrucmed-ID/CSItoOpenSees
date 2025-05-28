"""
@author: Daniela Novoa, Orlando Arroyo, Frank Vidales
@owner: EstrucMed Ingeniería especializada S.A.S

-------------------------------------------------------------------------------
                      CSI Model Converter to OpenSeesPy
                          Versión: ETABS17 v17.0.1
                            
    Módulo principal para convertir modelos CSI a OpenSeesPy a partir de
                           datos exportados de CSI

Este script contiene la clase principal `ModelBuilder`, encargada de organizar 
y ejecutar las rutinas necesarias para crear el modelo completo en OpenSeesPy 
desde la información procesada de CSI. Incluye desde la lectura de nodos 
hasta la definición de diafragmas rígidos.

También contiene dos clases especializadas:

    • `ElasticModelBuilder`: genera modelos elásticos con secciones y elementos 
      simplificados, ideal para análisis modales o lineales rápidos.
    
    • `NONLinearModelBuilder`: genera modelos no lineales con secciones de fibras 
      y materiales constitutivos detallados, adecuado para análisis pushover 
      o dinámicos inelásticos.

Ambas clases heredan de `ModelBuilder` para aprovechar las rutinas compartidas 
y asegurar coherencia en la construcción del modelo estructural.

Este módulo permite:
    
    1. Leer la información procesada del modelo CSI (formato json).
    2. Crear el modelo estructural completo en OpenSeesPy.
    3. Generar nodos, secciones, elementos, cargas y diafragmas.

Unidades: Sistema Internacional (SI) - metros (m)
Norma utilizada: NSR-10 (adaptable según el origen del modelo CSI).
-------------------------------------------------------------------------------

"""   
#%% ==> IMPORTAR LIBRERIAS 
   
import pandas as pd
import openseespy.opensees as ops
import opstool as opst
import opseestools.utilidades as optools_ut
import opseestools.analisis3D as an
import opstool.vis.plotly as opsvis
import sys
import os
import numpy as np
import math
import re
from tqdm import tqdm

utils_path = os.getcwd()
sys.path.append(utils_path)

# Importar módulos personalizados
import utilities as ut  

#%% ==> CLASE PRINCIPAL - FUNCIONES QUE GENERAN EL MODELO
class ModelBuilder:
    
    def __init__(self, data_file):
        
        self.data_file = data_file
        
        # Objects and elements tables
        self.OE_joints = data_file['TABLE:  "OBJECTS AND ELEMENTS - JOINTS"']
        self.OE_shells = data_file['TABLE:  "OBJECTS AND ELEMENTS - SHELLS"']
        self.OE_frames = data_file['TABLE:  "OBJECTS AND ELEMENTS - FRAMES"']
        
        # Assignments
        self.AS_restraints = data_file['TABLE:  "JOINT ASSIGNMENTS - RESTRAINTS"']
        self.AS_shells = data_file['TABLE:  "SHELL ASSIGNMENTS - SECTIONS"']
        self.AS_frames = data_file['TABLE:  "FRAME ASSIGNMENTS - SECTIONS"']
        self.AS_axes = data_file['TABLE:  "FRAME ASSIGNMENTS - LOCAL AXES"']
        self.AS_offsets = data_file['TABLE:  "FRAME ASSIGNMENTS - OFFSETS"']
        self.AS_pier = data_file['TABLE:  "SHELL ASSIGNMENTS - PIER SPANDR"']
        
        # Material properties
        self.material = data_file['TABLE:  "MATERIAL PROPERTIES - CONCRETE"']
        self.C_rebar = data_file['TABLE:  "CONCRETE COLUMN REBAR DATA"']
        self.B_rebar = data_file['TABLE:  "CONCRETE BEAM REBAR DATA"']
        self.pier_rebar = data_file['TABLE:  "SHEAR WALL PIER SUMMARY - ACI 3"']
        
        # Sections
        self.SC_shells = data_file['TABLE:  "SHELL SECTIONS - SLAB"']
        self.SC_frames = data_file['TABLE:  "FRAME SECTIONS"']
        self.SC_walls = data_file['TABLE:  "SHELL SECTIONS - WALL"']
        self.SC_piers = data_file['TABLE:  "PIER SECTION PROPERTIES"']
        
        # Cargas
        self.LD_shells = data_file['TABLE:  "SHELL LOADS - UNIFORM"']
        self.LD_frames = data_file['TABLE:  "FRAME LOADS - DISTRIBUTED"']
        
        # Modal Analisis
        self.masses = data_file['TABLE:  "MASS SUMMARY BY DIAPHRAGM"']
        self.modal = data_file['TABLE:  "MODAL PARTICIPATING MASS RATIOS"']
        
        # Base reactions and drift analisis
        self.column_forces = data_file['TABLE:  "COLUMN FORCES"']
        self.pier_forces = data_file['TABLE:  "PIER FORCES"']
        self.functions = data_file['TABLE:  "RS FUNCTION - COLOMBIA NSR-10"']
        self.displacements = data_file['TABLE:  "DIAPGRAGM CENTER OF MASS DISPLACEMENT"']
        
        # Return dictionary
        self.dict = {}
    
    # Cleans and filters the joint data
    def _debbugingJoints(self):
        """
        Cleans and filters the joint data to remove unused or disconnected nodes 
        before generating them in OpenSeesPy.
    
        This function ensures that only joints connected to at least one frame or shell 
        are kept. Disconnected nodes are removed to avoid errors during model generation.
    
        Returns:
            Updates self.OE_joints (DataFrame): The cleaned list of joints to be used in the model.
            Also updates self.dict with the cleaned joint data.
            
        """
        # Filter only rows that correspond to actual joints
        joints = self.OE_joints[self.OE_joints["Object Type"] == "Joint"]
     
        # Get all joints that are part of frames (Joint I and Joint J)
        # ** Modificacion considerando que el modelo tiene frames **
        alone_joints1 = None
        if isinstance(self.OE_frames, pd.DataFrame):
            unique_jframes = pd.concat([
                self.OE_frames['Joint I'], 
                self.OE_frames['Joint J']
            ]).unique()
            
            # Identify joints that are not in frame elements
            alone_joints1 = [x for x in joints['Element Label'].tolist() if x not in unique_jframes]
     
        # Get all joints that are part of shell elements (4-node shells)
        unique_jshells = pd.concat([
            self.OE_shells['Joint 1'], 
            self.OE_shells['Joint 2'], 
            self.OE_shells['Joint 3'], 
            self.OE_shells['Joint 4']
        ]).unique()
     
        # Identify joints that are not in shell elements
        alone_joints2 = [x for x in joints['Element Label'].tolist() if x not in unique_jshells]
     
        # Find joints that are not connected to any element (not in frames nor shells)
        if alone_joints1:
            drop_joints = [x for x in alone_joints1 if x in alone_joints2]
        else:
            drop_joints = alone_joints2
     
        # Remove unused joints from the DataFrame
        joints = joints[~joints['Element Label'].isin(drop_joints)]
        
        # Store the cleaned data in the internal dictionary for further use
        self.dict['TABLE:  "OBJECTS AND ELEMENTS - JOINTS"'] = joints[~joints['Element Label'].isin(drop_joints)]
     
        # Update the class variable with the cleaned joint data
        self.OE_joints = joints
     
        
    # Generates the nodes
    def _genNodes(self):
        """
       Generates the nodes for the OpenSeesPy model using joint data from ETABS.
    
       Each node is created based on the coordinates and labels defined in the 
       OE_joints DataFrame. The node tags correspond to the 'Element Label' 
       used in ETABS, preserving consistency between platforms.
    
       Returns:
           None. This function creates OpenSees nodes directly using ops.node().
       """
       
        # Extract coordinates from OE_joints DataFrame
        self.xcoord = self.OE_joints['Global X'].to_numpy() # x coordinate
        self.ycoord = self.OE_joints['Global Y'].to_numpy() # y coordinate
        self.zcoord = self.OE_joints['Global Z'].to_numpy() # z coordinate
        
        # The tags of the generated nodes (nodes_label) correspond to the “Element Label” of the Joint defined by ETABS.
        nlabel = self.OE_joints['Element Label'].to_numpy(dtype=int) # Joint Tag
        
        # Cycle to create the nodes of the model
        nnodes = len(self.OE_joints)
        
        for i in range(nnodes):
            ops.node(int(nlabel[i]),float(self.xcoord[i]),float(self.ycoord[i]),float(self.zcoord[i]))
    
    # Processes and formats the restraint data 
    def _debugging_restr(self):
        """
        Processes and formats the restraint data from ETABS to be used in OpenSeesPy.
    
        This function:
        - Renames the column 'Unique Name' to 'Element Label' for consistency.
        - Converts the string-based restraint conditions ('Yes'/'No') into binary values [1/0]
          for each degree of freedom: UX, UY, UZ, RX, RY, RZ.
        - Stores the cleaned and formatted data in self.AS_restraints.
        - Updates the internal dictionary with the formatted restraint table.
    
        Returns:
            None. Updates self.AS_restraints and self.dict with cleaned data.
        """
        
        # Rename column to match the expected joint identifier
        self.AS_restraints.rename(columns={'Unique Name': 'Element Label'}, inplace=True)
    
        constrValues_list = []
        # Iterate through each row to extract and convert restraint flags
        for indx, row in self.AS_restraints.iterrows():
            # Extract values for each DOF restraint
            ListValues = [row['UX'], row['UY'], row['UZ'], row['RX'], row['RY'], row['RZ']]
    
            # Convert 'Yes'/'No' to binary: 1 = restrained, 0 = free
            constrValues = [1 if val == 'Yes' else 0 for val in ListValues]
    
            # Append the converted values
            constrValues_list.append(constrValues)
    
        # Add binary restraint values to the DataFrame
        self.AS_restraints['constrValues'] = constrValues_list
    
        # Keep only necessary columns for modeling
        self.AS_restraints = self.AS_restraints[['Story', 'Element Label', 'constrValues']]
    
        # Store the cleaned data in the internal dictionary
        self.dict['TABLE:  "JOINT ASSIGNMENTS - RESTRAINTS"'] = self.AS_restraints
    
    # Applies boundary conditions 
    def _fixNodes(self):
        """
        Applies boundary conditions (restraints) to the nodes in the OpenSeesPy model.
    
        For each joint defined in the `self.AS_restraints` DataFrame, the corresponding 
        node is fixed according to the 'constrValues' list, which defines constraints 
        on all six degrees of freedom: [UX, UY, UZ, RX, RY, RZ].
    
        Returns:
            None. This function calls `ops.fix()` for each node.
        """
        # Iterate through each restraint entry and apply fixity conditions
        for indx, row in self.AS_restraints.iterrows():            
            nodetag = int(row['Element Label'])    # Node tag
            ops.fix(nodetag, *row['constrValues']) # Apply constraints (1 = fixed, 0 = free)
    
    # Links section definitions to their corresponding material properties 
    def _debugging_materials(self):
        """
        Links section definitions to their corresponding material properties 
        for both frame and shell elements.
    
        This function:
        - Renames the material column to 'Material' for consistency.
        - Merges frame sections with concrete material properties (Fc, G).
        - Merges shell sections with concrete strength (Fc) only.
        - Renames 'Name' columns to 'Section' for clarity.
        - Stores the processed data in the internal dictionary for further use.
    
        Returns:
            None. Updates self.SC_frames, self.SC_shells, and self.dict.
        """
    
        # Standardize material column name for consistent merging
        self.material.rename(columns={'Name': 'Material'}, inplace=True)
    
        # Merge frame sections with material properties (Fc and G)
        # ** Modificacion considerando que el modelo tiene frames **
        SC_frames = 'Table Not Found'
        if isinstance(self.SC_frames, pd.DataFrame):
            SC_frames = pd.merge(
                self.SC_frames, 
                self.material[['Material', 'Fc', 'G']],
                on='Material',
                how='inner'
            )
            SC_frames.rename(columns={'Name': 'Section'}, inplace=True)
            
        # ** Modificacion considerando que el modelo tiene walls **
        SC_walls = 'Table Not Found'
        if isinstance(self.SC_walls, pd.DataFrame):
            SC_walls = pd.merge(
                self.SC_walls, 
                self.material[['Material', 'Fc', 'G']],
                on='Material',
                how='inner'
            )
            SC_walls.rename(columns={'Name': 'Section'}, inplace=True)
        
        # Merge shell (slab) sections with material strength (Fc only)
        SC_shells = pd.merge(
            self.SC_shells, 
            self.material[['Material', 'Fc']],
            on='Material',
            how='inner'
        )
        SC_shells.rename(columns={'Name': 'Section'}, inplace=True)
    
        # Store the processed section tables in the internal dictionary
        self.dict['TABLE:  "FRAME SECTIONS"'] = SC_frames
        self.dict['TABLE:  "SHELL SECTIONS - WALL"'] = SC_walls
        self.dict['TABLE:  "SHELL SECTIONS - SLAB"'] = SC_shells
        
        self.SC_frames = SC_frames
        self.SC_walls = SC_walls
        self.SC_shells = SC_shells
        
    def _assgNonlinearMaterials(self, ed_type):
        """
        Assigns material tags and generates concrete and steel materials for frame sections.
    
        For each frame section:
        - Extracts concrete strength (f'c) in MPa.
        - Creates unique tags for steel, confined concrete, and unconfined concrete materials.
        - Calls `ut.col_materials()` to define and register materials in OpenSees.
        - Stores the returned material tags in lists for future assignment.
    
        Returns:
            None. Updates internal lists of material tags (Unconf_Tag, Conf_Tag, Steel_Tag).
        """
    
        def generate_nonlinear_materials(df):
            Unconf_Tag, Conf_Tag, Steel_Tag = [], [], []
            
            for index in range(len(df)):
                aa = df.iloc[index]
                fc = aa['Fc'] / 1000  # Convert f'c from kPa to MPa
        
                # Define unique tags for materials
                Tag_Steel = index
                Tag_UnConf = len(df) + index
                Tag_Conf = len(df) * 2 + index
        
                # Call utility to define materials and retrieve their tags
                unctag, conftag, steeltag = optools_ut.col_materials(
                    fcn = fc,
                    detailing = ed_type, 
                    steeltag = Tag_Steel,
                    unctag = Tag_UnConf, 
                    conftag = Tag_Conf
                )
        
                # Store material tags
                Unconf_Tag.append(unctag)
                Conf_Tag.append(conftag)
                Steel_Tag.append(steeltag)
            
            return Unconf_Tag, Conf_Tag, Steel_Tag
    
        # Iterate over all frame sections (if their exist in the model)
        if isinstance(self.SC_frames, pd.DataFrame):
            Unconf_Tag, Conf_Tag, Steel_Tag = generate_nonlinear_materials(self.SC_frames)
            self.SC_frames['Steel_Label'] = Steel_Tag
            self.SC_frames['UnConf_Label'] = Unconf_Tag
            self.SC_frames['Conf_Label'] = Conf_Tag
        
        # Iterate over all wall sections (if their exist in the model)
        if isinstance(self.SC_walls, pd.DataFrame):
            Unconf_Tag, Conf_Tag, Steel_Tag = generate_nonlinear_materials(self.SC_walls)
            self.SC_walls['Steel_Label'] = Steel_Tag
            self.SC_walls['UnConf_Label'] = Unconf_Tag
            self.SC_walls['Conf_Label'] = Conf_Tag
    
    def _assgWallElasticMaterials(self):
        
        # Slabs and Walls Joint information 
        OE_slabs = self.OE_shells[self.OE_shells['Area Type'] == 'Floor']
        OE_walls = self.OE_shells[self.OE_shells['Area Type'] == 'Wall']
        
        # Assing to each slab object it's respective section and properties
        AS_shells = self.AS_shells.copy()
        AS_shells.rename(columns = {'Unique Name':'Element Label'}, inplace = True)
        SC_shells = self.SC_shells.copy()
        
        OE_slabs = pd.merge(OE_slabs, AS_shells[['Element Label', 'Section']], on = 'Element Label', how = 'inner')
        OE_slabs = pd.merge(OE_slabs, SC_shells, on = 'Section', how = 'inner')
        
        # Assing to each wall object it's respective section and properties
        SC_walls = self.SC_walls.copy()

        OE_walls = pd.merge(OE_walls, AS_shells[['Element Label', 'Section']], on = 'Element Label', how = 'inner')
        OE_walls = pd.merge(OE_walls, SC_walls, on = 'Section', how = 'inner')
        
        # Store the coordinates of nodes of each wall
        nodesW = []
        for i, row in OE_walls.iterrows():
            p1 = int(row['Joint 1'])
            p2 = int(row['Joint 2'])
            p3 = int(row['Joint 3'])
            p4 = int(row['Joint 4'])
            nodesW.append([p1,p2,p3,p4])
        
        # Store list whit the order of the wall nodes and lenght of the walls
        order_list, lenght_list = [], []
        for nodes in nodesW:
            orientation, lenght, node = self.wallsOrientation(nodes)
            lenght_list.append(lenght) # Wall lenght
            order_list.append(node)    # Wall orientation
        
        OE_walls['Nodes Orientation'] = order_list # Add to the dataframe the walls orientation
        OE_walls['Nodes Lenght'] = lenght_list     # Add to the dataframe the walls length
        
        tag_initial = int(min(OE_slabs['Slab_Label']))
        tagshear_list = []
        for i, row in OE_walls.iterrows():
            fc = row['Fc']
            E = 1000*4400*(fc/1000)**0.5
            tagshear = int(tag_initial-(1+i))
            G = E*0.4
            ops.uniaxialMaterial('Elastic', tagshear, G*1.5)
            tagshear_list.append(tagshear)
        
        OE_walls['Shear_Label'] = tagshear_list
        
        # Store the processed section tables in the internal dictionary
        self.dict['TABLE:  "OBJECTS AND ELEMENTS - WALL"'] = OE_walls
        self.dict['TABLE:  "OBJECTS AND ELEMENTS - SLAB"'] = OE_slabs
        
        self.OE_walls = OE_walls
        self.OE_slabs = OE_slabs
        self.GN_Slabs = self.OE_slabs
        
        self.dict['TABLE:  "SLAB GENERATION"'] = self.GN_Slabs
        
    def wallsOrientation(self, nodes):
        
        order = []
        nodeA = nodes[0]
        nodeB = nodes[1]
        nodeC = nodes[2]
        nodeD = nodes[3]
        CA = ops.nodeCoord(nodeA)
        CB = ops.nodeCoord(nodeB)
        CC = ops.nodeCoord(nodeC)
        CD = ops.nodeCoord(nodeD)
        # en la versión 17 de ETABS, van de primero los nodos de abajo y de segundo los de arriba
        if CA[0] == CB[0]:
            orient = 2
            longitud = abs(CB[1] - CA[1])
            # long.append(longitud)
            if CA[1] < CB[1]:
                order.append(nodeA)
                order.append(nodeB)
                if CC[1] < CD[1]:
                    order.append(nodeD)
                    order.append(nodeC)
                elif CD[1] < CC[1]:
                    order.append(nodeC)
                    order.append(nodeD)
                    
            if CB[1] < CA[1]:
                order.append(nodeB)
                order.append(nodeA)
                if CC[1] < CD[1]:
                    order.append(nodeD)
                    order.append(nodeC)
                elif CD[1] < CC[1]:
                    order.append(nodeC)
                    order.append(nodeD)
            
        elif CA[1] == CB[1]:
            orient = 1
            longitud = abs(CB[0] - CA[0])
            
            if CA[0] < CB[0]:
                order.append(nodeA)
                order.append(nodeB)
                if CC[0] < CD[0]:
                    order.append(nodeD)
                    order.append(nodeC)
                elif CD[0] < CC[0]:
                    order.append(nodeC)
                    order.append(nodeD)
                    
            if CB[0] < CA[0]:
                order.append(nodeB)
                order.append(nodeA)
                if CC[0] < CD[0]:
                    order.append(nodeD)
                    order.append(nodeC)
                elif CD[0] < CC[0]:
                    order.append(nodeC)
                    order.append(nodeD)
        
        return orient,longitud,order
    
    # Assigns elastic membrane plate sections
    def _generateElasticSlabs(self, shell_craking):
        """
        Assigns elastic membrane plate sections to shell (slab) elements in the model.
    
        For each shell section, this function:
        - Calculates the modulus of elasticity based on the concrete strength (Fc),
          using the empirical formula E = 4700 * sqrt(fc) [MPa], converted to kPa.
        - Applies a user-defined cracking factor (shell_craking) to reduce stiffness.
        - Creates a unique negative section tag for each slab section (as per OpenSees convention).
        - Defines the shell section using ElasticMembranePlateSection in OpenSeesPy.
    
        Parameters:
            shell_craking (float): Cracking factor to reduce the stiffness of the shell.
    
        Returns:
            None. Updates OpenSees with shell section definitions and adds 'Slab_Label' column to SC_shells.
        """
    
        Slab_Tag = []
    
        # Iterate through each shell section to define its material and section
        for index in range(len(self.SC_shells)):
            aa = self.SC_shells.iloc[index]
            fc = aa['Fc']                              # Concrete compressive strength [kPa]
            hslab = aa['Slab Thickness']               # Thickness of the slab [m]
    
            # Modulus of elasticity using empirical formula (converted to kPa)
            Eslab = 1000 * 4700 * (fc / 1000) ** 0.5
    
            # Negative integer tag to identify shell sections (OpenSees convention)
            seclosa = int(-(index + 1))
    
            # Define shell section in OpenSees
            ops.section('ElasticMembranePlateSection', seclosa, shell_craking * Eslab, 0.3, hslab, 0.0)
    
            Slab_Tag.append(seclosa)
    
        # Store the assigned section tags in the DataFrame
        self.SC_shells['Slab_Label'] = Slab_Tag
    
    def _debugging_FiberSection(self):
                
        # Step 1). Merge section data with relevant properties for columns
        self.C_rebar.rename(columns={'Frame Property':'Section'}, inplace=True)
        self.C_rebar = pd.merge(
            self.C_rebar, 
            self.SC_frames[['Section','t3','t2','Steel_Label','UnConf_Label','Conf_Label','Fc', 'Area']],
            on='Section',
            how='inner')
        
        # Step 2). Merge section data with relevant properties for beams
        self.B_rebar.rename(columns={'Frame Property':'Section'}, inplace=True)
        self.B_rebar = pd.merge(
            self.B_rebar, 
            self.SC_frames[['Section','t3','t2','Steel_Label','UnConf_Label','Conf_Label','Fc', 'Area']],
            on='Section',
            how='inner')
        
        # Store the processed section tables in the internal dictionary
        self.dict['TABLE:  "CONCRETE COLUMN REBAR DATA"'] = self.C_rebar
        self.dict['TABLE:  "CONCRETE BEAM REBAR DATA"'] = self.B_rebar
    
    def _generateFiberSections(self, npint):
        
        # Initialize tags for column fiber sections
        colsectags = []
        for index in range(len(self.C_rebar)):
            aa = self.C_rebar.iloc[index]
            steel_area = aa['Corner Bar Area']/1000000 # --> Area (mm^2)
            
            id_tag = 1000 + index
            
            optools_ut.BuildRCSection(id_tag, aa['t3'], aa['t2'], aa['Cover'], aa['Cover'], int(aa['Conf_Label']), 
                              int(aa['UnConf_Label']), int(aa['Steel_Label']), int(aa['# Long. Bars 2-axis']), 
                              steel_area, int(aa['# Long. Bars 2-axis']), steel_area, 
                              int(2*(aa['# Long. Bars 3-axis']-2)), steel_area, 12, 12, 8, 8)
            # Add integration points for the column section
            ops.beamIntegration('Lobatto', id_tag, id_tag, npint)
    
            colsectags.append(id_tag)
            
        # Add column tags to the DataFrame
        self.C_rebar['sectag'] = colsectags
        
        # Initialize tags for beam fiber sections
        ncols = len(colsectags)
        beamsectags = []
        for ind1 in range(len(self.B_rebar)):
            tagbeam = 1000 + (ind1+ncols)
            aa = self.B_rebar.iloc[ind1]
            steel_top = aa['area top']/1000000
            steel_bot = aa['area bottom']/1000000
            
            optools_ut.BuildRCSection(tagbeam, aa['t3'], aa['t2'], aa['Top Cover'], aa['Top Cover'], int(aa['Conf_Label']), 
                              int(aa['UnConf_Label']), int(aa['Steel_Label']), int(aa['#top']), steel_top, 
                              int(aa['#bottom']), steel_bot, 2, 1e-14, 12, 12, 8, 8)
            # Add integration points for the beam section
            ops.beamIntegration('Lobatto', tagbeam, tagbeam, npint)

            beamsectags.append(tagbeam)
            
        # Add beam tags to the DataFrame
        self.B_rebar['sectag'] = beamsectags
        
        self.SC_frames = pd.concat([self.C_rebar[['Section', 't3', 't2', 'sectag', 'Area']],self.B_rebar[['Section', 't3', 't2', 'sectag', 'Area']]])
        
        
    # Generates elastic section definitions for frame elements
    def _generateElasticFrames(self, npint):
        """
        Generates elastic section definitions for frame elements in the OpenSeesPy model.
    
        For each frame section:
        - Computes the modulus of elasticity based on the concrete compressive strength (Fc).
        - Defines an 'Elastic' section in OpenSees using geometric and material properties.
        - Assigns a unique section tag.
        - Applies 'Lobatto' integration with a predefined number of integration points (npint).
        - Stores the section tag in the SC_frames DataFrame.
    
        Returns:
            None. Updates OpenSees with section and integration definitions, and adds 'sectag' column to SC_frames.
        """
    
        sectags = []
    
        # Iterate over all frame sections
        for i, row in self.SC_frames.iterrows():
            fc = row['Fc'] / 1000                         # Convert Fc from kPa to MPa
            E = 4700 * np.sqrt(fc) * 1000                 # Calculate modulus of elasticity in kPa
    
            id_tag = 1000 + i                             # Unique section ID
    
            # Define an elastic section in OpenSees
            ops.section(
                'Elastic', id_tag, E,
                row['Area'], row['I33'], row['I22'],
                row['G'], row['J']
            )
    
            # Define integration scheme for force-based elements
            ops.beamIntegration('Lobatto', id_tag, id_tag, npint)
    
            # Store section tag
            sectags.append(id_tag)
    
        # Add section tags to the frame section DataFrame
        self.SC_frames['sectag'] = sectags
    
    # Consolidates and prepares frame and shell element information for model generation
    def _consolidateInfo(self):
        """
        Consolidates and prepares frame and shell element information for model generation.
    
        This function:
        - Cleans and formats the frame data, resolving missing joint references.
        - Merges frame data with section and design information from assignments.
        - Handles optional local axes and offset definitions, applying default values if not available.
        - Merges geometric properties (area, thickness, tags) from section definitions.
        - Computes 3D coordinates for start and end joints (CoordI and CoordJ).
        - Consolidates slab information for further processing.
    
        Returns:
            None. Updates internal attributes:
            - self.GN_frames: complete frame element data
            - self.GN_Slabs: complete slab element data
            - self.nofound: flag indicating if offset table was found
        """
    
        # === Step 0: Check if local axis table is available ===
        Frame_LocalAxes = 0
        if isinstance(self.AS_axes, str) and self.AS_axes == "Table Not Found":
            Frame_LocalAxes = 0
        else:
            Frame_LocalAxes = 1
    
        # === Step 1: Clean and format frame element data ===
        self.OE_frames = self.OE_frames[self.OE_frames['Object Type'] == 'Frame'].copy()
        self.OE_frames['Element Label'] = self.OE_frames['Element Label'].astype(str).apply(ut.modify_element_label_as_int)
        self.OE_frames['Joint I'] = pd.to_numeric(self.OE_frames['Joint I'], errors='coerce')  # Convert to numeric
        self.OE_frames['Joint J'] = pd.to_numeric(self.OE_frames['Joint J'], errors='coerce')
    
        # Resolve missing joint values (caused by wrong label formatting)
        missing_joints = self.OE_frames[self.OE_frames['Joint I'].isna() | self.OE_frames['Joint J'].isna()]
        resolved_rows = (
            missing_joints.groupby(['Object Label', 'Story'])
            .agg({'Joint I': 'first', 'Joint J': 'first', 'Element Label': 'first'})
            .reset_index()
        )
        resolved_rows['Object Type'] = 'Frame'
    
        # Replace problematic rows with resolved data
        self.OE_frames = pd.concat([self.OE_frames.drop(missing_joints.index), resolved_rows], ignore_index=True)
    
        # === Step 2: Merge section assignments ===
        self.AS_frames = self.AS_frames.rename(columns={
            'Unique Name': 'Element Label',
            'Label': 'Object Label',
            'Analysis Section': 'Section'
        })
        self.OE_frames = pd.merge(
            self.OE_frames,
            self.AS_frames[['Object Label', 'Story', 'Section', 'Design Type']],
            on=['Object Label', 'Story'],
            how='inner'
        )
    
        # === Step 3: Merge local axes data (if available) ===
        if Frame_LocalAxes != 0:
            self.AS_axes = self.AS_axes[self.AS_axes['Design Type'] == 'Column'].rename(columns={'Unique Name': 'Element Label'})
            self.OE_frames = pd.merge(self.OE_frames, self.AS_axes[['Element Label', 'Angle']], on='Element Label', how='left')
    
            # Fill missing angle values with zero
            Angles = [0 if math.isnan(row['Angle']) else row['Angle'] for _, row in self.OE_frames.iterrows()]
        else:
            # If no axis table exists, use angle = 0 for all
            Angles = [0] * len(self.OE_frames)
    
        self.OE_frames['Angle'] = Angles
    
        # === Step 4: Merge with section geometry data (Area, section, tag) ===
        self.GN_frames = pd.merge(
            self.OE_frames,
            self.SC_frames[['Section', 't3', 't2', 'sectag', 'Area']],
            on='Section',
            how='inner'
        )
    
        # === Step 5: Compute coordinates of start joints (CoordI) ===
        coordI_list = []
        for i in range(len(self.GN_frames)):
            data = self.GN_frames.iloc[i]
            index = np.where(self.OE_joints['Element Label'] == data['Joint I'])[0][0]
            coordI_list.append([self.xcoord[index], self.ycoord[index], self.zcoord[index]])
    
        self.GN_frames['CoordI'] = coordI_list
    
        # === Step 6: Compute coordinates of end joints (CoordJ) ===
        coordJ_list = []
        for i in range(len(self.GN_frames)):
            data = self.GN_frames.iloc[i]
            index = np.where(self.OE_joints['Element Label'] == data['Joint J'])[0][0]
            coordJ_list.append([self.xcoord[index], self.ycoord[index], self.zcoord[index]])
    
        self.GN_frames['CoordJ'] = coordJ_list
    
        # === Step 7: Process optional offsets (OffsetI and OffsetJ) ===
        if isinstance(self.AS_offsets, str) and self.AS_offsets == "Table Not Found":
            self.GN_frames['OffsetI'] = [0] * len(self.GN_frames)
            self.GN_frames['OffsetJ'] = [0] * len(self.GN_frames)
            self.nofound = 'Yes'
        else:
            self.AS_offsets.rename(columns={'Label': 'Object Label'}, inplace=True)
            self.GN_frames = pd.merge(
                self.GN_frames,
                self.AS_offsets[['Object Label', 'Story', 'Offset I-end', 'Offset J-end']],
                on=['Object Label', 'Story'],
                how='left'
            )
            self.GN_frames['OffsetI'] = [0 if math.isnan(val) else val for val in self.GN_frames['Offset I-end']]
            self.GN_frames['OffsetJ'] = [0 if math.isnan(val) else val for val in self.GN_frames['Offset J-end']]
            self.nofound = 'No'
    
        # === Step 8: Process slab (shell) information ===
        self.GN_Slabs = self.OE_shells[self.OE_shells['Area Type'] == 'Floor']
        self.AS_shells = self.AS_shells.rename(columns={'Unique Name': 'Element Label'})
        self.GN_Slabs = pd.merge(self.GN_Slabs, self.AS_shells[['Element Label', 'Section']], on='Element Label', how='inner')
        self.GN_Slabs = pd.merge(self.GN_Slabs, self.SC_shells[['Section', 'Slab Thickness', 'Fc', 'Slab_Label']], on='Section', how='inner')
    
        # Store the processed section tables in the internal dictionary
        self.dict['TABLE:  "FRAME GENERATION"'] = self.GN_frames
        self.dict['TABLE:  "SLAB GENERATION"'] = self.GN_Slabs
        self.dict['TABLE:  "FRAME ASSIGNMENTS - OFFSETS"'] = self.AS_offsets
        
    def _consolidateInfoPiers(self):
        
        AS_pier = self.AS_pier.copy()
        SC_piers = self.SC_piers.copy()
        OE_walls = self.OE_walls.copy()
        pier_rebar = self.pier_rebar.copy()
        
        # Store in pier assignments the pier properties
        AS_pier = pd.merge(AS_pier, SC_piers[['Story','Pier','# Area Objects','Width Bottom']], on=['Story', 'Pier'], how='left')
        AS_pier.rename(columns={'Unique Name':'Element Label'}, inplace = True)
        
        # Include on wall objects the pier assignments
        OE_walls = pd.merge(OE_walls, AS_pier[['Element Label','Pier','# Area Objects','Width Bottom']], on='Element Label',how='inner')
        
        # Clean and filter the pier rebar
        pier_rebar.rename(columns={'Pier Label':'Pier'}, inplace=True) 
        pier_rebar =  pier_rebar[pier_rebar['Station'] == 'Bottom']
        pier_rebar = pier_rebar.fillna(0)
        pier_rebar['Cuantia_Val'] = pier_rebar['Required Reinf']/100
        
        OE_walls = pd.merge(OE_walls, pier_rebar[['Story','Pier','Cuantia_Val','Boundary Zone Left','Boundary Zone Right']],on=['Story', 'Pier'],how='left')
        
        OE_WallRebar = self.Wall_Rebar_Data_ETABS(OE_walls)
        df_dividido = self.dividir_ancho_num_macro_ETABS(OE_walls, OE_WallRebar)
        OE_walls = pd.merge(OE_walls, df_dividido, on=['Story','Area Label'], how='left')
        
        self.dict['TABLE:  "OBJECTS AND ELEMENTS - WALL"'] = OE_walls
        self.OE_walls = OE_walls
    
    def Wall_Rebar_Data_ETABS(self, df_OEWall):
        
        num_macro, ancho, concreto, acero, cuantia, espesor, story, pier = [], [], [], [], [], [], [], []
        for index in range(len(df_OEWall)):
            
            aa = df_OEWall.iloc[index]
            storyval = aa['Story']
            pierval = aa['Pier']
            # cuando no hay elemento de borde
            if aa['Boundary Zone Left'] == 0 and aa['Boundary Zone Right'] == 0:
                nummacro = int(aa['Width Bottom'])*2
                anchoval = [float(aa['Width Bottom']/nummacro)]*nummacro
                concval = [int(aa['UnConf_Label'])]*nummacro
                aceroval = [int(aa['Steel_Label'])]*nummacro
                cuantiaval = [float(aa['Cuantia_Val'])]*nummacro
                espeval = [float(aa['Thickness'])]*nummacro
                nummacro = [1]*int(aa['Width Bottom'])*2
                
            # cuando hay elemento de borde
            elif aa['Boundary Zone Left'] > 0 or aa['Boundary Zone Right'] > 0:
                nummacro = int(aa['Width Bottom'])*2
                anchoval = [0]*nummacro
                anchoval[0] = float(aa['Boundary Zone Left'])
                anchoval[-1] = float(aa['Boundary Zone Right'])
                concval = [0]*nummacro
                concval[0] = int(aa['Conf_Label'])
                concval[-1] = int(aa['Conf_Label'])
                for index2 in range(1,len(anchoval)-1):
                    anchoval[index2] = float((aa['Width Bottom']-(aa['Boundary Zone Left']+aa['Boundary Zone Right']))/(nummacro-2))    
                    concval[index2] = int(aa['UnConf_Label'])
                aceroval = int(aa['Steel_Label']*nummacro)
                cuantiaval = float([aa['Cuantia_Val']]*nummacro)
                espeval = [float(aa['Thickness'])]*nummacro
                nummacro = [1]*int(aa['Width Bottom'])*2
                
            num_macro.append(nummacro)
            ancho.append(anchoval)
            concreto.append(concval)
            acero.append(aceroval)
            cuantia.append(cuantiaval)
            espesor.append(espeval)
            story.append(storyval)
            pier.append(pierval)
            
            # Finalmente ensamblamos el diccionario
            data = {
                'Story': story,
                'Num_Macro': num_macro,
                'Pier': pier,
                'Ancho': ancho,
                'Concreto':concreto,
                'Acero':acero,
                'Cuantia':cuantia
            }
            df_WallRebar = pd.DataFrame(data)
                
        return df_WallRebar
    
    def dividir_ancho_num_macro_ETABS(self, df, df_ancho):
        
        resultado = []
        for story, group in df.groupby(['Story', 'Pier']):
            
            thickness = float(group['Thickness'].values[0])
            
            # Extraemos la lista Ancho y Num_Macro correspondiente al grupo actual (Story, Pier)
            ancho_lista = df_ancho[(df_ancho['Story'] == story[0]) & (df_ancho['Pier'] == story[1])]['Ancho'].values[0]
            num_macro_lista = df_ancho[(df_ancho['Story'] == story[0]) & (df_ancho['Pier'] == story[1])]['Num_Macro'].values[0]
            concreto_lista = df_ancho[(df_ancho['Story'] == story[0]) & (df_ancho['Pier'] == story[1])]['Concreto'].values[0]
            acero_lista = df_ancho[(df_ancho['Story'] == story[0]) & (df_ancho['Pier'] == story[1])]['Acero'].values[0]
            cuantia_lista = df_ancho[(df_ancho['Story'] == story[0]) & (df_ancho['Pier'] == story[1])]['Cuantia'].values[0]
            
            thickness_list = []
            for index in range(len(num_macro_lista)):
                thickness_list.append(thickness)
                
            # Creamos un índice para recorrer la lista Ancho y Num_Macro
            index = 0
            nuevo_ancho, nuevo_num_macro, nuevo_concreto, nuevo_acero, nuevo_cuantia, nuevo_thickness = [], [], [], [], [], []

            for _, row in group.iterrows():
                longitud_muro = row['Nodes Lenght']
                suma_ancho = 0
                muro_ancho,muro_num_macro,muro_concreto,muro_acero,muro_cuantia, muro_thickness= [], [], [], [], [], []

                # Extraemos los valores de Ancho y Num_Macro hasta que la suma iguale la longitud del muro
                while suma_ancho < longitud_muro and index < len(ancho_lista):
                    muro_ancho.append(ancho_lista[index])
                    muro_num_macro.append(num_macro_lista[index])
                    muro_concreto.append(concreto_lista[index])
                    muro_acero.append(acero_lista[index])
                    muro_cuantia.append(cuantia_lista[index])
                    muro_thickness.append(thickness_list[index])
                    suma_ancho += ancho_lista[index]
                    index += 1

                # Guardamos la lista de Ancho y Num_Macro para el muro actual
                nuevo_ancho.append(muro_ancho)
                nuevo_num_macro.append(muro_num_macro)
                nuevo_concreto.append(muro_concreto)
                nuevo_acero.append(muro_acero)
                nuevo_cuantia.append(muro_cuantia)
                nuevo_thickness.append(muro_thickness)

            # Añadimos los resultados de este Story y Pier a la lista
            for i, (_, row) in enumerate(group.iterrows()):
                resultado.append({
                    'Story': row['Story'],
                    'Pier': row['Pier'],
                    'Area Label': row['Area Label'],
                    'long': row['Nodes Lenght'],
                    'Ancho': nuevo_ancho[i],
                    'Num_Macro': nuevo_num_macro[i],
                    'Concreto':nuevo_concreto[i],
                    'Acero':nuevo_acero[i],
                    'Cuantia':nuevo_cuantia[i],
                    'Thickness_List':nuevo_thickness[i]
                })
                
        # Convertimos el resultado en un DataFrame
        return pd.DataFrame(resultado)
    
    # Creates elastic/non-linear frame elements
    def _genElements(self, model_type):
        """
        Creates elastic/non-linear frame elements in the OpenSeesPy model based on the information
        in self.genframes.
    
        This function:
        - Iterates through each frame (beam or column).
        - Defines appropriate geometric transformation ('PDelta' for columns, 'Linear' for beams).
        - Computes vector of orientation and joint offsets depending on the design type.
        - Creates the element using the elasticBeamColumn command in OpenSees.
        - Stores transformation vectors and offset vectors in the DataFrame for reference.
    
        Returns:
            None. Updates OpenSees model with frame elements, and augments self.genframes 
            with transformation and offset data.
        """
    
        vectores, dI_listdf, dJ_listdf = [], [], []
        nels = len(self.GN_frames)
    
        for ind2 in range(nels):
            aa = self.GN_frames.iloc[ind2]
    
            nodoI = aa['Joint I']
            nodoJ = aa['Joint J']
            eleNodes = [int(nodoI), int(nodoJ)]
    
            # === COLUMN ELEMENT ===
            if aa['Design Type'] == 'Column':
                theta = aa['Angle']
                theta_rad = math.radians(theta)
                # Local x-axis orientation vector (perpendicular to rotation angle)
                vectrans = [-math.sin(theta_rad), math.cos(theta_rad), 0]
                transfTag = int(aa['Element Label'])
    
                # Offset vectors for each node
                dI_list = [0, 0, aa['OffsetI']]
                dJ_list = [0, 0, -aa['OffsetJ']]
    
                # Define geometric transformation with or without joint offsets
                if self.nofound == 'Yes':
                    ops.geomTransf('PDelta', transfTag, *vectrans)
                else:
                    ops.geomTransf('PDelta', transfTag, *vectrans, '-jntOffset', *dI_list, *dJ_list)
    
            # === BEAM ELEMENT ===
            elif aa['Design Type'] == 'Beam':
                A = np.array(aa['CoordI'])
                B = np.array(aa['CoordJ'])
                AB = B - A
    
                # Local axis orientation vector (perpendicular to global Z)
                zg = [0, 0, 1]
                cP = np.cross(AB, zg)
                vectrans = cP / np.linalg.norm(cP)
                transfTag = int(aa['Element Label'])
    
                # Unit vector for offset direction along beam axis
                vect_offset = ut.offset_direction(A, B)
    
                # Offset vectors along beam axis
                dI_list = np.array([aa['OffsetI']] * 3) * np.array(vect_offset)
                dJ_list = np.array([-aa['OffsetJ']] * 3) * np.array(vect_offset)
    
                # Define geometric transformation (Linear, without offsets)
                ops.geomTransf('Linear', transfTag, *vectrans)
    
            # Store vectors for reference
            vectores.append(vectrans)
            dI_listdf.append(dI_list)
            dJ_listdf.append(dJ_list)
    
            # Create elastic beam-column element in OpenSees if model_type = EM
            if model_type == 'EM':
                ops.element('elasticBeamColumn', int(aa['Element Label']), *eleNodes, int(aa['sectag']), transfTag)
            else:
                ops.element('forceBeamColumn', int(aa['Element Label']), *eleNodes, transfTag, int(aa['sectag']))
            
        # Save transformation and offset vectors to DataFrame for post-processing
        self.GN_frames['Vect_TrG'] = vectores
        self.GN_frames['dI_list'] = dI_listdf
        self.GN_frames['dJ_list'] = dJ_listdf
    
    # Generates shell elements (slabs) 
    def _genSlabs(self):
        """
        Generates shell elements (slabs) in the OpenSeesPy model using the information in self.genslabs.
    
        For each slab:
        - Determines whether the slab is triangular or quadrilateral based on missing joint values (NaNs).
        - Uses 'ShellDKGT' for triangular slabs and 'ShellDKGQ' for quadrilateral slabs.
        - Assigns a negative element tag (OpenSees convention for shell elements).
        - Applies the previously defined 'ElasticMembranePlateSection' using the slab's section label.
        - Stores the assigned element tags in the DataFrame.
    
        Returns:
            None. Updates OpenSees with shell elements and adds 'Slab_Tags' column to self.genslabs.
        """
    
        slabtags = []
    
        for index in range(len(self.GN_Slabs)):
            aa = self.GN_Slabs.iloc[index]
            seclosa = int(aa['Slab_Label'])         # Section tag for the slab
            tag = int(-aa['Element Label'])         # Negative tag to distinguish shell elements
    
            # === TRIANGULAR SLAB ===
            if (
                math.isnan(aa['Joint 1']) or math.isnan(aa['Joint 2']) or
                math.isnan(aa['Joint 3']) or math.isnan(aa['Joint 4'])
            ):
                # Identify which joint is missing and build triangle with the remaining 3
                if math.isnan(aa['Joint 1']):
                    nodoslosa = [int(aa['Joint 2']), int(aa['Joint 3']), int(aa['Joint 4'])]
                elif math.isnan(aa['Joint 2']):
                    nodoslosa = [int(aa['Joint 1']), int(aa['Joint 3']), int(aa['Joint 4'])]
                elif math.isnan(aa['Joint 3']):
                    nodoslosa = [int(aa['Joint 1']), int(aa['Joint 2']), int(aa['Joint 4'])]
                else:  # Joint 4 is NaN
                    nodoslosa = [int(aa['Joint 1']), int(aa['Joint 2']), int(aa['Joint 3'])]
    
                # Define triangular shell element
                ops.element('ShellDKGT', tag, *nodoslosa, seclosa)
    
            # === QUADRILATERAL SLAB ===
            else:
                nodoslosa = [
                    int(aa['Joint 1']), int(aa['Joint 2']),
                    int(aa['Joint 3']), int(aa['Joint 4'])
                ]
    
                # Define quadrilateral shell element
                ops.element('ShellDKGQ', tag, *nodoslosa, seclosa)
    
            # Save the tag used for this slab element
            slabtags.append(tag)
    
        # Store the shell element tags in the genslabs DataFrame
        self.GN_Slabs['Slab_Tags'] = slabtags
        
    def _genWalls(self):
        
        element_labels = []
        for i, row in self.OE_walls.iterrows():
            elelabel = int(row['Element Label'])
            node_ele = row['Nodes Orientation']
            t = row['Thickness_List']
            width = row['Ancho']
            rho2 = row['Cuantia']
            concrete2 = row['Concreto']
            steel2 = row['Acero']
            tagshear2 = int(row['Shear_Label'])
            num_fib = int(np.sum(row['Num_Macro']))
            
            ops.element('MVLEM_3D',int(elelabel),*node_ele,num_fib,'-thick',*t,'-width',*width,'-rho',*rho2,'-matConcrete',*concrete2,'-matSteel',*steel2,'-matShear',tagshear2)
            element_labels.append(int(elelabel))
        
        self.OE_walls['Element_Label'] = element_labels
        self.GN_frames = self.OE_walls
        
        self.dict['TABLE:  "WALL GENERATION"'] = self.GN_frames
        
    def _calculate_beamloads(self, load_case):
        
        self.slabloads = self.data_file['TABLE:  "SHELL LOADS - UNIFORM"']
        slabloads = self.slabloads
        
        oeshells = self.GN_Slabs
        secslabs = self.SC_shells
        joints = self.OE_joints
        
        def extraer_coeficientes(formula):
            match = re.search(r'\(.*?\)\s*([\d.]+)\s*CM\s*\+\s*([\d.]+)\s*CV', formula)
            if match:
                coef_cm = float(match.group(1))
                coef_cv = float(match.group(2))
                return coef_cm, coef_cv
            else:                
                print(f"Error: 'NO se pudo interpretar el caso de carga: '{formula}'.")
                sys.exit()
                
        F_cm, F_cv = extraer_coeficientes(load_case)
        
        # Agregar coordenadas X y Y de los joints a la tabla objects and elements - shells
        for i in range(1, 5):
            joint_col = f'Joint {i}'
            joints_subset = joints[['Element Label', 'Global X', 'Global Y']].rename(
                columns={'Element Label': joint_col, 'Global X': f'XJ{i}', 'Global Y': f'YJ{i}'}
            )
            oeshells = pd.merge(oeshells, joints_subset, on=joint_col, how='left')


        oeshells['LJ1-J2'] = [np.sqrt( (row['XJ1']-row['XJ2'])**2 + (row['YJ1']-row['YJ2'])**2 ) for i, row in oeshells.iterrows()]


        # Calcular longitudes entre nodos consecutivos
        for i in range(1, 5):
            j = 1 if i == 4 else i + 1  # para cerrar el ciclo J4-J1
            oeshells[f'LJ{i}-J{j}'] = [np.sqrt(
                    (row[f'XJ{i}'] - row[f'XJ{j}'])**2 +
                    (row[f'YJ{i}'] - row[f'YJ{j}'])**2
                ) for m, row in oeshells.iterrows()]

        oeshells['perimetro_losa'] = oeshells['LJ1-J2']+oeshells['LJ2-J3']+oeshells['LJ3-J4']+oeshells['LJ4-J1']

        # Calculo del area de la losa
        def calcular_area_losa(row):
            coords = []

            # Añadir coordenadas si existen
            if not pd.isna(row['XJ1']):
                coords.append((row['XJ1'], row['YJ1']))
            if not pd.isna(row['XJ2']):
                coords.append((row['XJ2'], row['YJ2']))
            if not pd.isna(row['XJ3']):
                coords.append((row['XJ3'], row['YJ3']))
            if not pd.isna(row['XJ4']):
                coords.append((row['XJ4'], row['YJ4']))

            if len(coords) < 3:
                return np.nan  # No es una losa válida

            # Fórmula del área de un polígono (shoelace)
            x = [p[0] for p in coords]
            y = [p[1] for p in coords]
            x.append(x[0])  # cerrar el polígono
            y.append(y[0])
            area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(len(coords))))
            return area

        oeshells['area_losa'] = oeshells.apply(calcular_area_losa, axis=1)

        # Esta modelado desde ETABS como losa en una direccion?
        oeshells = pd.merge(oeshells, secslabs[['Section','One-Way Load Distribution?']], on='Section', how='inner')

        def determinar_direccion_losa(row):
            # Tomar solo los lados que existen (evita problemas con losas triangulares o datos faltantes)
            lados = [
                row.get('LJ1-J2', np.nan),
                row.get('LJ2-J3', np.nan),
                row.get('LJ3-J4', np.nan),
                row.get('LJ4-J1', np.nan)
            ]
            lados_validos = [l for l in lados if not pd.isna(l)]

            if len(lados_validos) < 2:
                return np.nan  # No se puede determinar dirección

            lado_max = max(lados_validos)
            lado_min = min(lados_validos)

            relacion = lado_max / lado_min if lado_min != 0 else np.nan

            if pd.isna(relacion):
                return np.nan
            elif relacion >= 2:
                return '1d'
            else:
                return '2d'

        oeshells['direccion_losa'] = oeshells.apply(determinar_direccion_losa, axis=1)

        oeshells['shell-direction'] = ['1direction' if row['One-Way Load Distribution?'] == 'Yes' or row['direccion_losa'] == '1d'
                                       else '2direction' for index,row in oeshells.iterrows()]

        # Calcular areas tributarias
        for i in range(1, 5):
            j = 1 if i == 4 else i + 1
            col_name = f'AT-J{i}-J{j}'
            
            at_list = []
            for _, row in oeshells.iterrows():
                if row['shell-direction'] == '2direction':
                    # Método habitual por proporción de perímetro
                    at = (row['area_losa'] / row['perimetro_losa']) * row[f'LJ{i}-J{j}']
                else:
                    # 1 dirección: solo repartir entre los dos lados más largos
                    longitudes = {
                        'AT-J1-J2': row.get('LJ1-J2', 0),
                        'AT-J2-J3': row.get('LJ2-J3', 0),
                        'AT-J3-J4': row.get('LJ3-J4', 0),
                        'AT-J4-J1': row.get('LJ4-J1', 0),
                    }
                    # Obtener los dos lados más largos
                    lados_mayores = sorted(longitudes.items(), key=lambda x: x[1], reverse=True)[:2]
                    if col_name in dict(lados_mayores):
                        at = row['area_losa'] * 0.5
                    else:
                        at = 0.0
                at_list.append(at)
            
            oeshells[col_name] = at_list

        joints = self.OE_joints
        
        self.oeframeloadB = self.data_file['TABLE:  "OBJECTS AND ELEMENTS - FRAMES"']
        oeframeload = self.oeframeloadB
        
        oeframeloadC = self.GN_frames
        
        self.frameloads = self.data_file['TABLE:  "FRAME LOADS - DISTRIBUTED"']
        frameloads = self.frameloads
        
        asgframes = self.AS_frames
        framessection = self.SC_frames

        # filtrar objects and elements - frames
        oeframeload['Design Type'] = ['Beam' if 'B' in obj else 'Column' for obj in oeframeload['Object Label'].tolist()]
        oeframeload = oeframeload[oeframeload['Design Type'] == 'Beam']

        oeframeloadC['Design Type'] = ['Beam' if 'B' in obj else 'Column' for obj in oeframeloadC['Object Label'].tolist()]
        oeframeloadC = oeframeloadC[oeframeloadC['Design Type'] == 'Column']

        joints_copy = joints.copy()

        joints_copy.rename(columns = {'Element Label':'Joint I'}, inplace = True)
        oeframeload = pd.merge(oeframeload, joints_copy[['Joint I', 'Global X', 'Global Y']], on = 'Joint I', how = 'inner')
        oeframeload.rename(columns = {'Global X':'Xi', 'Global Y':'Yi'}, inplace = True)

        joints_copy.rename(columns = {'Joint I':'Joint J'}, inplace = True)
        oeframeload = pd.merge(oeframeload, joints_copy[['Joint J', 'Global X', 'Global Y']], on = 'Joint J', how = 'inner')
        oeframeload.rename(columns = {'Global X':'Xj', 'Global Y':'Yj'}, inplace = True)

        oeframeload['lenght'] = [np.sqrt( (row['Xj']-row['Xi'])**2 + (row['Yj']-row['Yi'])**2 ) for i, row in oeframeload.iterrows()]

        asgframes = pd.merge(asgframes, framessection[['Section','Area']], on = 'Section', how = 'inner')
        oeframeload = pd.merge(oeframeload, asgframes[['Object Label','Story','Area']], on = ['Object Label','Story'], how = 'inner')

        # filtrar objects and elements - shells
        oeshells = oeshells[oeshells['Area Type'] == 'Floor']

        # filtrar shell loads
        slabloads = slabloads[slabloads['Direction'] == 'Gravity']

        # filtrar frame loads
        frameloads.rename(columns = {'Unique Name':'Element Label'}, inplace = True)
        frameloads = frameloads[frameloads['Direction'] == 'Gravity']

        def slabjoints_combinations(row):
            
            joints = [row['Joint 1'], row['Joint 2'], row['Joint 3'], row['Joint 4']]
            valid_joints = [int(j) for j in joints if pd.notna(j)]
            
            result = {
                'Story': row['Story'],
                'Floor Label': row['Area Label'],
                'eL': row['Slab Thickness'],
                'at-J1_J2': row['AT-J1-J2'],
                'at-J2_J3': row['AT-J2-J3'],
                'at-J3_J4': row['AT-J3-J4'],
                'at-J4_J1': row['AT-J4-J1'],
                'at-J2_J1': row['AT-J1-J2'],
                'at-J3_J2': row['AT-J2-J3'],
                'at-J4_J3': row['AT-J3-J4'],
                'at-J1_J4': row['AT-J4-J1']
            }

            combinaciones = []
            if len(valid_joints) == 4:
                j1, j2, j3, j4 = valid_joints
                combinaciones = [
                    ('J1_J4', f'{j1} - {j4}'),
                    ('J4_J1', f'{j4} - {j1}'),
                    ('J1_J2', f'{j1} - {j2}'),
                    ('J2_J1', f'{j2} - {j1}'),
                    ('J2_J3', f'{j2} - {j3}'),
                    ('J3_J2', f'{j3} - {j2}'),
                    ('J3_J4', f'{j3} - {j4}'),
                    ('J4_J3', f'{j4} - {j3}')
                ]
            elif len(valid_joints) == 3:
                j1, j2, j3 = valid_joints
                combinaciones = [
                    ('J1_J2', f'{j1} - {j2}'),
                    ('J2_J1', f'{j2} - {j1}'),
                    ('J2_J3', f'{j2} - {j3}'),
                    ('J3_J2', f'{j3} - {j2}'),
                    ('J3_J1', f'{j3} - {j1}'),
                    ('J1_J3', f'{j1} - {j3}')
                ]
            else:
                raise ValueError(f"Losa con cantidad inesperada de nodos: {len(valid_joints)}")
            
            for nombre, combinacion in combinaciones:
                result[nombre] = combinacion
            
            return result

        def beamjoints_combinations(row):
            j1, j2 = map(int, [row['Joint I'], row['Joint J']])
            
            return [
                row['Story'],
                row['Object Label'],
                row['Element Label'],
                row['lenght'],
                row['Area'],
                f'{j1} - {j2}',
                f'{j2} - {j1}'
            ]

        oeshells = oeshells.apply(slabjoints_combinations, axis = 1, result_type = 'expand')

        oeframeload = oeframeload.apply(beamjoints_combinations, axis = 1, result_type = 'expand')
        oeframeload.columns = [
            'Story','Beam Label','Element Label','Lenght','Atv','Ji_Jj', 'Jj_Ji'
        ]

        duplicados = oeframeload[oeframeload.duplicated(subset=['Beam Label', 'Story'], keep=False)]

        # Agrupar duplicados y combinarlos
        filas_combinadas = []
        final_lengths = {}
        for (beam, story), group in duplicados.groupby(['Beam Label', 'Story']):
            group = group.sort_values('Element Label')  # ordena por Element Label
            
            # Extraer valores combinados
            element_label = int(group['Element Label'].iloc[0].split('-')[0])
            lenght_total = group['Lenght'].sum()
            
            # Crear combinaciones de juntas
            ji_start = group['Ji_Jj'].iloc[0].split(' - ')[0]
            ji_end = group['Ji_Jj'].iloc[-1].split(' - ')[1]
            ji_jj_comb = f'{ji_start} - {ji_end}'
            jj_ji_comb = f'{ji_end} - {ji_start}'
            
            # Guardar longitud combinada para cada fila original
            idxs = group.index.tolist()
            for idx in idxs:
                final_lengths[idx] = lenght_total

            filas_combinadas.append({
                'Story': story,
                'Beam Label': beam,
                'Element Label': element_label,
                'Lenght': lenght_total,
                'Final Lenght': lenght_total,
                'Ji_Jj': ji_jj_comb,
                'Jj_Ji': jj_ji_comb
            })

        # Asignar 'final lenght' a duplicados
        mapped = oeframeload.index.map(final_lengths)
        oeframeload['Final Lenght'] = mapped.where(mapped.notna(), oeframeload['Lenght'])

        df_combinados = pd.DataFrame(filas_combinadas)
        oeframeload = pd.concat([oeframeload, df_combinados], ignore_index=True)

        # Listas para guardar resultados
        floor_matches = []
        floor_thickness = []
        floor_at_values = []  # <-- nueva lista para el valor de AT
        combinaciones2 = []
        for _, row2 in oeframeload.iterrows():
            
            combinacion1 = row2['Ji_Jj']
            combinacion2 = row2['Jj_Ji']

            coincidencias = []
            thickness = []
            at_values = []

            for _, row1 in oeshells.iterrows():
                combinaciones = {
                    'J1_J2': row1.get('J1_J2'), 'J2_J1': row1.get('J2_J1'),
                    'J2_J3': row1.get('J2_J3'), 'J3_J2': row1.get('J3_J2'),
                    'J3_J4': row1.get('J3_J4'), 'J4_J3': row1.get('J4_J3'),
                    'J4_J1': row1.get('J4_J1'), 'J1_J4': row1.get('J1_J4')
                }
                
                combinaciones2.append(combinaciones)
                
                if combinacion1 in combinaciones.values():
                    
                    coincidencias.append(row1['Floor Label'])
                    thickness.append(row1['eL'])
                    
                    df_combi = pd.DataFrame(combinaciones.values()).reset_index()
                    index = np.where(df_combi==combinacion1)[0][0]
                    
                    df_combi2 = pd.DataFrame.from_dict(combinaciones, orient='index').reset_index()
                    df_combi2.rename(columns={'index': 'key'}, inplace=True)
                    at_values.append(row1[f'at-{df_combi2["key"][index]}'])  # <-- AT correspondiente
                    
                elif combinacion2 in combinaciones.values():
                    coincidencias.append(row1['Floor Label'])
                    thickness.append(row1['eL'])
                    
                    df_combi = pd.DataFrame(combinaciones.values()).reset_index()
                    index = np.where(df_combi==combinacion2)[0][0]
                    
                    df_combi2 = pd.DataFrame.from_dict(combinaciones, orient='index').reset_index()
                    df_combi2.rename(columns={'index': 'key'}, inplace=True)
                    at_values.append(row1[f'at-{df_combi2["key"][index]}'])  # <-- AT correspondiente

            floor_matches.append(coincidencias)
            floor_thickness.append(thickness)
            floor_at_values.append(at_values)  # <-- guardar lista de AT

        oeframeload['Floor Labels Matched'] = floor_matches
        oeframeload['Floor Thickness Matched'] = floor_thickness
        oeframeload['Floor tributary Matched'] = floor_at_values

        Wcl_resultados = []
        Wcv_resultados = []
        for idx, row in oeframeload.iterrows():
            tributary_area = row['Floor tributary Matched']
            lenght = row['Final Lenght']
            floor_labels = row['Floor Labels Matched']
            espesores = row['Floor Thickness Matched']
            story = row['Story']
            
            Wcv_resultados.append(F_cm*24*row['Atv'])
            
            wcl_total = 0.0

            for i, label in enumerate(floor_labels):
                espesor = espesores[i]
                At_L = tributary_area[i]/lenght

                # Filtrar cargas por label y tipo
                cargas_label = slabloads[(slabloads['Label'] == label) & (slabloads['Story'] == story)]
                                
                carga_muerta = cargas_label[cargas_label['Load Pattern'] == 'SobreImpuesta']['Load'].sum()
                carga_viva = cargas_label[cargas_label['Load Pattern'] == 'CargaViva']['Load'].sum()

                # Si no hay valor, se usa 0.0 (ya lo maneja .sum())
                contribucion = At_L * (F_cm * (carga_muerta + 24 * espesor) + F_cv * carga_viva)
                wcl_total += contribucion

            Wcl_resultados.append(wcl_total)

        # Agregar columna al DataFrame
        oeframeload['Wcl'] = Wcl_resultados
        oeframeload['Wcv'] = Wcv_resultados

        # Filas que son combinadas (label como entero)
        oeframeload['EsCombinado'] = oeframeload['Element Label'].apply(lambda x: isinstance(x, (int, float)) and float(x).is_integer())
        # Agrupar por Beam Label y Story para sumar Wcl
        wcl_totales = oeframeload.groupby(['Beam Label', 'Story'])['Wcl'].sum()
        # DataFrame solo con las filas combinadas
        oeframeload = oeframeload[oeframeload['EsCombinado']].copy()
        # Actualizar Wcl con suma total de duplicados
        oeframeload['Wcl'] = oeframeload.set_index(['Beam Label', 'Story']).index.map(wcl_totales)
        # Limpiar columna auxiliar
        oeframeload.drop(columns='EsCombinado', inplace=True)


        oeframeload = pd.merge(oeframeload, frameloads[['Element Label', 'Force at End']], on = 'Element Label', how = 'left')
        oeframeload['Wv'] = [0 if math.isnan(val) else val for val in oeframeload['Force at End'].tolist()]

        oeframeload['Distributed force'] = oeframeload['Wcl']+oeframeload['Wcv']+oeframeload['Wv']
        
        self.oeframeloadB = oeframeload
        
        
        jointsC = self.OE_joints

        jointsC.rename(columns={'Element Label':'Joint I'}, inplace=True)
        oeframeloadC = pd.merge(oeframeloadC, jointsC[['Joint I','Global Z']], on='Joint I', how='inner').rename(columns={'Global Z': 'Z_JointI'})

        jointsC.rename(columns={'Joint I':'Joint J'}, inplace=True)
        oeframeloadC = pd.merge(oeframeloadC, jointsC[['Joint J','Global Z']], on='Joint J', how='inner').rename(columns={'Global Z': 'Z_JointJ'})

        # Asegurarnos que el Joint J tiene mayor z que el Joint I
        oeframeloadC['Max_Joint'] = oeframeloadC.apply(lambda row: row['Joint I'] if row['Z_JointI'] >= row['Z_JointJ'] else row['Joint J'], axis=1)
        
        # Calcular la longitud de la columna
        oeframeloadC['Length'] = np.abs(oeframeloadC['Z_JointJ']-oeframeloadC['Z_JointI'])

        # Calcular carga sobre el nodo
        oeframeloadC['Force Z'] =  (-1) * (24) * (oeframeloadC['Length'] * oeframeloadC['t3'] * oeframeloadC['t2'])
        
        max_z = np.max(jointsC['Global Z']) # Maxima altura
        oeframeloadC.loc[(oeframeloadC['Z_JointI'] == max_z) | (oeframeloadC['Z_JointJ'] == max_z), 'Force Z'] = 0 
        
        self.oeframeloadC = oeframeloadC
        
        # Store the processed section tables in the internal dictionary
        self.dict['TABLE:  "DISTRIBUTED LOAD - BEAMS"'] = self.oeframeloadB
        self.dict['TABLE:  "PUNTUAL LOAD - COLUMNS"'] = self.oeframeloadC

    # Applies gravitational loads
    def _appLoads(self):
        """
        Applies gravitational loads to the OpenSeesPy model.
    
        This function:
        - Defines a linear time series and a plain load pattern (ID = 1).
        - Applies uniformly distributed loads to beam elements using `eleLoad`.
        - Applies point loads at nodes (typically column tops) using `load`.
    
        Returns:
            None. Updates the OpenSees model with both element and nodal loads.
        """
    
        # Define linear time series and associate it with a plain load pattern
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
    
        # === Apply uniformly distributed loads to frame (beam) elements ===
        for ind3 in range(len(self.oeframeloadB)):
            aa = self.oeframeloadB.iloc[ind3]
            # Apply load in global Z direction (negative for gravity)
            ops.eleLoad(
                '-ele', int(aa['Element Label']),
                '-type', '-beamUniform',
                -aa['Distributed force'], 0.0
            )
    
        # === Apply point loads at nodes (typically columns) ===
        nlabel = self.oeframeloadC['Max_Joint'].to_numpy()
        zload = self.oeframeloadC['Force Z'].to_numpy()
    
        for i in range(len(nlabel)):
            # Apply load only in global Z direction at the node
            ops.load(int(nlabel[i]), 0.0, 0.0, float(zload[i]), 0.0, 0.0, 0.0)
            
    # Creates mass center nodes and rigid diaphragms for each floor in the model
    def _massDiaph(self):
        """
        Creates mass center nodes and rigid diaphragms for each floor in the model.
    
        This function:
        - Extracts mass and geometry data from the diaphragm summary table.
        - Identifies the Z-levels (floor elevations) to define mass planes.
        - Creates a node at the mass center of each floor.
        - Fixes the vertical and rotational DOFs of each mass center node.
        - Assigns mass and rotational inertia to these nodes.
        - Generates rigid diaphragms linking all floor nodes to the mass center.
    
        Returns:
            None. Updates self.center_of_mass_nodes and self.altur.
        """
    
        # Load diaphragm mass summary and joint information
        self.OE_joints_COPY = self.OE_joints.copy()
        self.OE_joints_COPY.rename(columns={'Joint J': 'Element Label'}, inplace=True)
    
        # Get floor elevation values (Z coordinates)
        alturas = np.sort(pd.unique(self.OE_joints_COPY['Global Z']))
        altur = alturas[1::]  # Skip base level (assumed at index 0)
    
        # Extract diaphragm data (reversed to match top-down assignment)
        Xcent = np.flipud(self.masses['X Mass Center'])           # X coordinates of mass centers
        Ycent = np.flipud(self.masses['Y Mass Center'])           # Y coordinates of mass centers
        Masa = np.flipud(self.masses['Mass X']) / 1000            # Convert mass to metric tons
        Inercia = np.flipud(self.masses['Mass Moment of Inertia'])
    
        self.center_of_mass_nodes = []
    
        for ind, alt in enumerate(altur):
            # Get all joint tags at this floor level
            index1 = self.OE_joints_COPY['Global Z'] == alt
            dia1 = self.OE_joints_COPY[index1]['Element Label'].astype(int).to_list()
    
            node_id = 100000000 * ind  # Unique node tag for mass center
    
            # Create mass center node
            ops.node(node_id, float(Xcent[ind]), float(Ycent[ind]), float(alt))
    
            # Fix the node in Z and all rotational DOFs except torsion
            ops.fix(node_id, 0, 0, 1, 1, 1, 0)
    
            # Assign mass and rotational inertia (about vertical axis)
            ops.mass(node_id, float(Masa[ind]), float(Masa[ind]), 0.0, 0.0, 0.0, float(Inercia[ind]))
    
            # Create rigid diaphragm at this level linking joints to mass center
            ops.rigidDiaphragm(3, node_id, *dia1)
    
            self.center_of_mass_nodes.append(node_id)
    
        self.altur = altur  # Store elevation levels for later use
        
        # Store the processed section tables in the internal dictionary
        self.dict['TABLE:  "MASS SUMMARY BY DIAPHRAGM"'] = self.masses
        
    # Performs modal analysis
    @staticmethod
    def modal_analysis(elastic_model):
        """
        Performs modal analysis of the structure using OpenSeesPy.
    
        This function:
        - Calculates eigenvalues for a number of modes equal to 3 times the number of floors.
        - Calls OpenSees’ `modalProperties()` to extract results and optionally write them to file.
    
        Returns:
            None. Stores results in self.eigenvalues and self.modal_results.
        """
    
        # Number of modes = 3 per floor (Ux, Uy, Rotation)
        Nmodes = len(elastic_model.altur) * 3
    
        # Perform eigenvalue analysis
        ops.eigen('fullGenLapack', Nmodes)
    
        # Save modal properties to file
        modal_results = ops.modalProperties('-return', '-unorm')
        
        return modal_results
    
    @staticmethod
    def modal_results_table(modal_results):
        
        def sums(key:str):
            sumux = 0
            sumux_list = []
            for modal in list(modal_results.get(key)):
                sumux += modal 
                sumux_list.append(sumux)    
            
            return sumux_list
        
        dict_results = {
            'Mode': range(1, len(modal_results.get('eigenPeriod'))+1),
            'Period [sec]': modal_results.get('eigenPeriod'),
            'UX [%]': modal_results.get('partiMassRatiosMX'),
            'UY [%]': modal_results.get('partiMassRatiosMY'),
            'UZ [%]': modal_results.get('partiMassRatiosMZ'),
            'Sum UX [%]': sums('partiMassRatiosMX'),
            'Sum UY [%]': sums('partiMassRatiosMY'),
            'Sum UZ [%]': sums('partiMassRatiosMZ'),
            'RX [%]': modal_results.get('partiMassRatiosRMX'),
            'RY [%]': modal_results.get('partiMassRatiosRMY'),
            'RZ [%]': modal_results.get('partiMassRatiosRMZ'),
            'Sum RUX [%]': sums('partiMassRatiosRMX'),
            'Sum RUY [%]': sums('partiMassRatiosRMY'),
            'Sum RUZ [%]': sums('partiMassRatiosRMZ'),
            }
            
        return pd.DataFrame(dict_results)
    
    @staticmethod
    def column_forces_table(genframes, load_case):
        
        an.gravedad()
        ops.loadConst('-time', 0.0)
        
        # Story 1 Column Data --->
        Base_Data = genframes[(genframes['Design Type'] == 'Column')]
        
        # Columns label--->
        Label_List = Base_Data['Element Label'].astype(int).tolist()
        Object_label = Base_Data['Object Label'].tolist()
        
        # Stories --->
        Stories_List = Base_Data['Story'].tolist()
        
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
        return pd.DataFrame({
            'Story': Stories_List,
            'Object Label': Object_label,
            'Element Label':Label_List,
            'Load Case': [load_case]*len(Stories_List),
            'P [kN]':Axial_Force,
            'V2 [kN]':VBasal_x,
            'V3 [kN]':VBasal_y,
            'T [kN-m]':Torcion,
            'M2 [kN-m]':Momento_x,
            'M3 [kN-m]':Momento_y
        })
    
    @staticmethod
    def frame_responses_plot():
        
        fig = opsvis.plot_model(show_nodal_loads=True, show_ele_loads=True)
        return fig      
    
    @staticmethod
    def nodal_responses_plot():
        
        ODB = opst.post.CreateODB(odb_tag=5)
        for i in range(10):
            ops.analyze(1)
            ODB.fetch_response_step()
        ODB.save_response()
        
        fig = opsvis.plot_nodal_responses(odb_tag=5, slides=False, step="absMax", resp_type="disp", resp_dof=["UX", "UY", "UZ"])
        
        opsvis.set_plot_props(point_size=2.0)
        opsvis.set_plot_colors(frame="black", cmap="turbo")
        
        fig2 = opsvis.plot_frame_responses(
            odb_tag=5,
            slides=False,
            step="absMax",
            resp_type="sectionForces",
            resp_dof="My",
            scale=3.0,
            show_values=True,
            line_width=5,
        )
                
        return fig, fig2      

    @staticmethod
    def plot_model():
        
        nodes = ops.getNodeTags()
        coords = np.array([ops.nodeCoord(n) for n in nodes])
        x_min, x_max = coords[:,0].min(), coords[:,0].max()
        y_min, y_max = coords[:,1].min(), coords[:,1].max()
        z_min, z_max = coords[:,2].min(), coords[:,2].max()
        
        margin = 2
        
        
        fig = opsvis.plot_model(odb_tag=0, show_outline=True)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(
                    range=[x_min - margin, x_max + margin], 
                    color='black',
                    gridcolor='gray',
                    zerolinecolor='black',
                    title='Eje X'
                ),
                yaxis=dict(
                    range=[y_min - margin, y_max + margin], 
                    color='black',
                    gridcolor='gray',
                    zerolinecolor='black',
                    title='Eje Y'
                ),
                zaxis=dict(
                    range=[z_min-0.5, z_max + margin], 
                    color='black',
                    gridcolor='gray',
                    zerolinecolor='black',
                    title='Eje Z'
                )
            )
        )
        
        for trace in fig.data:
            if trace.name == 'Nodes':
                trace.marker.color = 'black'  # color de los nodos
                trace.marker.size = 2
            if trace.name == 'mp constraint':
                trace.line.color = 'rgba(0,0,0,0)'  # color del diafragma 

        
        return fig
            
#%% ==> GENERADOR DE MODELO ELASTICO

class ElasticModelBuilder(ModelBuilder):
        
    def genInitialElasticModel(self, main_path, model_type = 'EM', load_case = '(0) 1CM + 0.25CV', npint = 5, shell_craking = 1):
        self.outputs_model = os.path.abspath(os.path.join(main_path, 'outputs', 'opensees_models'))
        
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        tqdm.write(" 🔄 Generando el arquetipo del modelo elástico ...") 
        self._debbugingJoints()                     # Cleans and filter the joint data
        self._genNodes()                            # Generate nodes
        self._debugging_restr()                     # Cleans and filter restraint data
        self._fixNodes()                            # Applies boundary conditions
        self._debugging_materials()                 # Links section definitions to their corresponding material properties 
        self._generateElasticSlabs(shell_craking)   # Assigns elastic membrane plate sections
        self._generateElasticFrames(npint)          # Generates elastic section definitions for frame elements
        self._consolidateInfo()                     # Consolidates and prepares frame and shell element information for model generation
        self._genElements(model_type)               # Creates elastic/non-linear frame elements
        self._genSlabs()                            # Generates shell elements (slabs) 
        self._calculate_beamloads(load_case)        # Calculate beam distributed loads
        self._appLoads()                            # Applies gravitational loads
        self._massDiaph()                           # Creates mass center nodes and rigid diaphragms for each floor in the model
        tqdm.write(" ✅​ Arquetipo generado ")
        
        self.dict['shell_craking'] = shell_craking
        self.dict['load_case'] = load_case
        self.dict['npint'] = npint
        
        return self.dict
        
    def genElasticModel(self, data_dict):
        
        self.OE_joints = data_dict['TABLE:  "OBJECTS AND ELEMENTS - JOINTS"']
        self.AS_restraints = data_dict['TABLE:  "JOINT ASSIGNMENTS - RESTRAINTS"']
        self.SC_frames = data_dict['TABLE:  "FRAME SECTIONS"']
        self.SC_shells = data_dict['TABLE:  "SHELL SECTIONS - SLAB"'] 
        self.GN_frames = data_dict['TABLE:  "FRAME GENERATION"']
        self.GN_Slabs = data_dict['TABLE:  "SLAB GENERATION"']
        self.oeframeloadB =  data_dict['TABLE:  "DISTRIBUTED LOAD - BEAMS"'] 
        self.oeframeloadC = data_dict['TABLE:  "PUNTUAL LOAD - COLUMNS"']
        self.masses = data_dict['TABLE:  "MASS SUMMARY BY DIAPHRAGM"']
        self.AS_offsets = data_dict['TABLE:  "FRAME ASSIGNMENTS - OFFSETS"']
        
        shell_craking = data_dict['shell_craking']
        npint = data_dict['npint']
        
        if isinstance(self.AS_offsets, str) and self.AS_offsets == "Table Not Found":
            self.nofound = 'Yes'
        else:
            self.nofound = 'No'
        
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        self._genNodes()                            # Generate nodes
        self._fixNodes()                            # Applies boundary conditions
        self._generateElasticSlabs(shell_craking)   # Assigns elastic membrane plate sections
        self._generateElasticFrames(npint)          # Generates elastic section definitions for frame elements
        self._genElements('EM')                     # Creates elastic/non-linear frame elements
        self._genSlabs()                            # Generates shell elements (slabs) 
        self._appLoads()                            # Applies gravitational loads
        self._massDiaph()                           # Creates mass center nodes and rigid diaphragms for each floor in the model
        
#%% ==> GENERADOR DE MODELO INELASTICO

class NONLinearModelBuilder(ModelBuilder):
        
    def genInitialNONLinearModel(self, main_path, model_type = 'NLM', load_case = '(0) 1CM + 0.25CV', npint = 5, shell_craking = 1, ed_type = 'DMO'):
        self.outputs_model = os.path.abspath(os.path.join(main_path, 'outputs', 'opensees_models'))
        
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        tqdm.write(" 🔄 Generando el arquetipo del modelo no lineal ...") 
        self._debbugingJoints()                     # Cleans and filter the joint data
        self._genNodes()                            # Generate nodes
        self._debugging_restr()                     # Cleans and filter restraint data
        self._fixNodes()                            # Applies boundary conditions
        self._debugging_materials()                 # Links section definitions to their corresponding material properties 
        self._assgNonlinearMaterials(ed_type)       # Assigns non-linear materials to frame sections
        self._generateElasticSlabs(shell_craking)   # Assigns elastic membrane plate sections
        self._debugging_FiberSection()              # Cleans and filter fiber section data
        self._generateFiberSections(npint)          # Generates fiber section definitions for frame elements
        self._consolidateInfo()                     # Consolidates and prepares frame and shell element information for model generation
        self._genElements(model_type)               # Creates elastic/non-linear frame elements
        self._genSlabs()                            # Generates shell elements (slabs) 
        self._calculate_beamloads(load_case)        # Calculate beam distributed loads
        self._appLoads()                            # Applies gravitational loads
        self._massDiaph()                           # Creates mass center nodes and rigid diaphragms for each floor in the model
        tqdm.write(" ✅​ Arquetipo generado ")
        
        self.dict['shell_craking'] = shell_craking
        self.dict['load_case'] = load_case
        self.dict['npint'] = npint
        self.dict['ed_type'] = ed_type
        
        return self.dict
        
    def genNONLinearModel(self, data_dict):
        
        self.OE_joints = data_dict['TABLE:  "OBJECTS AND ELEMENTS - JOINTS"']
        self.AS_restraints = data_dict['TABLE:  "JOINT ASSIGNMENTS - RESTRAINTS"']
        self.SC_frames = data_dict['TABLE:  "FRAME SECTIONS"']
        self.SC_shells = data_dict['TABLE:  "SHELL SECTIONS - SLAB"'] 
        self.GN_frames = data_dict['TABLE:  "FRAME GENERATION"']
        self.GN_Slabs = data_dict['TABLE:  "SLAB GENERATION"']
        self.oeframeloadB =  data_dict['TABLE:  "DISTRIBUTED LOAD - BEAMS"'] 
        self.oeframeloadC = data_dict['TABLE:  "PUNTUAL LOAD - COLUMNS"']
        self.masses = data_dict['TABLE:  "MASS SUMMARY BY DIAPHRAGM"']
        self.C_rebar = data_dict['TABLE:  "CONCRETE COLUMN REBAR DATA"'] 
        self.B_rebar= data_dict['TABLE:  "CONCRETE BEAM REBAR DATA"'] 
        self.AS_offsets = data_dict['TABLE:  "FRAME ASSIGNMENTS - OFFSETS"']

                
        shell_craking = data_dict['shell_craking']
        npint = data_dict['npint']
        ed_type = data_dict['ed_type']
        
        if isinstance(self.AS_offsets, str) and self.AS_offsets == "Table Not Found":
            self.nofound = 'Yes'
        else:
            self.nofound = 'No'
        
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        self._genNodes()                            # Generate nodes
        self._fixNodes()                            # Applies boundary conditions
        self._assgNonlinearMaterials(ed_type)       # Assigns non-linear materials to frame sections
        self._generateElasticSlabs(shell_craking)   # Assigns elastic membrane plate sections
        self._generateFiberSections(npint)          # Generates fiber section definitions for frame elements
        self._genElements('NLM')                    # Creates elastic/non-linear frame elements
        self._genSlabs()                            # Generates shell elements (slabs) 
        self._appLoads()                            # Applies gravitational loads
        self._massDiaph()                           # Creates mass center nodes and rigid diaphragms for each floor in the model

class ElasticModelBuilderWCMF(ModelBuilder):
    
    def genInitialElasticModel(self, main_path, model_type = 'EM', load_case = '(0) 1CM + 0.25CV', npint = 5, shell_craking = 1):
        self.outputs_model = os.path.abspath(os.path.join(main_path, 'outputs', 'opensees_models'))
        
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        tqdm.write(" 🔄 Generando el arquetipo del modelo elástico ...") 
        self._debbugingJoints()
        self._genNodes()
        self._debugging_restr()
        self._fixNodes()
        self._debugging_materials()
        self._assgNonlinearMaterials('DMO')
        self._generateElasticSlabs()
        self._assgWallElasticMaterials()
        self._consolidateInfoPiers()
        self._genWalls()
        self._genSlabs()
        tqdm.write(" ✅​ Arquetipo generado ")

#%% PRUEBAS
# # Esto genera un json para no sobreescribir la data
# main_path = os.getcwd()
# folder_data = os.path.join(main_path, 'data', 'temporary', 'json')
# ut.import_csi_data(main_path, 'input_model_0011', folder_data)

# data_file = ut.process_json_data(folder_data, 'model_information')

# #%% PRUEBAS GENERAR MODELO
# main_class = ModelBuilder(data_file)
# # Generar modelo base
# ops.wipe()
# ops.model('basic', '-ndm', 3, '-ndf', 6)
# # Empezar a generar el modelo con las funciones -- aplicables para muros, porticos, combinados
# main_class._debbugingJoints()
# main_class._genNodes()
# main_class._debugging_restr()
# main_class._fixNodes()
# main_class._debugging_materials()
# main_class._assgNonlinearMaterials('DMO')
# main_class._generateElasticSlabs()
# main_class._assgWallElasticMaterials()
# main_class._consolidateInfoPiers()
# main_class._genWalls()
# main_class._genSlabs()

# import vfo.vfo as vfo
# vfo.plot_model()