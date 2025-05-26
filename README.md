# CSI to OpenSees - 3D Structural Model Converter
EstrucMed Ingenieria Especializada S.A.S

## Introduction

This manual aims to guide users in utilizing a tool developed to simplify the transfer of three-dimensional structural models from SAP2000 or ETABS to OpenSeesPy.

The tool, **CSI to OpenSees**, developed in Python, enables the conversion of structural models created in ETABS or SAP into a format compatible with OpenSeesPy, facilitating the execution of advanced nonlinear analyses.

OpenSeesPy is widely recognized for its flexibility in nonlinear simulations and high customizability, making it ideal for detailed analyses not possible within CSI software. However, manually recreating complex models in OpenSeesPy is tedious and error-prone, potentially affecting the accuracy of simulations. The tool described in this manual automates this process, ensuring a fast and accurate conversion.

The objective of this guide is to provide clear instructions on how to:

- Prepare the necessary structural and design data from ETABS or SAP2000 models.
- Correctly configure the Python environment.
- Execute the conversion scripts.
- Verify the fidelity of the generated OpenSeesPy model.

This manual is intended for structural engineers familiar with CSI software (ETABS/SAP2000), with or without prior experience in OpenSeesPy. Its design minimizes the learning curve and maximizes efficiency in the conversion process, allowing users to focus on performing advanced analyses.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage Instructions](#usage-instructions)
5. [Posible Errors and Warnings](#posible-errors-and-warnings)
6. [Current Limitations](#current-limitations) 
7. [Load Combination Input Requirements](#load-combination-input-requirements)
8. [Future Development Roadmap](#future-development-roadmap)
9. [Authors](#authors)
10. [License](#license)

---

## Requirements

### Software Versions

- **ETABS** version 17.0.1
- **SAP2000** version 20 or higher
- **Python** version 3.12.x recommended
- **OpenSeesPy** version 3.5.0 or higher

### Python Packages

> These will be automatically installed when running the tool for the first time, but can also be manually installed via `pip install -r requirements.txt`.

Main packages:
- `openseespy`
- `vfo`
- `opseestools`
- `opsvis`
- `numpy`
- `pandas`
- `plotly`
- `dash` *(optional for user interface)*

### Recommended Hardware

- CPU: 4 cores or more
- RAM: 8 GB minimum (16 GB recommended for large models)
- Storage: SSD recommended for faster data handling

---

## Installation

This repository is private and accessible only to authorized collaborators.

Follow these steps to clone and set up the environment using Visual Studio Code (VSC), Anaconda, and Git:

1. **Install required software**:

    - Visual Studio Code (VSC)
    - Anaconda Distribution
    - Git for Windows (Download Git)`https://git-scm.com/downloads/win`

2. **Create a local project folder** where you will clone the repository.

3. **Open Visual Studio Code** and open a new integrated terminal (`Ctrl` + `Shift` + `+`).

    ⚠️ Important: Make sure the terminal is using **Command Prompt** (CMD) and not PowerShell or Bash.

4. **Create a new Python environment** using Anaconda:

```bash
conda create -p venv python=3.12
```

5. **Activate the environment**

```bash
conda activate venv/
```

6. **Initialize Git and clone the repository manually:**

    Run the following commands one by one (do not paste all at once):

```bash
echo "# CSI-to-OpenSees" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/dnovoar/CSI-to-OpenSees.git
git push -u origin main
```

7. **Important:**

    - Your GitHub user account must have access permissions to the private repository.
    - Otherwise, cloning and pushing will not work.

8. **Install required Python packages:**

    After cloning, in the activated environment, run:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```plaintext
/convertidor_csi_opensees/
│
├── main.py                                 # Master script
│
├── README.md                               # Quick user guide
│
├── /modules/                               # Specialized function modules
│   ├── archetype_elastic_model.py          # Functions to generate OpenSeesPy archetype for elastic models
│   ├── archetype_nonlinear_model.py        # Functions to generate OpenSeesPy archetype for non-linear    models
│   ├── validations.py                      # Functions to validate input files
│   ├── visual_verification.py              # Functions for visual checking
│   ├── modal_analysis.py                   # Modal analysis and comparison
│   ├── reaction_comparison.py              # Base reaction comparison
│   ├── drift_verification.py               # Drift verification (Equivalent Lateral Force method)
│   ├── pushover.py                         # Pushover analysis
│   ├── method_n2.py                        # N2 method application
│   ├── dynamic_analysis.py                 # Reduced dynamic analysis
│   ├── results_processing.py               # Processing and visualization of results
│   ├── forms.py                            # Forms to enter reinforcement and Section Designer sections
│   └── utilities.py                        # Helpers and general utility functions
│
├── /app/                                   # Integrate functions modules
│   ├── services/                           # Services
│   │   ├── generate_pushover_analysis.py   # Generate pushover analysis
│   │   ├── generate_verifications.py       # Proccess all verifications

├── /data/                                  # Main storage folder
│   ├── inputs/                             # User inputs
│   │   ├── models/                         # .e2k, .s2k files
│   │   ├── records/                        # Earthquake records (json)
│   │   └── configurations/                 # Configuration files (e.g., default_config.json)
│   ├── temporary/                          # Temporary or intermediate files
│   ├── previous_results/                   # Validated and tested building models
│   │   ├── opensees_models/                # OpenSeesPy models (.py or .tcl) (json probable)
│   │   └── reports/                        # Final reports from previous projects
│
├── /outputs/                               # Outputs from current analysis
│   ├── opensees_models/
│   ├── verification_results/
│   ├── pushover_results/
│   ├── n2_results/
│   ├── dynamic_analysis_results/
│   └── reports/
│
├── /dash_app/                              # (Optional) Dash application
│   ├── app.py
│   ├── layouts.py
│   └── callbacks.py
│
└── /notebooks/                             # (Required) Jupyter notebooks
    ├── 0_check_input.ipynb
    ├── 1_generate_model.ipynb
    ├── 2_verify_model.ipynb
    ├── 3_pushover_n2.ipynb
    └── 4_dynamic_analysis.ipynb
```

---

## Usage Instructions

1. Export your ETABS or SAP2000 model in `.e2k` or `.s2k` format.
2. Place the exported file into the  `/data/inputs/models/` folder.
3. Review configuration setting in `/data/inputs/configurations/default_config.json`.
4. run `main.py`.
5. Follow on-screen instructions for model verifications and analysis execution.
6. Outputs and reports will be saved in the `/outputs/` folder.

---

## Posible Errors and Warnings

| Issue                        | Description                                                                            | Action                                                    |
| :--------------------------- | :------------------------------------------------------------------------------------- | :-------------------------------------------------------- |
| Incorrect units              | Only SI units (kg, m, s) are supported.                                                | Re-export model with correct units.                       |
| Mass definition missing      | Self-weight and imposed load must be included.                                         | Check mass definitions in ETABS/SAP2000.                  |
| No diaphragms assigned       | Rigid diaphragms are required.                                                         | Assign diaphragms in ETABS/SAP before export.             |
| Section Designer sections    | Sections created via Section Designer in ETABS are not read automatically yet.         | Manually input properties via form (in future updates).   |
| Shell columns                | Columns modeled as shells are not supported.                                           | Model columns with frames (beam-column elements).         |
| Slabs with more than 4 nodes | OpenSees does not support more than 4-node slabs.                                      | Simplify meshing if necessary.                            |
| Unrestrained  nodes at base  | Nodes at the base without restraints will cause the stiffness matrix to be unsolvable. | Check boundary conditions before export.                  |
| Large reaction errors        | Reaction differences >10% may occur due to gravity loads not perfectly transferred.    | Optionally manually adjust distributed loads on beams.    |
| Missing reinforced in beams  | Beam reinforcement is not automatically extracted from ETABS.                          | Fill reinforcement data manually using the provided form. |

---

## Current Limitations

- Only **Moment Resisting Frame (MRF)** structures in reinforced concrete are currently supported.
- Models without diaphragms or with irregular slab geometries may fail.
- Beam/column sections defined in ETABS Section Designer must be manually input.
- Shell-based modeling of columns is not supported.
- Ribbed slabs modeled as beams in ETABS are **not correctly interpreted**. Users must delete rib beams from ETABS before export and adjust the slab thickness to an equivalent thickness manually.
- Combination of loads for base reaction verification must be **explicitly provided by the user** in a specific format.

---

## Load Combination Input Requirements

When performing base reaction verification between ETABS/SAP and OpenSeesPy:

- The user must provide the load combination manually.
- In ETABS, the combination name must follow the format:
  `"(0) 1CM + 0.25CV"`
Where:
- `CM` = Dead load + Superimposed dead load (both multiplied by factor 1.0 individually).
- `CV` = Live load.

This format allows OpenSeesPy, via internal parsing scripts, to automatically read and apply the correct load factors for dead and live loads.

**Important**: The dead load must already include both self-weight and superimposed dead load combined in CM.

---

## Future Development Roadmap

Future Development Roadmap
- Expand support to dual systems (Frames + Shear Walls).
- Automatic reading of Section Designer properties.
- Implementation of a full Dash-based user interface.
- Automatic generation of detailed PDF reports including formulas, assumptions, and analysis results.
- Expand compatibility with more ETABS/SAP versions.
- Extend functionalities for dynamic time-history analysis setup.

---

## Authors

Developed by Daniela Novoa, Frank Vidales and Orlando Arroyo and collaborators at EstrucMed Ingenieria Especializada S.A.S.
2025.

---

## License

[Private/Internal Use Only] This tool is intended for internal use within EstrucMed S.A.S. and its authorized collaborators.
