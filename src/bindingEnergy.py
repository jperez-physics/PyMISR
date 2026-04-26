"""
ITERATIVE MODEL CONSTRUCTION ALGORITHM USING SYMBOLIC REGRESSION (PyMISR)
Purpose: Find a mathematical equation that describes the Binding Energy (BE) 
comparing it with the MISR theoretical model. Version: 0.3

Version 0.2: 
- Training set reduced to Z <= 50 only
Version 0.3:
- Added mutual_info_regression for selecting the best variables
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from pysr import PySRRegressor
import matplotlib.pyplot as plt
import tempfile
import sympy as sp
import textwrap
from pathlib import Path
import optuna

# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================
# Locate the dataset relative to the script folder
script_dir = Path(__file__).resolve().parent
dataset_path = script_dir.parent / "datasets" / "be_exp.csv"
dataset = pd.read_csv(dataset_path)

# Cleaning: Ensure columns are numeric and remove null values.
for col in dataset.columns:
    if col != 'spinAndParity': 
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

dataset = dataset.dropna()

# =============================================================================
# FILTER: Train only with nuclei Z <= 50
# =============================================================================
full_dataset = dataset.copy()          # Full copy for final comparison
dataset = dataset[dataset['Z'] <= 50].reset_index(drop=True)
print(f"\n[FILTER] Training restricted to Z <= 50: {len(dataset)} nuclei (out of {len(full_dataset)} total)")

# Select numeric columns
numeric_cols = dataset.select_dtypes(include=[np.number]).columns

# Creation of physical variables for the FULL dataset
full_dataset['A']    = full_dataset['Z'] + full_dataset['N']
full_dataset['BEpA'] = full_dataset['BE'] / full_dataset['A']
full_dataset['I']    = (full_dataset['N'] - full_dataset['Z']) / full_dataset['A']
full_dataset['P']    = (full_dataset['nn'] * full_dataset['np']) / (full_dataset['nn'] + full_dataset['np'])

# Creation of physical variables
# A = Z + N (Mass number)
dataset['A'] = dataset['Z'] + dataset['N']

# BEpA = BE / A (Binding energy per nucleon)
dataset['BEpA'] = dataset['BE'] / dataset['A']

# I = (N - Z) / A (Relative isospin)
dataset['I'] = (dataset['N'] - dataset['Z']) / dataset['A']

# Casten Factor P = (nn * np) / (nn + np)
# Note: nn is the number of valence neutrons and np is the number of valence protons
dataset['P'] = (dataset['nn'] * dataset['np']) / (dataset['nn'] + dataset['np'])

# View the dataset with new variables
print("\nDataset with new variables (A, BEpA, I, P):")
print(dataset[['Z', 'N', 'A', 'BE', 'BEpA', 'I', 'nn', 'np', 'P']].head())

# Definition of features (X) and target variable (y)
X = dataset[['Z', 'N', 'A', 'P', 'I', 'nn', 'np']]
y = dataset['BE']

# Experimental sigma
sigma_exp = dataset['bindingEnergyUncertainty'] if 'bindingEnergyUncertainty' in dataset.columns else 0.0
weights = 1.0 / (1.0 + sigma_exp)

# K-Fold configuration
# The entire Z <= 50 set is used for training with cross-validation.
n_splits = 5
X_train_cv = X.reset_index(drop=True)
y_train_cv = y.reset_index(drop=True)
weights_train_cv = weights.reset_index(drop=True)

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

s = 4    # Size of the subset of variables to select

# PySRRegressor parameters
niterations        = 10
n_t                = 35
ncycles_per_iter   = 2000
maxdepth           = 10
parsimony          = 0.0001
random_state_pysr  = 11
binary_ops         = ["+", "-", "*", "/"]
unary_ops          = ["square", "sqrt", "cbrt", "inv"]

# Main while loop variables
max_iter = 1
theta = 0.04
iteration = 1
improvement_ratio = 1.0
previous_mse = float('inf')

final_models = [] # To store the sum of equations (final model)

print(f"\nStarting Iterative Model Construction...")
while iteration <= max_iter and improvement_ratio >= theta:
    print(f"\n==========================================")
    print(f" ITERATION {iteration}")
    print(f"==========================================")
    
    best_equations = []

    print(f"\nPerforming K-Folding and Symbolic Regression:")
    for fold, (train_index, val_index) in enumerate(kf.split(X_train_cv)):
        X_train, X_val = X_train_cv.iloc[train_index], X_train_cv.iloc[val_index]
        y_train, y_val = y_train_cv.iloc[train_index], y_train_cv.iloc[val_index]
        weights_train = weights_train_cv.iloc[train_index]
        
        # 1. Select a subset of variables X_sub based on importance scores
        
        # 1. Define which variables are discrete to improve Mutual Information precision
        # X columns: ['Z', 'N', 'A', 'P', 'I', 'nn', 'np']
        is_discrete = [True, True, True, False, False, True, True]
        
        # Score 1: Mutual Information (MI)
        mi_scores = mutual_info_regression(X_train, y_train, discrete_features=is_discrete, random_state=42)
        mi_s = pd.Series(mi_scores, index=X.columns)
        
        # Score 2: Gradient Boosting (GBR) for predictive importance
        gbr = GradientBoostingRegressor(random_state=42)
        gbr.fit(X_train, y_train)
        gbr_s = pd.Series(gbr.feature_importances_, index=X.columns)
        
        # 2. HYBRID SELECTION: Average of normalized scores
        # We normalize so that both metrics have the same weight (0 to 1)
        mi_norm = mi_s / mi_s.max() if mi_s.max() > 0 else mi_s
        gbr_norm = gbr_s / gbr_s.max() if gbr_s.max() > 0 else gbr_s
        
        hybrid_importance = (mi_norm + gbr_norm) / 2
        hybrid_importance = hybrid_importance.sort_values(ascending=False)
        
        selected_vars = hybrid_importance.head(s).index.tolist()
        
        X_sub_train = X_train[selected_vars]
        
        print(f"\n--- Fold {fold+1} ---")
        print(f"Selected variables: {selected_vars}")
        
        # 2. Perform Symbolic Regression with n_t terms (maxsize)
        X_sub_train_pysr = X_sub_train.rename(columns={'N': 'Neut', 'I': 'Iso'})
        X_units = ["" for _ in selected_vars] # Dimensionless variables

        model = PySRRegressor(
            # --- Available Operators ---
            binary_operators=binary_ops,
            unary_operators=unary_ops,
            
            # --- Search Parameters ---
            niterations=niterations,
            ncycles_per_iteration=ncycles_per_iter,
            maxsize=n_t,
            maxdepth=maxdepth,
            
            # --- Complexity Constraints ---
            constraints={
                "square": 1,
                "cbrt": 1,
                "sqrt": 1,
                "inv": 1
            },
            nested_constraints={
                "square": {"square": 0},
                "cbrt": {"cbrt": 0},
                "sqrt": {"sqrt": 0},
                "inv": {"inv": 0}
            },
            
            # --- Penalties and Loss Functions ---
            elementwise_loss="loss(prediction, target, weight) = (weight * (prediction - target))^2",
            parsimony=parsimony,
            complexity_of_variables=1,
            complexity_of_constants=1,
            dimensional_constraint_penalty=10**5,

            # --- Execution Configuration ---
            random_state=random_state_pysr,
            deterministic=True,
            parallelism='serial',
            verbosity=0,
            progress=True,
            tempdir=tempfile.gettempdir(),
            delete_tempfiles=True
        )
        
        try:
            model.fit(
                X_sub_train_pysr, 
                y_train, 
                weights=weights_train,
                X_units=X_units,
                y_units="Constants.MeV"
            )
            
            # Evaluation on internal validation
            X_sub_val_pysr = X_val[selected_vars].rename(columns={'N': 'Neut', 'I': 'Iso'})
            y_pred_val = model.predict(X_sub_val_pysr)
            mse_val = np.mean((y_val - y_pred_val)**2)
            
            best_equation = str(model.sympy())
            best_equations.append({
                'fold': fold + 1,
                'variables': selected_vars,
                'equation': best_equation,
                'mse_val': mse_val,
                'model': model
            })
            print(f"Best Equation Fold {fold+1}: {best_equation} (MSE Val: {mse_val:.4f})")
        except Exception as e:
            print(f"Error in Symbolic Regression in Fold {fold+1}: {e}")

    # Final iteration summary
    print(f"\n--- Best Performing Models Stored in Iteration {iteration} ---")
    for r in best_equations:
        print(f"Fold {r['fold']}: [Vars: {r['variables']}] -> {r['equation']} (Validation MSE: {r['mse_val']:.4f})")

    # -- Select the best equation among all groups and update Y --
    if best_equations:
        # 1. Select the best equation (lowest validation MSE)
        best_global_result = min(best_equations, key=lambda x: x['mse_val'])
        print(f"\n[!] The best global equation of iteration {iteration} is from Fold {best_global_result['fold']} with an MSE of {best_global_result['mse_val']:.4f}")
        
        best_model = best_global_result['model']
        optimal_vars = best_global_result['variables']
        
        # Store in final models list
        final_models.append({
            'iteration': iteration,
            'equation': best_global_result['equation'],
            'model': best_model,
            'variables': optimal_vars,
            'mse_val': best_global_result['mse_val']
        })
        
        # Calculate improvement ratio
        if previous_mse != float('inf') and previous_mse > 0:
            improvement_ratio = (previous_mse - best_global_result['mse_val']) / previous_mse
            print(f"Improvement ratio compared to previous iteration: {improvement_ratio:.4f}")
        
        previous_mse = best_global_result['mse_val']
        
        # 2. Update Y to be the residuals of the current best model
        X_global_sub = X[optimal_vars].rename(columns={'N': 'Neut', 'I': 'Iso'})
        y_pred_global = best_model.predict(X_global_sub)
        
        y = y - y_pred_global # Update to Residuals
        
        print("\nVariable Y updated to residuals of the best model.")
        print("First 5 values of the new residuals (Y):")
        print(y.head())
    else:
        print("\n[!] No valid equations found in this iteration. Aborting.")
        break
        
    iteration += 1

# =============================================================================
# MODEL FINALIZATION
# =============================================================================
print("\n==========================================")
print(" MODEL FINALIZATION")
print("==========================================")
print("Equations selected in each iteration:")
sympy_equations = []
for m in final_models:
    print(f"Iteration {m['iteration']}: {m['equation']} (Vars: {m['variables']})")
    sympy_equations.append(m['model'].sympy())

try:
    total_sympy_equation = sum(sympy_equations)
    simplified_equation = sp.simplify(total_sympy_equation)
    # Round all float numbers to 2 decimal places
    rounded_equation = simplified_equation.xreplace(
        {n: sp.Float(round(float(n), 2)) for n in simplified_equation.atoms(sp.Number) if isinstance(n, sp.Float)}
    )
    simplified_created_model_str = str(rounded_equation)

    if isinstance(rounded_equation, sp.Piecewise):
        equation_for_latex = rounded_equation.args[-1][0]
    else:
        equation_for_latex = rounded_equation
    
    latex_equation = sp.latex(equation_for_latex)
    
    replacements = {"Neut": "N", "Iso": "I", "nn": "n_n", "np": "n_p"}
    for old, new in replacements.items():
        latex_equation = latex_equation.replace(old, new)
    print(f"\nFinal Summed and Simplified Model:\n{simplified_created_model_str}")
except Exception as e:
    print(f"Error simplifying with sympy: {e}")
    simplified_created_model_str = " + ".join([f"({m['equation']})" for m in final_models])
    latex_equation = simplified_created_model_str.replace("Neut", "N").replace("Iso", "I").replace("nn", "n_n").replace("np", "n_p").replace("*", "\\cdot ")
    print(f"\nFinal Summed Model:\n{simplified_created_model_str}")


print("\n==========================================")
print(" COMPARISON WITH THEORETICAL MISR EQUATION")
print("==========================================")

# Provided adjusted parameters
a_misr = 1.10
b_misr = 32.43
c_misr = 16.70
eta_0 = 1.0

Z_total = full_dataset['Z']
N_total = full_dataset['N']
A_total = full_dataset['A']
I_total = full_dataset['I']
BE_real = full_dataset['BE']

# Calculation of BE_MISR
# BE = eta_0 * Z * (1 + 1/N - (a*N/Z^2)) * [I * (b - (A^{1/3}*N)/Z) + c]
parenthesis_term = 1 + 1/N_total - (a_misr * N_total) / (Z_total**2)
bracket_term = I_total * (b_misr - (A_total**(1/3) * N_total) / Z_total) + c_misr

BE_MISR = eta_0 * Z_total * parenthesis_term * bracket_term

# Calculation of total prediction from the iterative PyMISR model
BE_PyMISR = np.zeros(len(full_dataset))
for m in final_models:
    vars_opt = m['variables']
    model = m['model']
    X_sub = full_dataset[vars_opt].rename(columns={'N': 'Neut', 'I': 'Iso'})
    BE_PyMISR += model.predict(X_sub)

# Calculation of Mean Absolute Errors (MAE)
mae_pymisr   = np.mean(np.abs(BE_real - BE_PyMISR))
mae_misr   = np.mean(np.abs(BE_real - BE_MISR))

print(f"PyMISR Model MAE: {mae_pymisr:.4f}")
print(f"MISR Equation MAE:                 {mae_misr:.4f}")

# --- Comparative Plot Generation ---
print("\nGenerating comparative plot...")

# Activate global LaTeX font for the entire figure
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

plt.figure(figsize=(10, 30))
plt.subplots_adjust(top=0.95, hspace=0.45)

# Main title with the equation
try:
    plt.suptitle(r"Total Simplified PyMISR Equation:" + "\n" + rf"$BE = {latex_equation}$",
                 fontsize=14, color='darkblue', fontweight='bold')
except Exception:
    plt.suptitle(f"Total Simplified PyMISR Equation:\nBE = {simplified_created_model_str[:100]}...",
                 fontsize=12, color='darkblue', fontweight='bold')

# --- Panel 1: BE/A vs N ---
plt.subplot(5, 1, 1)
plt.scatter(N_total, BE_real/A_total, alpha=0.6, label=r'Experimental', color='gray', s=5)
plt.scatter(N_total, BE_PyMISR/A_total, alpha=0.8, label=r'PyMISR', color='blue', s=5)
plt.scatter(N_total, BE_MISR/A_total, alpha=0.8, label=r'MISR (Theoretical)', color='red', marker='x', s=5)

plt.xlabel(r'Neutron Number $N$')
plt.ylabel(r'$BE/A$ (MeV)')
plt.title(r'Binding Energy per Nucleon vs $N$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# --- Panel 2: |\delta BE/A| vs N (Absolute Error per nucleon) ---
plt.subplot(5, 1, 2)
abs_err_pymisr_per_A = np.abs(BE_PyMISR - BE_real) / A_total
abs_err_misr_per_A = np.abs(BE_MISR - BE_real) / A_total

plt.scatter(N_total, abs_err_pymisr_per_A, alpha=0.8, label=rf'PyMISR (MAE/A: {np.mean(abs_err_pymisr_per_A):.4f})', color='blue', s=5)
plt.scatter(N_total, abs_err_misr_per_A, alpha=0.8, label=rf'MISR (MAE/A: {np.mean(abs_err_misr_per_A):.4f})', color='red', marker='x', s=5)

plt.xlabel(r'Neutron Number $N$')
plt.ylabel(r'$|\Delta BE/A|$ (MeV)')
plt.title(r'Absolute Error per Nucleon vs $N$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# --- Panel 3: delta BE vs N ---
plt.subplot(5, 1, 3)
pymisr_residuals_N = BE_PyMISR - BE_real
misr_residuals_N = BE_MISR - BE_real

plt.scatter(N_total, pymisr_residuals_N, alpha=0.8, label=rf'PyMISR (MAE: {mae_pymisr:.2f})', color='blue', s=5)
plt.scatter(N_total, misr_residuals_N, alpha=0.8, label=rf'MISR (MAE: {mae_misr:.2f})', color='red', marker='x', s=5)
plt.axhline(0, color='k', linestyle='--', alpha=0.7)

plt.xlim(10, 90)
plt.xlabel(r'Neutron Number $N$')
plt.ylabel(r'$\Delta BE$ (MeV)')
plt.title(r'Residuals vs $N$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# --- Panel 4: delta BE/A vs Z ---
plt.subplot(5, 1, 4)
pymisr_residuals_per_A = (BE_PyMISR - BE_real) / A_total
misr_residuals_per_A = (BE_MISR - BE_real) / A_total

plt.scatter(Z_total, pymisr_residuals_per_A, alpha=0.8, label=rf'PyMISR (MAE: {mae_pymisr:.2f})', color='blue', s=5)
plt.scatter(Z_total, misr_residuals_per_A, alpha=0.8, label=rf'MISR (MAE: {mae_misr:.2f})', color='red', marker='x', s=5)
plt.axhline(0, color='k', linestyle='--', alpha=0.7)
plt.axvline(50, color='gray', linestyle=':', linewidth=1.5, alpha=0.8, label=r'$Z = 50$ (training limit)')

plt.xlim(10, 120)
plt.xlabel(r'Atomic Number $Z$')
plt.ylabel(r'$\Delta BE / A$ (MeV)')
plt.title(r'Residuals per Nucleon vs $Z$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# --- Panel 5: Configuration parameters table ---
ax_cfg = plt.subplot(5, 1, 5)
ax_cfg.axis('off')

config_params = [
    [r"Version",               r"0.3"],
    [r"Max training $Z$", r"$Z \leq 50$"],
    [r"n_splits",    str(n_splits)],
    [r"s (selected vars.)", str(s)],
    [r"n_t (maxsize)",          str(n_t)],
    [r"max_iter",    str(max_iter)],
    [r"$\theta$ (min improvement ratio)",     str(theta)],
    [r"niterations",              str(niterations)],
    [r"ncycles_per_iteration",  str(ncycles_per_iter)],
    [r"maxdepth",                 str(maxdepth)],
    [r"parsimony",                str(parsimony)],
    [r"random_state",            str(random_state_pysr)],
    [r"Binary operators",               ", ".join(binary_ops)],
    [r"Unary operators",                ", ".join(unary_ops)],
]

table = ax_cfg.table(
    cellText=config_params,
    colLabels=[r"Parameter", r"Value"],
    cellLoc='left',
    loc='center',
    colWidths=[0.55, 0.35],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.55)

# Header style
for col in range(2):
    table[(0, col)].set_facecolor('#2c3e50')
    table[(0, col)].set_text_props(color='white', fontweight='bold')

# Alternating rows
for row in range(1, len(config_params) + 1):
    fc = '#eaf0f6' if row % 2 == 0 else 'white'
    for col in range(2):
        table[(row, col)].set_facecolor(fc)

ax_cfg.set_title(r'Configuration Parameters', fontsize=12, pad=12)

# Export plot to 'results' folder in the project root
output_dir = script_dir.parent / "results"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "MISRComparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n[OK] Plot successfully exported to: {output_path}")