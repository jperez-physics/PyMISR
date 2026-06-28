"""
ITERATIVE MODEL CONSTRUCTION ALGORITHM USING SYMBOLIC REGRESSION (PyMISR)
========================================================================
Model + results stage. Discovers an analytical equation for the nuclear
Binding Energy (BE) and compares it against the theoretical MISR model.

This script handles ONLY the model and the data export. It writes:
  - results/MISRComparison_results.csv   (Z, N, BE_exp, BE_misr, BE_pymisr, residuals)
  - results/run_metadata.json            (config + discovered equation + MAE)

The plotting lives in `plots.py`, which reads those two files (fully decoupled,
so figures can be regenerated without retraining).

Version: 0.4
  v0.2: training set reduced to Z <= 50 only.
  v0.3: mutual_info_regression added for variable selection.
  v0.4: soft physical constraint BE(Z=0) = 0 via synthetic anchor points
        (no nucleus without protons => binding energy must vanish at Z=0).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from pysr import PySRRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold

# =============================================================================
# CONFIGURATION (all knobs in one place)
# =============================================================================
VERSION = "0.5"

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR.parent / "datasets" / "be_exp.csv"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
CSV_PATH = RESULTS_DIR / "MISRComparison_results.csv"
META_PATH = RESULTS_DIR / "run_metadata.json"

# --- Training filter ---
MAX_TRAIN_Z = 50  # train only with nuclei Z <= MAX_TRAIN_Z

# --- Features ---
FEATURES = ["Z", "N", "A", "P", "I", "nn", "np"]
IS_DISCRETE = [True, True, True, False, False, True, True]  # for mutual information

# --- Symbolic regression ---
N_SPLITS = 5
S = 5 
NITERATIONS = 40
N_T = 45  # maxsize (MISR expression is ~30 nodes)
NCYCLES_PER_ITER = 2000
MAXDEPTH = 12
PARSIMONY = 0.0001
RANDOM_STATE_PYSR = 11
POPULATIONS = 40
POPULATION_SIZE = 50
BINARY_OPS = ["+", "-", "*", "/"]
UNARY_OPS = ["square", "cbrt", "inv"]

# --- Soft physical constraint: BE(Z=0) = 0 ---
ANCHOR_Z0 = False  # enable the soft constraint
ANCHOR_WEIGHT = 10.0  # weight per anchor point (tune if needed)
ANCHOR_N_GRID = list(range(0, 181, 10))  # neutron-number sweep for the anchors

# --- Iterative loop ---
MAX_ITER = 1
THETA = 0.04  # minimum improvement ratio to keep iterating

# --- Theoretical MISR parameters ---
A_MISR = 1.10
B_MISR = 32.43
C_MISR = 16.70
ETA_0 = 1.0

RENAME_FOR_PYSR = {"N": "Neut", "I": "Iso"}


# =============================================================================
# DATA
# =============================================================================
def load_dataset(path=DATASET_PATH):
    """Load the experimental dataset, coerce to numeric and drop nulls."""
    df = pd.read_csv(path)
    for col in df.columns:
        if col != "spinAndParity":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def engineer_features(df):
    """Add the derived physical variables A, BEpA, I and the Casten factor P."""
    df = df.copy()
    df["A"] = df["Z"] + df["N"]  # mass number
    df["BEpA"] = df["BE"] / df["A"]  # binding energy per nucleon
    df["I"] = (df["N"] - df["Z"]) / df["A"]  # relative isospin
    df["P"] = (df["nn"] * df["np"]) / (df["nn"] + df["np"])  # Casten factor
    return df


# =============================================================================
# SOFT CONSTRAINT: anchors at Z = 0
# =============================================================================
def make_anchors(selected_vars, n_grid=ANCHOR_N_GRID, weight=ANCHOR_WEIGHT):
    """Build synthetic Z=0 anchor rows enforcing the soft constraint BE(Z=0)=0.

    For any subset of FEATURES the feature values are made physically consistent
    at Z=0 (A=N, I=1, P=0, nn=np=0). Returns the feature DataFrame (already
    renamed N->Neut, I->Iso), the target series (all zeros) and the weight series.
    """
    rows = []
    for n in n_grid:
        full = {
            "Z": 0.0,
            "N": float(n),
            "A": float(n),  # A = Z + N = N
            "I": 1.0,  # (N - Z)/A = N/N = 1   (N > 0)
            "P": 0.0,  # no valence protons (np = 0) -> P = 0
            "nn": 0.0,
            "np": 0.0,  # no valence protons/neutrons reference
        }
        rows.append({v: full[v] for v in selected_vars})
    Xa = pd.DataFrame(rows).rename(columns=RENAME_FOR_PYSR)
    ya = pd.Series([0.0] * len(n_grid))  # target BE = 0 at Z = 0
    wa = pd.Series([float(weight)] * len(n_grid))  # high weight to enforce it
    return Xa, ya, wa


# =============================================================================
# SYMBOLIC REGRESSION
# =============================================================================
def select_variables(X_train, y_train, s=S):
    """Hybrid variable selection: average of normalized MI and GBR importances."""
    # Score 1: Mutual Information (MI)
    mi_scores = mutual_info_regression(
        X_train, y_train, discrete_features=IS_DISCRETE, random_state=42
    )
    mi_s = pd.Series(mi_scores, index=X_train.columns)

    # Score 2: Gradient Boosting (GBR) predictive importance
    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y_train)
    gbr_s = pd.Series(gbr.feature_importances_, index=X_train.columns)

    # Normalize both metrics to [0, 1] and average
    mi_norm = mi_s / mi_s.max() if mi_s.max() > 0 else mi_s
    gbr_norm = gbr_s / gbr_s.max() if gbr_s.max() > 0 else gbr_s
    hybrid_importance = ((mi_norm + gbr_norm) / 2).sort_values(ascending=False)

    return hybrid_importance.head(s).index.tolist()


def build_pysr_model():
    """Build a PySRRegressor configured with the global hyperparameters."""
    return PySRRegressor(
        # --- Available Operators ---
        binary_operators=BINARY_OPS,
        unary_operators=UNARY_OPS,
        # --- Search Parameters ---
        niterations=NITERATIONS,
        ncycles_per_iteration=NCYCLES_PER_ITER,
        maxsize=N_T,
        maxdepth=MAXDEPTH,
        populations=POPULATIONS,
        population_size=POPULATION_SIZE,
        # --- Complexity Constraints ---
        # Allow unary operators to compose (e.g. A^(2/3)=cbrt(square(A)),
        # A^(-1/3)=inv(cbrt(A))) without forcing them. nested self=0 forbids odd
        # self-nesting. (sqrt removed in v0.5: not part of MISR's vocabulary.)
        constraints={"square": 3, "cbrt": 3, "inv": 3},
        nested_constraints={
            "square": {"square": 0},
            "cbrt": {"cbrt": 0},
            "inv": {"inv": 0},
        },
        # --- Penalties and Loss Functions ---
        elementwise_loss="loss(prediction, target, weight) = (weight * (prediction - target))^2",
        parsimony=PARSIMONY,
        complexity_of_variables=1,
        complexity_of_constants=1,
        dimensional_constraint_penalty=10**5,
        # --- Execution Configuration ---
        random_state=RANDOM_STATE_PYSR,
        deterministic=True,
        parallelism="serial",
        verbosity=0,
        progress=True,
        tempdir=tempfile.gettempdir(),
        delete_tempfiles=True,
    )


def run_fold(X_train, y_train, weights_train, X_val, y_val, fold):
    """Select variables, fit PySR (with anchors) and evaluate on the fold."""
    selected_vars = select_variables(X_train, y_train)
    print(f"\n--- Fold {fold + 1} ---")
    print(f"Selected variables: {selected_vars}")

    X_sub_train_pysr = X_train[selected_vars].rename(columns=RENAME_FOR_PYSR)
    X_units = ["" for _ in selected_vars]  # dimensionless variables

    model = build_pysr_model()

    # Inject soft Z=0 anchors ONLY into the PySR training set
    # (feature selection and validation MSE stay on real data only)
    if ANCHOR_Z0:
        Xa, ya, wa = make_anchors(selected_vars)
        X_fit = pd.concat([X_sub_train_pysr, Xa], ignore_index=True)
        y_fit = pd.concat([y_train.reset_index(drop=True), ya], ignore_index=True)
        w_fit = pd.concat([weights_train.reset_index(drop=True), wa], ignore_index=True)
    else:
        X_fit, y_fit, w_fit = X_sub_train_pysr, y_train, weights_train

    model.fit(X_fit, y_fit, weights=w_fit, X_units=X_units, y_units="Constants.MeV")

    # Evaluation on internal validation
    X_sub_val_pysr = X_val[selected_vars].rename(columns=RENAME_FOR_PYSR)
    y_pred_val = model.predict(X_sub_val_pysr)
    mse_val = np.mean((y_val - y_pred_val) ** 2)
    best_equation = str(model.sympy())
    print(f"Best Equation Fold {fold + 1}: {best_equation} (MSE Val: {mse_val:.4f})")

    return {
        "fold": fold + 1,
        "variables": selected_vars,
        "equation": best_equation,
        "mse_val": mse_val,
        "model": model,
    }


def run_symbolic_regression(X, y, weights):
    """Iterative model construction with K-Fold CV and residual updates."""
    X_train_cv = X.reset_index(drop=True)
    y_train_cv = y.reset_index(drop=True)
    weights_train_cv = weights.reset_index(drop=True)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    final_models = []
    iteration = 1
    improvement_ratio = 1.0
    previous_mse = float("inf")

    print("\nStarting Iterative Model Construction...")
    while iteration <= MAX_ITER and improvement_ratio >= THETA:
        print(f"\n==========================================")
        print(f" ITERATION {iteration}")
        print(f"==========================================")
        print("\nPerforming K-Folding and Symbolic Regression:")

        best_equations = []
        for fold, (train_index, val_index) in enumerate(kf.split(X_train_cv)):
            X_tr, X_val = X_train_cv.iloc[train_index], X_train_cv.iloc[val_index]
            y_tr, y_val = y_train_cv.iloc[train_index], y_train_cv.iloc[val_index]
            w_tr = weights_train_cv.iloc[train_index]
            try:
                best_equations.append(run_fold(X_tr, y_tr, w_tr, X_val, y_val, fold))
            except Exception as e:
                print(f"Error in Symbolic Regression in Fold {fold + 1}: {e}")

        print(f"\n--- Best Performing Models Stored in Iteration {iteration} ---")
        for r in best_equations:
            print(
                f"Fold {r['fold']}: [Vars: {r['variables']}] -> {r['equation']} "
                f"(Validation MSE: {r['mse_val']:.4f})"
            )

        if not best_equations:
            print("\n[!] No valid equations found in this iteration. Aborting.")
            break

        # Select the best equation (lowest validation MSE)
        best = min(best_equations, key=lambda x: x["mse_val"])
        print(
            f"\n[!] The best global equation of iteration {iteration} is from "
            f"Fold {best['fold']} with an MSE of {best['mse_val']:.4f}"
        )
        final_models.append(
            {
                "iteration": iteration,
                "equation": best["equation"],
                "model": best["model"],
                "variables": best["variables"],
                "mse_val": best["mse_val"],
            }
        )

        if previous_mse != float("inf") and previous_mse > 0:
            improvement_ratio = (previous_mse - best["mse_val"]) / previous_mse
            print(
                f"Improvement ratio compared to previous iteration: {improvement_ratio:.4f}"
            )
        previous_mse = best["mse_val"]

        # Update Y to be the residuals of the current best model
        X_global_sub = X[best["variables"]].rename(columns=RENAME_FOR_PYSR)
        y = y - best["model"].predict(X_global_sub)
        print("\nVariable Y updated to residuals of the best model.")
        print("First 5 values of the new residuals (Y):")
        print(y.head())

        iteration += 1

    return final_models


# =============================================================================
# FINALIZATION & PREDICTIONS
# =============================================================================
def finalize_equation(final_models):
    """Sum the per-iteration equations and simplify with SymPy.

    Returns (equation_str, equation_latex). Falls back to a plain string sum
    if SymPy simplification fails.
    """
    print("\n==========================================")
    print(" MODEL FINALIZATION")
    print("==========================================")
    print("Equations selected in each iteration:")
    sympy_equations = []
    for m in final_models:
        print(f"Iteration {m['iteration']}: {m['equation']} (Vars: {m['variables']})")
        sympy_equations.append(m["model"].sympy())

    try:
        total = sum(sympy_equations)
        simplified = sp.simplify(total)
        rounded = simplified.xreplace(
            {
                n: sp.Float(round(float(n), 2))
                for n in simplified.atoms(sp.Number)
                if isinstance(n, sp.Float)
            }
        )
        equation_str = str(rounded)

        eq_for_latex = (
            rounded.args[-1][0] if isinstance(rounded, sp.Piecewise) else rounded
        )
        equation_latex = sp.latex(eq_for_latex)
        for old, new in {"Neut": "N", "Iso": "I", "nn": "n_n", "np": "n_p"}.items():
            equation_latex = equation_latex.replace(old, new)
        print(f"\nFinal Summed and Simplified Model:\n{equation_str}")
    except Exception as e:
        print(f"Error simplifying with sympy: {e}")
        equation_str = " + ".join([f"({m['equation']})" for m in final_models])
        equation_latex = (
            equation_str.replace("Neut", "N")
            .replace("Iso", "I")
            .replace("nn", "n_n")
            .replace("np", "n_p")
            .replace("*", "\\cdot ")
        )
        print(f"\nFinal Summed Model:\n{equation_str}")

    return equation_str, equation_latex


def compute_misr(full_df):
    """Theoretical MISR binding energy on the full dataset."""
    Z, N, A, I = full_df["Z"], full_df["N"], full_df["A"], full_df["I"]
    parenthesis = 1 + 1 / N - (A_MISR * N) / (Z**2)
    bracket = I * (B_MISR - (A ** (1 / 3) * N) / Z) + C_MISR
    return ETA_0 * Z * parenthesis * bracket


def compute_pymisr(full_df, final_models):
    """Sum the predictions of every iteration's model on the full dataset."""
    be = np.zeros(len(full_df))
    for m in final_models:
        X_sub = full_df[m["variables"]].rename(columns=RENAME_FOR_PYSR)
        be += m["model"].predict(X_sub)
    return be


# =============================================================================
# EXPORT
# =============================================================================
def config_dict():
    """Serializable snapshot of the run configuration (for run_metadata.json)."""
    return {
        "max_train_z": MAX_TRAIN_Z,
        "n_splits": N_SPLITS,
        "s": S,
        "n_t": N_T,
        "max_iter": MAX_ITER,
        "theta": THETA,
        "niterations": NITERATIONS,
        "ncycles_per_iteration": NCYCLES_PER_ITER,
        "maxdepth": MAXDEPTH,
        "parsimony": PARSIMONY,
        "random_state": RANDOM_STATE_PYSR,
        "binary_ops": BINARY_OPS,
        "unary_ops": UNARY_OPS,
        "anchor_z0": ANCHOR_Z0,
        "anchor_weight": ANCHOR_WEIGHT,
    }


def export_results(full_df, BE_MISR, BE_PyMISR, equation_str, equation_latex, mae):
    """Write the results CSV and the run metadata JSON."""
    RESULTS_DIR.mkdir(exist_ok=True)

    df_results = pd.DataFrame(
        {
            "Z": full_df["Z"],
            "N": full_df["N"],
            "BE_exp": full_df["BE"],
            "BE_misr": BE_MISR,
            "BE_pymisr": BE_PyMISR,
            "misr_residuals": BE_MISR - full_df["BE"],
            "pymisr_residuals": BE_PyMISR - full_df["BE"],
        }
    )
    df_results.to_csv(CSV_PATH, index=False)
    print(f"[OK] CSV results exported to: {CSV_PATH}")

    metadata = {
        "version": VERSION,
        "equation_str": equation_str,
        "equation_latex": equation_latex,
        "mae": mae,
        "config": config_dict(),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[OK] Run metadata exported to: {META_PATH}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    # 1. Data
    dataset = load_dataset()
    full_dataset = engineer_features(dataset)  # full set for final comparison
    train = full_dataset[full_dataset["Z"] <= MAX_TRAIN_Z].reset_index(drop=True)
    print(
        f"\n[FILTER] Training restricted to Z <= {MAX_TRAIN_Z}: "
        f"{len(train)} nuclei (out of {len(full_dataset)} total)"
    )
    print("\nDataset with new variables (A, BEpA, I, P):")
    print(train[["Z", "N", "A", "BE", "BEpA", "I", "nn", "np", "P"]].head())

    # 2. Features / target / weights
    X = train[FEATURES]
    y = train["BE"]
    sigma = (
        train["bindingEnergyUncertainty"]
        if "bindingEnergyUncertainty" in train.columns
        else 0.0
    )
    weights = 1.0 / (1.0 + sigma)

    # 3. Symbolic regression
    final_models = run_symbolic_regression(X, y, weights)
    if not final_models:
        print("\n[!] No model discovered. Exiting.")
        return

    # 4. Finalize + compare with MISR
    equation_str, equation_latex = finalize_equation(final_models)

    print("\n==========================================")
    print(" COMPARISON WITH THEORETICAL MISR EQUATION")
    print("==========================================")
    BE_MISR = compute_misr(full_dataset)
    BE_PyMISR = compute_pymisr(full_dataset, final_models)
    mae = {
        "pymisr": float(np.mean(np.abs(full_dataset["BE"] - BE_PyMISR))),
        "misr": float(np.mean(np.abs(full_dataset["BE"] - BE_MISR))),
    }
    print(f"PyMISR Model MAE: {mae['pymisr']:.4f}")
    print(f"MISR Equation MAE:                 {mae['misr']:.4f}")

    # 5. Export
    export_results(full_dataset, BE_MISR, BE_PyMISR, equation_str, equation_latex, mae)


if __name__ == "__main__":
    main()
