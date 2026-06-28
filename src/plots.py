"""
PyMISR — Plotting stage.
========================
Reads the model outputs (results/MISRComparison_results.csv and
results/run_metadata.json) and produces every figure. Fully decoupled from the
model: run `model.py` once, then re-run this as many times as you like WITHOUT
retraining (the PySR fit takes minutes).

Outputs (results/):
  - MISRComparison.png                 (NEW: combined 2x2 figure)
  - MISRComparison_BE_per_nucleon.png
  - MISRComparison_absolute_error.png
  - MISRComparison_residuals_N.png
  - MISRComparison_residuals_Z.png
  - MISRComparison_config.png          (configuration table)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
CSV_PATH    = RESULTS_DIR / "MISRComparison_results.csv"
META_PATH   = RESULTS_DIR / "run_metadata.json"

# Colors / markers (shared by every panel)
C_EXP, C_PYMISR, C_MISR = 'gray', 'blue', 'red'
TRAIN_Z_DEFAULT = 50

PLT_STYLE = {
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


# =============================================================================
# DATA
# =============================================================================
def load_results():
    """Load the results CSV and the run metadata JSON.

    Adds the derived A = Z + N column and the per-nucleon residual columns the
    panels need. Returns (df, meta).
    """
    df = pd.read_csv(CSV_PATH)
    df['A'] = df['Z'] + df['N']

    meta = {}
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    return df, meta


def maes(df, meta):
    """MAE for each model (from metadata if present, else recomputed from CSV)."""
    if meta.get("mae"):
        return meta["mae"]["pymisr"], meta["mae"]["misr"]
    return (df['pymisr_residuals'].abs().mean(), df['misr_residuals'].abs().mean())


# =============================================================================
# PANELS (each draws on a provided Axes -> reusable standalone or in a grid)
# =============================================================================
def plot_be_per_nucleon(ax, df):
    """Binding energy per nucleon (BE/A) vs neutron number N."""
    ax.scatter(df['N'], df['BE_exp'] / df['A'], alpha=0.6, label=r'Experimental',
               color=C_EXP, s=5)
    ax.scatter(df['N'], df['BE_pymisr'] / df['A'], alpha=0.8, label=r'PyMISR',
               color=C_PYMISR, s=5)
    ax.scatter(df['N'], df['BE_misr'] / df['A'], alpha=0.8, label=r'MISR1',
               color=C_MISR, marker='x', s=5)
    ax.set_xlabel(r'Neutron Number $N$')
    ax.set_ylabel(r'$BE/A$ (MeV)')
    ax.set_title(r'Binding Energy per Nucleon vs $N$')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_absolute_error(ax, df):
    """Absolute error per nucleon |delta BE / A| vs N."""
    err_pymisr = df['pymisr_residuals'].abs() / df['A']
    err_misr = df['misr_residuals'].abs() / df['A']
    ax.scatter(df['N'], err_pymisr, alpha=0.8,
               label=rf'PyMISR (MAE/A: {err_pymisr.mean():.4f})', color=C_PYMISR, s=5)
    ax.scatter(df['N'], err_misr, alpha=0.8,
               label=rf'MISR (MAE/A: {err_misr.mean():.4f})', color=C_MISR, marker='x', s=5)
    ax.set_xlabel(r'Neutron Number $N$')
    ax.set_ylabel(r'$|\Delta BE/A|$ (MeV)')
    ax.set_title(r'Absolute Error per Nucleon vs $N$')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_residuals_N(ax, df, mae_pymisr, mae_misr):
    """Residuals delta BE vs N."""
    ax.scatter(df['N'], df['pymisr_residuals'], alpha=0.8,
               label=rf'PyMISR (MAE: {mae_pymisr:.2f})', color=C_PYMISR, s=5)
    ax.scatter(df['N'], df['misr_residuals'], alpha=0.8,
               label=rf'MISR (MAE: {mae_misr:.2f})', color=C_MISR, marker='x', s=5)
    ax.axhline(0, color='k', linestyle='--', alpha=0.7)
    ax.set_xlim(10, 90)
    ax.set_xlabel(r'Neutron Number $N$')
    ax.set_ylabel(r'$\Delta BE$ (MeV)')
    ax.set_title(r'Residuals vs $N$')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_residuals_Z(ax, df, mae_pymisr, mae_misr, train_z=TRAIN_Z_DEFAULT):
    """Residuals per nucleon delta BE / A vs Z."""
    res_pymisr = df['pymisr_residuals'] / df['A']
    res_misr = df['misr_residuals'] / df['A']
    ax.scatter(df['Z'], res_pymisr, alpha=0.8,
               label=rf'PyMISR (MAE: {mae_pymisr:.2f})', color=C_PYMISR, s=5)
    ax.scatter(df['Z'], res_misr, alpha=0.8,
               label=rf'MISR (MAE: {mae_misr:.2f})', color=C_MISR, marker='x', s=5)
    ax.axhline(0, color='k', linestyle='--', alpha=0.7)
    ax.axvline(train_z, color='gray', linestyle=':', linewidth=1.5, alpha=0.8,
               label=rf'$Z = {train_z}$ (training limit)')
    ax.set_xlim(10, 120)
    ax.set_xlabel(r'Atomic Number $Z$')
    ax.set_ylabel(r'$\Delta BE / A$ (MeV)')
    ax.set_title(r'Residuals per Nucleon vs $Z$')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_config_table(meta):
    """Render the configuration parameters table to its own PNG."""
    cfg = meta.get("config", {})
    rows = [
        [r"Version",                          meta.get("version", "?")],
        [r"Max training $Z$",                 rf"$Z \leq {cfg.get('max_train_z', '?')}$"],
        [r"Physical constraint",              r"soft $BE(Z{=}0)=0$"],
        [r"anchor_weight",                    str(cfg.get("anchor_weight", "?"))],
        [r"n_splits",                         str(cfg.get("n_splits", "?"))],
        [r"s (selected vars.)",               str(cfg.get("s", "?"))],
        [r"n_t (maxsize)",                    str(cfg.get("n_t", "?"))],
        [r"max_iter",                         str(cfg.get("max_iter", "?"))],
        [r"$\theta$ (min improvement ratio)", str(cfg.get("theta", "?"))],
        [r"niterations",                      str(cfg.get("niterations", "?"))],
        [r"ncycles_per_iteration",            str(cfg.get("ncycles_per_iteration", "?"))],
        [r"maxdepth",                         str(cfg.get("maxdepth", "?"))],
        [r"parsimony",                        str(cfg.get("parsimony", "?"))],
        [r"random_state",                     str(cfg.get("random_state", "?"))],
        [r"Binary operators",                 ", ".join(cfg.get("binary_ops", []))],
        [r"Unary operators",                  ", ".join(cfg.get("unary_ops", []))],
    ]

    plt.figure(figsize=(8, 6))
    ax = plt.subplot(1, 1, 1)
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=[r"Parameter", r"Value"],
                     cellLoc='left', loc='center', colWidths=[0.55, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for col in range(2):
        table[(0, col)].set_facecolor('#2c3e50')
        table[(0, col)].set_text_props(color='white', fontweight='bold')
    for row in range(1, len(rows) + 1):
        fc = '#eaf0f6' if row % 2 == 0 else 'white'
        for col in range(2):
            table[(row, col)].set_facecolor(fc)
    ax.set_title(r'Configuration Parameters', fontsize=12, pad=12)
    plt.tight_layout()
    out = RESULTS_DIR / "MISRComparison_config.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Config table exported to: {out}")


# =============================================================================
# FIGURE BUILDERS
# =============================================================================
def _save_single(plot_fn, name, *args):
    """Helper: render one panel on a fresh figure and save it."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_fn(ax, *args)
    fig.tight_layout()
    out = RESULTS_DIR / name
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Plot exported to: {out}")


def make_combined_figure(df, mae_pymisr, mae_misr, train_z=TRAIN_Z_DEFAULT, equation_latex=None):
    """Combined 2x2 figure -> MISRComparison.png (the README cover figure)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_be_per_nucleon(axes[0, 0], df)
    plot_absolute_error(axes[0, 1], df)
    plot_residuals_N(axes[1, 0], df, mae_pymisr, mae_misr)
    plot_residuals_Z(axes[1, 1], df, mae_pymisr, mae_misr, train_z)
    fig.suptitle(r'PyMISR vs MISR — Binding Energy Comparison', fontsize=15, y=0.99)
    # Show the discovered equation (from run_metadata.json) under the title.
    if equation_latex:
        try:
            fig.text(0.5, 0.945, rf'$BE = {equation_latex}$', ha='center', va='top',
                     fontsize=13, color='darkblue')
        except Exception:
            pass
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = RESULTS_DIR / "MISRComparison.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Combined figure exported to: {out}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    plt.rcParams.update(PLT_STYLE)
    RESULTS_DIR.mkdir(exist_ok=True)

    df, meta = load_results()
    mae_pymisr, mae_misr = maes(df, meta)
    train_z = meta.get("config", {}).get("max_train_z", TRAIN_Z_DEFAULT)

    print("Generating figures...")
    _save_single(plot_be_per_nucleon, "MISRComparison_BE_per_nucleon.png", df)
    _save_single(plot_absolute_error, "MISRComparison_absolute_error.png", df)
    _save_single(plot_residuals_N, "MISRComparison_residuals_N.png", df, mae_pymisr, mae_misr)
    _save_single(plot_residuals_Z, "MISRComparison_residuals_Z.png", df, mae_pymisr, mae_misr, train_z)
    plot_config_table(meta)
    make_combined_figure(df, mae_pymisr, mae_misr, train_z, meta.get("equation_latex"))
    print("Done.")


if __name__ == "__main__":
    main()
