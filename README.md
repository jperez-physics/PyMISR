# PyMISR

**Iterative symbolic regression for nuclear binding energy discovery.**

This project re-implements the MISR (*Multi-objective Iterated Symbolic Regression*) algorithm from the paper [*"Discovering nuclear models from symbolic machine learning"*](https://www.nature.com/articles/s41567-023-02134-2) (Nature Physics, 2023). It uses [PySR](https://github.com/MilesCranmer/PySR) as the symbolic regression engine to automatically discover interpretable analytical expressions for nuclear binding energy from experimental data.

**Original repository:** [munozariasjm/nuclear-misr](https://github.com/munozariasjm/nuclear-misr)

---

## Result

![PyMISR vs MISR Comparison](results/MISRComparison.png)

*Comparison between experimental binding energy, the theoretical MISR model, and the PyMISR-discovered model across all known nuclei.*

---

## Equations

**MISR (from paper):**

$$
BE = \eta_0 \, Z \left(1 + \frac{1}{N} - \frac{a\,N}{Z^2}\right) \left[I\!\left(b - \frac{A^{1/3}\,N}{Z}\right) + c\right]
$$

with parameters: $a = 1.10$, $b = 32.43$, $c = 16.70$, $\eta_0 = 1.0$

---

## Project Structure

```
PyMISR/
├── datasets/
│   └── be_exp.csv              # Experimental binding energies (~4753 nuclei)
├── notebooks/
│   └── bindingEnergy.ipynb     # Interactive exploration notebook (v0.2)
├── results/
│   ├── MISRComparison.png              # Main comparison figure (combined 2x2)
│   ├── MISRComparison_BE_per_nucleon.png
│   ├── MISRComparison_absolute_error.png
│   ├── MISRComparison_residuals_N.png
│   ├── MISRComparison_residuals_Z.png
│   ├── MISRComparison_config.png       # Configuration parameters table
│   ├── MISRComparison_results.csv      # Z, N, BE_exp, BE_misr, BE_pymisr, residuals
│   └── run_metadata.json              # Config + discovered equation + MAE (model -> plots)
├── src/
│   ├── model.py                # Stage 1: data + symbolic regression -> CSV + metadata (v0.4)
│   └── plots.py                # Stage 2: reads CSV + metadata -> figures
└── README.md
```

---

## Requirements

**Python 3.10+**

```
pandas
numpy
scikit-learn
pysr
sympy
matplotlib
```

**Julia 1.10+** (installed automatically by PySR via `juliacall`)

---

## Usage

The pipeline is split into two decoupled stages:

```bash
cd PyMISR
python src/model.py    # Stage 1: train + export CSV and run_metadata.json (slow, runs Julia)
python src/plots.py    # Stage 2: read those files and render all figures (fast)
```

`plots.py` only reads `results/MISRComparison_results.csv` and `results/run_metadata.json`,
so figures can be regenerated/tweaked any number of times **without** retraining.

**Stage 1 — `model.py`** will:
1. Load and preprocess the experimental dataset
2. Filter nuclei to Z ≤ 50 for training
3. Run iterative symbolic regression with 5-fold cross-validation
4. (Optional) Inject soft physical anchors at Z=0 (BE=0) — `anchor_z0`, **off in v0.5**
5. Simplify the discovered equation with SymPy
6. Compare against the MISR1 model on all nuclei
7. Export `MISRComparison_results.csv` and `run_metadata.json` to `results/`

**Stage 2 — `plots.py`** reads those outputs and exports the combined `MISRComparison.png`
plus the individual panels and the configuration table.

### Physical constraint: BE(Z=0) = 0  *(optional, `anchor_z0`)*

Without protons there is no nucleus, so the binding energy should vanish at $Z=0$. PyMISR can
enforce this as a **soft constraint** (`anchor_z0=True`): synthetic anchor points with $Z=0$,
$BE=0$ and a high weight (`anchor_weight`) are added to each fold's PySR training set, leaving
the equation form completely free (no structural template, so no `f(...)/Z` loophole). In v0.4
the search converged to a form with $Z$ as a global factor, satisfying $BE(Z=0)=0$ exactly.

> **v0.5 disables this** (`anchor_z0=False`). The anchors penalize MISR's divergent terms
> ($1/N$, $N/Z^2$), so they conflict with the goal of approaching MISR — which itself diverges
> at $Z=0$. Re-enable `anchor_z0` if you want the physical $BE(Z=0)=0$ behavior back.

---

## Results

Performance across all nuclei (MAE in MeV):

| Region | MISR (theoretical) | PyMISR (v0.5) |
|---|---|---|
| Z ≤ 50 (training) | 5.31 | **2.28** |
| Z > 50 (extrapolation) | 110.31 | **5.16** |
| **Global** | 66.65 | **3.97** |

The v0.5 configuration (`s=5`, larger search budget, no `sqrt`, anchors off) makes PyMISR
**outperform MISR in every region by a wide margin**, including the training region (2.28 vs
5.31). The discovered form is `(a·N² + b·N·Z − c·(Z+…)(d·Z² − …) + …)/(A + …)`, whose
`(Z², N·Z, N²)/A` structure echoes the Coulomb/asymmetry terms of the semi-empirical mass
formula. Trade-off: with the anchors disabled, $BE(Z=0)=0$ is **no longer guaranteed**
(consistent with MISR, which diverges at $Z=0$).

> **⚠️ On the reported errors.** These MAE are computed from the **full-precision** PySR model
> (`model.predict`). The discovered equation shown in `results/run_metadata.json` is rounded to
> 2 decimals for readability and **does not reproduce these MAE on its own** — rounding small
> coefficients (e.g. the `0.01·Z²` term, with $Z$ up to ~116) shifts the result by hundreds of
> MeV. Treat the printed equation as the discovered functional *form*, not a plug-in formula; the
> numbers above reflect the actual fitted model.

---

## References

1. Munoz, J. M., Udrescu, S. M., & Garcia Ruiz, R. F. (2023). *Discovering nuclear models from symbolic machine learning*. Nature Physics.
2. Munoz, J. M., Udrescu, S. M., & Garcia Ruiz, R. F. (2023). *Supplementary Information for: Discovering nuclear models from symbolic machine learning*. Nature Physics.
3. Pérez, J. & Vargas, K. (2026). PyMISR: Iterative Model Construction Algorithm using Symbolic Regression.
