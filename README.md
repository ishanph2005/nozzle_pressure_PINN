# Nozzle Flow PINN User Guide

## 1. Overview

This project implements a Physics-Informed Neural Network (PINN) designed to predict the 1D steady compressible pressure distribution through a converging-diverging nozzle.

The challenge in supersonic nozzle flow is the presence of **normal shocks**: abrupt, discontinuous jumps in pressure that occur when the flow transitions from supersonic back to subsonic. Standard Neural Networks struggle to represent these discontinuities and tend to smooth them out. This PINN uses a specialized **Shock-Aware Dual-Branch Architecture** combined with physics-inspired regularization to accurately locate and model these shockwaves, ensuring physical plausibility.

---

## 2. PINN Architecture

The model (`ShockAwarePINN`) employs a hybrid data-driven and shock-aware approach:

### Inputs and Outputs
* **Inputs (3):** Axial distance `x` (normalised wrt overall length), Nozzle Area `A(x)` (normalised wrt throat area, basically A/A*), and Normalized Back Pressure `P_back`.
* **Output (1):** Normalized static pressure `P / P_0`.

### Network Structure
Instead of a single smooth Multi-Layer Perceptron (MLP), the network is split into three learned components:
1. **Pre-Shock Branch:** A smooth MLP specialized in learning the isentropic expansion (decreasing pressure) of the supersonic flow.
2. **Post-Shock Branch:** A smooth MLP specialized in learning the subsonic pressure recovery.
3. **Shock Parameter Network:** A smaller network that maps the global boundary condition `P_back` to a specific **shock location ($x_{shock}$)** and **steepness factor ($k$)**.

These branches are dynamically blended using a learned sigmoid transition function:
$$ P(x) = (1 - S(x)) \cdot P_{pre}(x) + S(x) \cdot P_{post}(x) $$
Where $S(x) = \sigma(k \cdot (x - x_{shock}))$.

### Loss Components & Physics Integration
Because the experimental dataset is relatively coarse (only 8 spatial measurements per condition), full PDE residual enforcement via automatic differentiation is numerically unstable. Instead, the model relies on:
* **$L_{data}$ (MSE):** Mean Squared Error against the augmented experimental pressure data.
* **$L_{tv}$ (Total Variation):** An $L_1$ penalty on the spatial derivative of the individual branches, enforcing smoothness within the constituent branches while still allowing the blended output to be extremely sharp.
* **$L_{boundary}$:** A soft constraint at the nozzle exit ($x=1.0$) encouraging monotonicity with respect to increasing $P_{back}$.

---

## 3. Code Structure

* `main.py`: The unified Command-Line Interface (CLI) entry point.
* `data.py`: Handles data loading from CSV, group averaging, cubic spline (PCHIP) spatial augmentation (increasing 8 data points to 50 for spatial density), and dataset normalization.
* `model.py`: Contains the actual implementation of the `ShockAwarePINN` using PyTorch.
* `physics.py`: Implements the physics-inspired regularization (total variation and boundary penalties).
* `train.py`: The dual-phase training pipeline (Adam + Cosine Annealing followed by L-BFGS fine-tuning) with detailed metric logging.
* `validate.py`: Functions to evaluate the model on held-out test data, compute quantitative metrics (RMSE, MAE, Relative L2), and generate multi-plot visualizations.
* `infer.py`: The inference script that queries a trained model to predict pressure profiles for an arbitrary $P_{back}$ and actively searches for the location of normal shocks via spatial gradient analysis ($dP/dx$).
* `nozzle_geometry.py`: The single source of truth for the nozzle's mathematical area profile $A(x)$.

---

## 4. How to Use

### Setup

Ensure you are within the root directory (`Physics-Informed-Neural-Networks`).

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Required packages: `torch`, `numpy`, `pandas`, `matplotlib`, `scipy`*.

### Training

To train the model from scratch, use the `train` command. By default, it will train for 3000 epochs using the Adam optimizer followed by L-BFGS fine-tuning.

```bash
python src/main.py train --data data/cleaned_nozzle_dataset.csv --epochs 3000 --lr 1e-3
```

*(Optional flags: `--batch-size`, `--no-lbfgs`, `--lbfgs-iter`, `--output-dir`)*

### Inference

Once trained, you can predict the pressure distribution and find the normal shock location for any arbitrary back pressure (measured in kPa). Use the `--plot` flag to generate a visual profile.

```bash
python src/main.py predict --p_back 300.0 --plot
```

### Evaluation

To evaluate the model's performance on the held-out validation sets (by default, $P_{back} = 200$ and $500$ kPa) and generate generalized error metrics:

```bash
python src/main.py evaluate --data data/cleaned_nozzle_dataset.csv
```

---

## 5. Outputs

Outputs are saved in the `outputs/` directory explicitly passed via `--output-dir`.

### During Training:
* `training.log`: A complete text log of training progress, losses, and learning rates.
* `model_best.pth` & `model.pth`: PyTorch state dictionaries (best Adam checkpoint and final L-BFGS checkpoint).
* `loss_history.png`: A dual-axis plot visualizing the convergence of the Adam optimizer across data MSE and physical regularizations.
* `norm_config.json`: Saves normalization scaling constants needed for accurate inference.

### During Evaluation:
* `pressure_profiles.png`: A high-quality comparative scatter vs. line plot showing predictions vs. true values for the test datasets, highlighting the location of detected shocks in red and yellow.
* `metrics.json`: Contains raw RMSE, MAE, and Relative L2 scores per test pressure.

### During Inference:
* `prediction_Pback_{X}.png`: The pressure profile prediction with visual bounds and shock location labels.
* `prediction_Pback_{X}.csv`: Raw comma-separated data of $x, A(x)$ and $P(x)$ at 200 dense interpolated query points across the nozzle.

---

## 6. Notes on Physics

* **Limitations:** The initial codebase attempted to construct PDE residuals mapping $dP/dx$ and $du/dx$ but provided a highly sparse dataset (8 stations). Pure PDE residuals are impossible to optimize effectively given this setup. This refactor opts for a data-driven approach guided by smoothness and spatial variation heuristics instead.
* **Shock Handling:** Shocks are not tracked via Eulerian sub-grids; they are localized analytically during inference. In `infer.py`, the network takes advantage of the near-vertical discontinuity outputted by the Dual-Branch Sigmoid combination. It searches the diverging section of the nozzle for the maximum absolute value of the spatial gradient ($|dP/dx|$).
