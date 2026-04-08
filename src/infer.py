"""
Inference module for the shock-aware nozzle PINN.

Predicts pressure distribution for a given back pressure, detects shock
location, generates plots, and saves results to CSV.
"""

import json
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import ShockAwarePINN
from nozzle_geometry import nozzle_area_np

logger = logging.getLogger(__name__)


def predict_pressure(model, p_back, norm_config, device='cpu', n_points=200):
    """
    Predict the pressure distribution along the nozzle for a given P_back.

    Returns arrays of (x, A, P_pred) and shock detection results.
    """
    max_pb = norm_config['max_p_back']
    pb_scaled = p_back / max_pb

    x = np.linspace(0.125, 1.0, n_points)
    A = nozzle_area_np(x)
    feat = np.column_stack([x, A, np.full_like(x, pb_scaled)])
    feat_t = torch.tensor(feat, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        P_pred = model(feat_t).cpu().numpy().flatten()

    # Shock detection via second derivative of pressure
    dP = np.gradient(P_pred, x)
    d2P = np.gradient(dP, x)

    # Search the diverging section (x > throat at 0.25) for max |dP/dx|
    div_mask = x > 0.3
    if np.any(div_mask):
        dP_div = np.abs(dP[div_mask])
        x_div = x[div_mask]
        peak_idx = np.argmax(dP_div)
        shock_x = x_div[peak_idx]
        shock_strength = dP_div[peak_idx]
    else:
        shock_x, shock_strength = None, 0.0

    # Also get learned shock info from the model
    learned_shock = model.get_shock_info(pb_scaled)

    shock_info = {
        'gradient_shock_x': float(shock_x) if shock_x else None,
        'gradient_strength': float(shock_strength),
        'detected': shock_strength > 0.3,
        'learned_shock_x': learned_shock['x_shock'],
        'learned_steepness': learned_shock['steepness'],
        'learned_in_domain': learned_shock['in_domain'],
    }

    return x, A, P_pred, shock_info


def run_prediction(args):
    """CLI entry point for prediction."""
    output_dir = getattr(args, 'output_dir',
                         os.path.join(os.path.dirname(__file__), 'outputs'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load normalization config
    cfg_path = os.path.join(output_dir, 'norm_config.json')
    if not os.path.exists(cfg_path):
        print(f"Error: {cfg_path} not found. Run training first.")
        return
    with open(cfg_path) as f:
        norm_config = json.load(f)

    # Load model
    model = ShockAwarePINN().to(device)
    model_path = os.path.join(output_dir, 'model.pth')
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run training first.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    p_back = args.p_back
    x, A, P_pred, shock_info = predict_pressure(model, p_back, norm_config, device)

    # Print tabular results
    print(f"\n{'='*62}")
    print(f"  Pressure Profile  |  P_back = {p_back:.1f} kPa")
    print(f"{'='*62}")
    print(f"  {'x':>8}  |  {'A(x)':>8}  |  {'P/P₀':>10}")
    print(f"  {'-'*8}  |  {'-'*8}  |  {'-'*10}")

    # Print at 20 evenly-spaced stations
    indices = np.linspace(0, len(x) - 1, 20, dtype=int)
    for idx in indices:
        print(f"  {x[idx]:8.4f}  |  {A[idx]:8.3f}  |  {P_pred[idx]:10.5f}")

    # Print shock detection results
    print(f"\n{'─'*62}")
    print("  Shock Detection Results:")
    if shock_info['detected']:
        print(f"    Gradient-based:  x ≈ {shock_info['gradient_shock_x']:.3f}  "
              f"(|dP/dx| = {shock_info['gradient_strength']:.2f})")
    else:
        print(f"    Gradient-based:  No strong shock detected "
              f"(max |dP/dx| = {shock_info['gradient_strength']:.2f})")

    if shock_info['learned_in_domain']:
        print(f"    Network-learned: x ≈ {shock_info['learned_shock_x']:.3f}  "
              f"(steepness k = {shock_info['learned_steepness']:.1f})")
    else:
        print(f"    Network-learned: Shock outside domain "
              f"(x = {shock_info['learned_shock_x']:.3f})")
    print(f"{'='*62}")

    # Save to CSV
    csv_path = os.path.join(output_dir, f'prediction_Pback_{int(p_back)}.csv')
    np.savetxt(csv_path, np.column_stack([x, A, P_pred]),
               header='x,A(x),P_pred', delimiter=',', comments='',
               fmt='%.6f')
    print(f"  Results saved to {csv_path}")

    # Generate plot if requested
    if getattr(args, 'plot', False):
        _plot_prediction(x, P_pred, p_back, shock_info, output_dir)


def _plot_prediction(x, P_pred, p_back, shock_info, save_dir):
    """Generate a publication-quality prediction plot."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(x, P_pred, '-', color='#E8430C', lw=2.5, label=f'PINN (P_back={p_back:.0f})')
    ax.fill_between(x, 0, P_pred, alpha=0.08, color='#E8430C')

    if shock_info['detected']:
        sx = shock_info['gradient_shock_x']
        ax.axvline(sx, color='#555', ls='--', lw=1.2)
        ax.annotate(f'Shock ≈ x={sx:.2f}',
                    xy=(sx, ax.get_ylim()[1] * 0.9),
                    xytext=(sx + 0.08, ax.get_ylim()[1] * 0.95),
                    fontsize=10, color='#555',
                    arrowprops=dict(arrowstyle='->', color='#555'))

    if shock_info['learned_in_domain']:
        lx = shock_info['learned_shock_x']
        ax.axvline(lx, color='#E89005', ls=':', lw=1.8,
                   label=f'Learned shock x={lx:.2f}')

    ax.set_xlabel('Axial Position x')
    ax.set_ylabel('Normalized Pressure P/P₀')
    ax.set_title(f'Predicted Nozzle Pressure Distribution — P_back = {p_back:.0f} kPa')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 1.05)
    ax.set_ylim(bottom=0)

    path = os.path.join(save_dir, f'prediction_Pback_{int(p_back)}.png')
    plt.savefig(path, dpi=250)
    plt.close()
    print(f"  Plot saved to {path}")
