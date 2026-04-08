"""
Model evaluation and visualization.

Computes quantitative metrics (RMSE, MAE, relative L2 error) and generates
publication-quality plots of pressure profiles with shock highlighting.
"""

import json
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from nozzle_geometry import nozzle_area_np

logger = logging.getLogger(__name__)

# Plot styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 250,
    'savefig.bbox': 'tight',
})


def compute_metrics(P_true, P_pred):
    """Compute RMSE, MAE, and relative L2 error."""
    diff = P_true - P_pred
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(P_true) + 1e-12)
    return {'rmse': float(rmse), 'mae': float(mae), 'rel_l2': float(rel_l2)}


def validate_model(model, test_df, norm_config, device='cpu', save_dir='outputs'):
    """
    Full evaluation: compute metrics per P_back condition and generate plots.

    Generates:
        - pressure_profiles.png: predicted vs true for each test P_back
        - metrics.json: quantitative error metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    max_pb = norm_config['max_p_back']
    unique_pbs = sorted(test_df['P_back_raw'].unique())

    # Dense evaluation grid
    x_dense = np.linspace(0.125, 1.0, 200)
    A_dense = nozzle_area_np(x_dense)

    all_metrics = {}
    n_plots = len(unique_pbs)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5.5), squeeze=False)
    axes = axes[0]

    colors_true = '#2E86AB'
    colors_pred = '#E8430C'

    for i, pb in enumerate(unique_pbs):
        ax = axes[i]
        subset = test_df[test_df['P_back_raw'] == pb].sort_values('x')
        x_data = subset['x'].values
        P_true = subset['P'].values

        # Predict on dense grid
        pb_scaled = pb / max_pb
        feat = np.column_stack([x_dense, A_dense, np.full_like(x_dense, pb_scaled)])
        feat_t = torch.tensor(feat, dtype=torch.float32).to(device)
        with torch.no_grad():
            P_pred_dense = model(feat_t).cpu().numpy().flatten()

        # Predict at data points for metrics
        feat_data = np.column_stack([
            x_data, nozzle_area_np(x_data), np.full_like(x_data, pb_scaled)])
        feat_data_t = torch.tensor(feat_data, dtype=torch.float32).to(device)
        with torch.no_grad():
            P_pred_pts = model(feat_data_t).cpu().numpy().flatten()

        metrics = compute_metrics(P_true, P_pred_pts)
        all_metrics[f'P_back={pb}'] = metrics
        logger.info("P_back=%.0f | RMSE=%.4f MAE=%.4f relL2=%.4f",
                     pb, metrics['rmse'], metrics['mae'], metrics['rel_l2'])
        print(f"  P_back={pb:6.0f} | RMSE={metrics['rmse']:.5f} | "
              f"MAE={metrics['mae']:.5f} | Rel.L2={metrics['rel_l2']:.4f}")

        # Shock detection via pressure gradient
        dP = np.gradient(P_pred_dense, x_dense)
        shock_idx = np.argmax(np.abs(dP[len(dP)//4:]))  # search diverging half
        shock_idx += len(dP) // 4
        shock_x = x_dense[shock_idx]
        shock_grad = np.abs(dP[shock_idx])

        # Plot
        ax.scatter(x_data, P_true, s=50, c=colors_true, marker='o',
                   edgecolors='k', linewidths=0.5, zorder=5, label='Measured')
        ax.plot(x_dense, P_pred_dense, '-', color=colors_pred, lw=2.0,
                label='PINN Prediction')

        if shock_grad > 0.5:
            ax.axvline(shock_x, color='gray', ls='--', lw=1.0, alpha=0.7,
                       label=f'Shock ≈ x={shock_x:.2f}')
            ax.axvspan(shock_x - 0.05, shock_x + 0.05, alpha=0.08, color='red')

            # Get learned shock info
            shock_info = model.get_shock_info(pb_scaled)
            if shock_info['in_domain']:
                ax.axvline(shock_info['x_shock'], color='orange', ls=':',
                           lw=1.5, label=f'Learned shock x={shock_info["x_shock"]:.2f}')

        ax.set_xlabel('Axial Position x')
        ax.set_ylabel('Normalized Pressure P/P₀')
        ax.set_title(f'P_back = {pb:.0f} kPa\nRMSE = {metrics["rmse"]:.4f}')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.1, 1.05)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'pressure_profiles.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"  Validation plot saved to {fig_path}")

    # Save metrics
    met_path = os.path.join(save_dir, 'metrics.json')
    with open(met_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Metrics saved to {met_path}")

    return all_metrics
