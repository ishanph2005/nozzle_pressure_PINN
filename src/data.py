"""
Data loading and preprocessing for the nozzle flow PINN.

Pipeline: CSV → group averaging → PCHIP interpolation → normalization → DataLoaders.
Saves normalization config to JSON for consistent inference.
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator
from torch.utils.data import Dataset, DataLoader

from nozzle_geometry import nozzle_area_np

logger = logging.getLogger(__name__)


class NozzleDataset(Dataset):
    """PyTorch dataset wrapping feature/target arrays."""

    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_and_preprocess(csv_path, n_interp=50):
    """
    Load CSV, average across experimental groups, and interpolate spatially.

    Returns the group-averaged raw data and a denser interpolated dataset.
    PCHIP interpolation preserves monotonicity and avoids Gibbs oscillation at shocks.
    """
    df = pd.read_csv(csv_path)

    # Average pressure across the 5 experimental groups
    avg_df = df.groupby(['P_back', 'x']).agg({
        'A(x)': 'first',
        'P': 'mean'
    }).reset_index()

    # Interpolate each P_back curve from 8 stations to n_interp points
    interp_rows = []
    x_lo, x_hi = avg_df['x'].min(), avg_df['x'].max()
    x_interp = np.linspace(x_lo, x_hi, n_interp)

    for p_back in sorted(avg_df['P_back'].unique()):
        subset = avg_df[avg_df['P_back'] == p_back].sort_values('x')
        x_pts = subset['x'].values
        P_pts = subset['P'].values

        P_dense = PchipInterpolator(x_pts, P_pts)(x_interp)
        A_dense = nozzle_area_np(x_interp)

        for i in range(len(x_interp)):
            interp_rows.append({
                'x': x_interp[i],
                'A(x)': A_dense[i],
                'P_back': p_back,
                'P': float(np.clip(P_dense[i], 0.01, None)),
            })

    interp_df = pd.DataFrame(interp_rows)
    logger.info("Loaded %d raw rows, averaged to %d, interpolated to %d",
                len(df), len(avg_df), len(interp_df))
    return avg_df, interp_df


def prepare_training_data(csv_path, test_p_backs=None, batch_size=64,
                          n_interp=50, save_dir=None):
    """
    Full data pipeline returning DataLoaders, dataframes, and normalization config.

    Parameters
    ----------
    csv_path : str        Path to the cleaned nozzle dataset CSV.
    test_p_backs : list   P_back values held out for testing (default: [200, 500]).
    batch_size : int      DataLoader batch size.
    n_interp : int        Number of interpolated x-stations per P_back curve.
    save_dir : str        Directory to save normalization config JSON.
    """
    if test_p_backs is None:
        test_p_backs = [200.0, 500.0]

    avg_df, interp_df = load_and_preprocess(csv_path, n_interp=n_interp)

    # Normalization constants
    max_p_back = float(interp_df['P_back'].max())
    if max_p_back == 0:
        max_p_back = 1.0

    norm_config = {'max_p_back': max_p_back}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cfg_path = os.path.join(save_dir, 'norm_config.json')
        with open(cfg_path, 'w') as f:
            json.dump(norm_config, f, indent=2)
        logger.info("Saved normalization config to %s", cfg_path)

    # Build feature columns: [x, A(x), P_back_scaled]
    processed = pd.DataFrame()
    processed['x'] = interp_df['x']
    processed['A'] = interp_df['A(x)']
    processed['P_back_scaled'] = interp_df['P_back'] / max_p_back
    processed['P_back_raw'] = interp_df['P_back']
    processed['P'] = interp_df['P']

    # Train/test split by P_back
    test_mask = processed['P_back_raw'].isin(test_p_backs)
    train_df = processed[~test_mask].reset_index(drop=True)
    test_df = processed[test_mask].reset_index(drop=True)

    logger.info("Train: %d samples | Test: %d samples (P_back=%s)",
                len(train_df), len(test_df), test_p_backs)
    print(f"[DATA] Train: {len(train_df)} samples | Test: {len(test_df)} samples")

    feat_cols = ['x', 'A', 'P_back_scaled']
    train_loader = DataLoader(
        NozzleDataset(train_df[feat_cols].values, train_df['P'].values),
        batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        NozzleDataset(test_df[feat_cols].values, test_df['P'].values),
        batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_df, test_df, norm_config
