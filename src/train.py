"""
Training pipeline for the shock-aware PINN.

Dual-phase optimization: Adam with cosine annealing LR, followed by
L-BFGS fine-tuning. Tracks individual loss components for diagnostics.
"""

import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from nozzle_geometry import nozzle_area_np
from physics import total_variation_penalty, boundary_loss

logger = logging.getLogger(__name__)


def _build_collocation_points(train_df, norm_config, n_per_pb=80):
    """Generate dense collocation points for regularization evaluation."""
    max_pb = norm_config['max_p_back']
    x_dense = np.linspace(0.125, 1.0, n_per_pb)
    A_dense = nozzle_area_np(x_dense)

    pb_values = sorted(train_df['P_back_raw'].unique())
    rows = []
    for pb in pb_values:
        pb_s = pb / max_pb
        for i in range(n_per_pb):
            rows.append([x_dense[i], A_dense[i], pb_s])

    return torch.tensor(rows, dtype=torch.float32)


def train_model(model, train_loader, norm_config, epochs=3000, lr=1e-3,
                device='cpu', use_lbfgs=True, lbfgs_iter=500,
                save_dir='outputs', train_df=None,
                lambda_tv=0.001, lambda_bnd=0.01):
    """
    Run the full training pipeline.

    Returns
    -------
    dict with keys: 'loss_total', 'loss_data', 'loss_tv', 'loss_bnd'
        Each is a list of per-epoch values.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.to(device).train()
    loss_fn = nn.MSELoss()

    # Collocation points for regularization (run on same device)
    if train_df is not None:
        colloc = _build_collocation_points(train_df, norm_config).to(device)
    else:
        colloc = None

    # Phase 1: Adam with cosine annealing LR
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {'loss_total': [], 'loss_data': [], 'loss_tv': [], 'loss_bnd': []}

    logger.info("Phase 1: Adam optimization (%d epochs, lr=%.1e)", epochs, lr)
    print(f"\n{'='*60}")
    print(f"  Phase 1: Adam  |  {epochs} epochs  |  lr={lr:.1e}")
    print(f"{'='*60}")

    t0 = time.time()
    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_data_loss = 0.0
        n_batches = 0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            P_pred = model(X_batch)
            L_data = loss_fn(P_pred, Y_batch)

            # Regularization (computed less frequently to save time)
            L_tv = torch.tensor(0.0, device=device)
            L_bnd = torch.tensor(0.0, device=device)
            if colloc is not None and epoch % 5 == 0:
                L_tv = lambda_tv * total_variation_penalty(model, colloc, device)
                L_bnd = lambda_bnd * boundary_loss(model, norm_config, device)

            loss = L_data + L_tv + L_bnd
            loss.backward()
            optimizer.step()

            epoch_data_loss += L_data.item()
            n_batches += 1

        scheduler.step()
        avg_data = epoch_data_loss / max(n_batches, 1)
        history['loss_total'].append(loss.item())
        history['loss_data'].append(avg_data)
        history['loss_tv'].append(L_tv.item())
        history['loss_bnd'].append(L_bnd.item())

        # Save best model
        if avg_data < best_loss:
            best_loss = avg_data
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.pth'))

        if epoch % 200 == 0 or epoch == epochs - 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:5d}/{epochs} | Data: {avg_data:.6e} | "
                  f"TV: {L_tv.item():.4e} | Bnd: {L_bnd.item():.4e} | "
                  f"LR: {lr_now:.2e}")
            logger.info("Epoch %5d | data=%.4e tv=%.4e bnd=%.4e lr=%.2e",
                        epoch, avg_data, L_tv.item(), L_bnd.item(), lr_now)

    elapsed_adam = time.time() - t0
    print(f"\n  Adam completed in {elapsed_adam:.1f}s  |  Best loss: {best_loss:.6e}")

    # Phase 2: L-BFGS fine-tuning on top of best Adam checkpoint
    if use_lbfgs:
        # Restore the best model from Adam before L-BFGS
        best_path = os.path.join(save_dir, 'model_best.pth')
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        model.train()

        logger.info("Phase 2: L-BFGS fine-tuning (%d iterations)", lbfgs_iter)
        print(f"\n{'='*60}")
        print(f"  Phase 2: L-BFGS  |  {lbfgs_iter} max iterations")
        print(f"{'='*60}")

        # Full-batch for L-BFGS
        X_all = torch.cat([b[0] for b in train_loader]).to(device)
        Y_all = torch.cat([b[1] for b in train_loader]).to(device)

        lbfgs = optim.LBFGS(model.parameters(), lr=0.01, max_iter=lbfgs_iter,
                            tolerance_grad=1e-7, tolerance_change=1e-9,
                            history_size=50, line_search_fn='strong_wolfe')
        iter_count = [0]
        best_lbfgs = [float('inf')]

        def closure():
            lbfgs.zero_grad()
            P_pred = model(X_all)
            loss = loss_fn(P_pred, Y_all)
            loss.backward()
            if loss.item() < best_lbfgs[0]:
                best_lbfgs[0] = loss.item()
            if iter_count[0] % 50 == 0:
                print(f"    L-BFGS iter {iter_count[0]:4d} | Loss: {loss.item():.6e}")
            iter_count[0] += 1
            return loss

        t1 = time.time()
        lbfgs.step(closure)
        elapsed_lbfgs = time.time() - t1
        print(f"  L-BFGS completed in {elapsed_lbfgs:.1f}s ({iter_count[0]} iters)")

        # If L-BFGS made things worse, revert to best Adam model
        final_loss = loss_fn(model(X_all), Y_all).item()
        if final_loss > best_loss * 2:
            print(f"  L-BFGS loss ({final_loss:.4e}) > Adam best ({best_loss:.4e}), reverting.")
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))

    # Save loss history
    hist_path = os.path.join(save_dir, 'loss_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f)
    logger.info("Training complete. Model and history saved to %s", save_dir)

    # Generate loss convergence plot
    _plot_loss(history, save_dir)

    return history


def _plot_loss(history, save_dir):
    """Generate publication-quality loss convergence plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(len(history['loss_data']))
    ax1.semilogy(epochs, history['loss_data'], 'b-', lw=1.2, label='Data MSE')
    ax1.semilogy(epochs, history['loss_total'], 'k--', lw=0.8, alpha=0.5, label='Total')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(epochs, history['loss_tv'], 'r-', lw=1.0, label='TV Reg.')
    ax2.semilogy(epochs, history['loss_bnd'], 'g-', lw=1.0, label='Boundary')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Regularization Loss')
    ax2.set_title('Regularization Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'loss_history.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Loss plot saved to {path}")
