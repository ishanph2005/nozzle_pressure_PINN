"""
Nozzle Flow PINN — Unified CLI

A shock-aware Physics-Informed Neural Network for predicting pressure
distributions in a converging-diverging nozzle with normal shock capability.

Usage:
    python main.py train  --epochs 3000 --data data/cleaned_nozzle_dataset.csv
    python main.py predict --p_back 300 --plot
    python main.py evaluate --data data/cleaned_nozzle_dataset.csv
"""

import argparse
import logging
import os
import sys

import torch


def setup_logging(output_dir):
    """Configure logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler(sys.stdout),
        ]
    )


def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(src_dir)
    default_data = os.path.join(base_dir, 'data', 'cleaned_nozzle_dataset.csv')
    default_output = os.path.join(base_dir, 'outputs')

    parser = argparse.ArgumentParser(
        description='Shock-Aware PINN for Nozzle Flow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest='command', help='Available commands')

    # --- Train ---
    p_train = sub.add_parser('train', help='Train the PINN model')
    p_train.add_argument('--data', type=str, default=default_data,
                         help='Path to nozzle dataset CSV')
    p_train.add_argument('--epochs', type=int, default=3000,
                         help='Number of Adam epochs (default: 3000)')
    p_train.add_argument('--lr', type=float, default=1e-3,
                         help='Initial learning rate (default: 1e-3)')
    p_train.add_argument('--batch-size', type=int, default=64,
                         help='Batch size (default: 64)')
    p_train.add_argument('--no-lbfgs', action='store_true',
                         help='Skip L-BFGS fine-tuning phase')
    p_train.add_argument('--lbfgs-iter', type=int, default=500,
                         help='Max L-BFGS iterations (default: 500)')
    p_train.add_argument('--output-dir', type=str, default=default_output)

    # --- Predict ---
    p_pred = sub.add_parser('predict', help='Predict pressure for a given P_back')
    p_pred.add_argument('--p_back', type=float, required=True,
                        help='Back pressure value in kPa (e.g. 300)')
    p_pred.add_argument('--plot', action='store_true',
                        help='Generate a plot of the prediction')
    p_pred.add_argument('--output-dir', type=str, default=default_output)

    # --- Evaluate ---
    p_eval = sub.add_parser('evaluate', help='Evaluate model on held-out test data')
    p_eval.add_argument('--data', type=str, default=default_data,
                        help='Path to nozzle dataset CSV')
    p_eval.add_argument('--output-dir', type=str, default=default_output)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    setup_logging(getattr(args, 'output_dir', default_output))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info("Device: %s", device)
    print(f"\n  Device: {device}")

    if args.command == 'train':
        from data import prepare_training_data
        from model import ShockAwarePINN
        from train import train_model
        from validate import validate_model

        print("\n" + "="*60)
        print("  NOZZLE FLOW PINN — TRAINING")
        print("="*60)

        train_loader, test_loader, train_df, test_df, norm_config = \
            prepare_training_data(args.data, batch_size=args.batch_size,
                                  save_dir=args.output_dir)

        model = ShockAwarePINN().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params:,}")

        train_model(model, train_loader, norm_config,
                    epochs=args.epochs, lr=args.lr, device=device,
                    use_lbfgs=not args.no_lbfgs, lbfgs_iter=args.lbfgs_iter,
                    save_dir=args.output_dir, train_df=train_df)

        print("\n" + "="*60)
        print("  VALIDATION ON HELD-OUT P_BACK CONDITIONS")
        print("="*60)
        validate_model(model, test_df, norm_config,
                       device=device, save_dir=args.output_dir)

    elif args.command == 'predict':
        from infer import run_prediction
        run_prediction(args)

    elif args.command == 'evaluate':
        import json
        from data import prepare_training_data
        from model import ShockAwarePINN
        from validate import validate_model

        cfg_path = os.path.join(args.output_dir, 'norm_config.json')
        with open(cfg_path) as f:
            norm_config = json.load(f)

        _, _, _, test_df, _ = prepare_training_data(
            args.data, save_dir=args.output_dir)

        model = ShockAwarePINN().to(device)
        model.load_state_dict(torch.load(
            os.path.join(args.output_dir, 'model.pth'),
            map_location=device, weights_only=True))

        print("\n" + "="*60)
        print("  EVALUATION RESULTS")
        print("="*60)
        validate_model(model, test_df, norm_config,
                       device=device, save_dir=args.output_dir)

    print("\n  Done.\n")


if __name__ == '__main__':
    main()
