import yaml
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from src.models.vit_ccattn import BoundaryAwareViT
from src.data.dataset_loader import ImageMaskDataset
from src.training.trainer import Trainer
from src.training.losses import DiceBCELoss

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

SEEDS = [42, 123, 2023]
NUM_FOLDS = 5


def train_5fold_cv(config_path):
    cfg = yaml.safe_load(open(config_path))
    dataset = ImageMaskDataset(cfg["data"]["root"])
    n_samples = len(dataset)
    indices = np.arange(n_samples)

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), 1):
        print(f"\n===== Fold {fold}/{NUM_FOLDS} =====")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")
            set_seed(seed)

            # DataLoaders
            g = torch.Generator()
            g.manual_seed(seed)

            train_loader = DataLoader(
                train_subset,
                batch_size=cfg["training"]["batch_size"],
                shuffle=True,
                num_workers=4,
                worker_init_fn=seed_worker,
                generator=g
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=cfg["training"]["batch_size"],
                shuffle=False,
                num_workers=4
            )

            # Model & optimizer
            model = BoundaryAwareViT(**cfg["model"]).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
            criterion = DiceBCELoss()
            trainer = Trainer(model, optimizer, criterion, device="cuda")

            # Training loop
            for epoch in range(cfg["training"]["epochs"]):
                loss = trainer.train_epoch(train_loader)
                print(f"[Fold {fold} Seed {seed}] Epoch {epoch}: {loss:.4f}")

            # Validation metrics for this fold & seed
            val_dice = trainer.evaluate(val_loader)  # implement this method in your Trainer
            print(f"[Fold {fold} Seed {seed}] Val Dice: {val_dice:.4f}")

            # Save checkpoint per fold & seed
            ckpt_path = f"{cfg['training']['save_dir']}/fold{fold}_seed{seed}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            fold_metrics.append({
                "fold": fold,
                "seed": seed,
                "val_dice": val_dice,
                "checkpoint": ckpt_path
            })

    # Aggregate metrics across folds & seeds
    all_dice = [m["val_dice"] for m in fold_metrics]
    print(f"\n5-Fold CV Mean Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    return fold_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train_5fold_cv(args.config)
