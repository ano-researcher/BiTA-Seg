import yaml
import torch
import random
import numpy as np
from torch.utils.data import DataLoader

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

def main(config_path):
    cfg = yaml.safe_load(open(config_path))

    for seed in SEEDS:
        print(f"\n===== Training with seed {seed} =====")
        set_seed(seed)

        # Dataset and DataLoader
        ds = ImageMaskDataset(cfg["data"]["root"])
        g = torch.Generator()
        g.manual_seed(seed)

        dl = DataLoader(
            ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g
        )

        # Model, optimizer, criterion, trainer
        model = BoundaryAwareViT(**cfg["model"]).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
        criterion = DiceBCELoss()
        trainer = Trainer(model, optimizer, criterion, device="cuda")

        # Training loop
        for epoch in range(cfg["training"]["epochs"]):
            loss = trainer.train_epoch(dl)
            print(f"[Seed {seed}] Epoch {epoch}: {loss:.4f}")

        # Save model checkpoint per seed
        torch.save(
            model.state_dict(),
            f"{cfg['training']['save_dir']}/model_seed{seed}.pth"
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
