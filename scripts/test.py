import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from src.models.vit_ccattn import BoundaryAwareViT
from src.evaluation.evaluator import dice
from src.data.dataset_loader import ImageMaskDataset

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(checkpoints, data_root="data/"):
    set_seed(42)  # Ensures reproducibility in evaluation

    dl = DataLoader(
        ImageMaskDataset(data_root),
        batch_size=1,
        shuffle=False
    )

    all_scores = []

    for ckpt in checkpoints:
        print(f"\nEvaluating checkpoint: {ckpt}")
        model = BoundaryAwareViT(img_size=256, patch_size=16, embed_dim=256, depth=6, num_heads=8)
        model.load_state_dict(torch.load(ckpt))
        model.cuda()
        model.eval()

        scores = []
        with torch.no_grad():
            for img, mask in dl:
                img, mask = img.cuda(), mask.cuda()
                pred = torch.sigmoid(model(img))
                scores.append(dice(pred, mask).item())

        mean_dice = sum(scores) / len(scores)
        print(f"Mean Dice for {ckpt}: {mean_dice:.4f}")
        all_scores.append(mean_dice)

    # Optional: mean ± std across seeds
    print(f"\nOverall Mean Dice: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to model checkpoints (per seed)")
    parser.add_argument("--data_root", default="data/")
    args = parser.parse_args()
    main(args.checkpoints, args.data_root)
