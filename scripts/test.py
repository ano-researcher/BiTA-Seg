import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from src.models.vit_ccattn import BoundaryAwareViT
from src.data.dataset_loader import ImageMaskDataset
from src.evaluation.evaluator import dice, segmentation_metrics

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(checkpoints, data_root="data/", batch_size=1):
    set_seed(42)  # reproducible evaluation

    # Fixed test dataset
    test_dataset = ImageMaskDataset(data_root)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_metrics = {
        "dice": [],
        "iou": [],
        "accuracy": [],
        "precision": [],
        "recall": []
    }

    for ckpt in checkpoints:
        print(f"\nEvaluating checkpoint: {ckpt}")
        model = BoundaryAwareViT(img_size=256, patch_size=16, embed_dim=256, depth=6, num_heads=8)
        model.load_state_dict(torch.load(ckpt))
        model.cuda()
        model.eval()

        dice_scores, ious, accs, precs, recs = [], [], [], [], []

        with torch.no_grad():
            for img, mask in test_loader:
                img, mask = img.cuda(), mask.cuda()
                pred = torch.sigmoid(model(img))

                # Dice
                dice_scores.append(dice(pred, mask).item())

                # Other metrics
                acc, iou, prec, rec = segmentation_metrics(pred, mask)
                accs.append(acc)
                ious.append(iou)
                precs.append(prec)
                recs.append(rec)

        # Store mean per checkpoint
        all_metrics["dice"].append(np.mean(dice_scores))
        all_metrics["iou"].append(np.mean(ious))
        all_metrics["accuracy"].append(np.mean(accs))
        all_metrics["precision"].append(np.mean(precs))
        all_metrics["recall"].append(np.mean(recs))

        print(f"[{ckpt}] Dice: {np.mean(dice_scores):.4f} | IoU: {np.mean(ious):.4f} | "
              f"Acc: {np.mean(accs):.4f} | Prec: {np.mean(precs):.4f} | Rec: {np.mean(recs):.4f}")

    # Aggregate over all fold × seed checkpoints
    print("\n===== Overall Test Metrics (mean ± std across all models) =====")
    for k, v in all_metrics.items():
        print(f"{k.capitalize()}: {np.mean(v):.4f} ± {np.std(v):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to all fold+seed checkpoints")
    parser.add_argument("--data_root", default="data/")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    main(args.checkpoints, args.data_root, args.batch_size)
