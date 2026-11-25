from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn

from config import TrainConfig
from data import prepare_dataloaders
from model import TabularMLP


def evaluate_checkpoint(
    cfg: TrainConfig,
    checkpoint_path: Path | None = None,
) -> dict[str, float]:
  """Load saved model weights and evaluate once on the held-out test set."""
  device = torch.device(cfg.device)
  bundle = prepare_dataloaders(cfg)

  # 1. Build matching architecture and load saved weights
  model = TabularMLP(
      input_features=bundle.input_features,
      hidden_dim=cfg.hidden_dim,
      output_features=cfg.output_features,
      dropout=cfg.dropout,
  ).to(device)

  ckpt = (
      checkpoint_path
      if checkpoint_path is not None
      else cfg.output_dir / f"{cfg.run_name}_best.pt"
  )
  state_dict = torch.load(ckpt, map_location=device, weights_only=True)
  model.load_state_dict(state_dict)

  # 2. Run inference on test_loader without gradients or dropout
  criterion = nn.BCEWithLogitsLoss()
  model.eval()

  total_loss = 0.0
  all_probs = []
  all_targets = []

  with torch.no_grad():
    for xb, yb in bundle.test_loader:
      xb = xb.to(device)
      yb = yb.to(device)

      logits = model(xb)
      loss = criterion(logits, yb)
      total_loss += loss.item() * xb.size(0)

      probs = torch.sigmoid(logits).cpu().numpy()
      all_probs.append(probs)
      all_targets.append(yb.cpu().numpy())

  test_loss = total_loss / len(bundle.test_loader.dataset)
  y_prob = np.vstack(all_probs).ravel()
  y_true = np.vstack(all_targets).ravel()
  y_pred = (y_prob >= 0.5).astype(int)

  acc = float(accuracy_score(y_true, y_pred))
  auc = float(roc_auc_score(y_true, y_prob))

  metrics = {
      "test_loss": test_loss,
      "test_accuracy": acc,
      "test_roc_auc": auc,
  }
  print(
      f"Checkpoint: {ckpt} | "
      f"Test Loss: {test_loss:.4f} | "
      f"Test Acc: {acc:.4f} | "
      f"Test ROC-AUC: {auc:.4f}"
  )
  return metrics


if __name__ == "__main__":
  evaluate_checkpoint(TrainConfig())