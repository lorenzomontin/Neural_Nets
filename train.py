from config import TrainConfig
from data import prepare_dataloaders
from model import TabularMLP
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import numpy as np
import random


# Reproducibility 

def set_seed(seed:int) -> None:
    """Fix random seed for reproducible runs"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
  """Run evaluation on validation or test loader without updating weights."""
  model.eval()
  total_loss = 0.0
  all_probs = []
  all_targets = []

  with torch.no_grad():
    for xb, yb in loader:
      xb = xb.to(device)
      yb = yb.to(device)

      logits = model(xb)
      loss = criterion(logits, yb)
      total_loss += loss.item() * xb.size(0)

      # convert raw logits to probabilities for ROC-AUC
      probs= torch.sigmoid(logits).cpu().numpy()
      all_probs.append(probs)
      all_targets.append(yb.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_prob = np.vstack(all_probs).ravel()
    y_true = np.vstack(all_targets).ravel()
    auc = roc_auc_score(y_true, y_prob)
    return avg_loss, float(auc)




# Training
def train(cfg: TrainConfig) -> None: 
    """Train the MLP, track validation performance, and save the best weights."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # 1. Prepare data loaders and input dimension
    bundle = prepare_dataloaders(cfg)

  # 2. Build model, loss function, and optimizer from config
    model = TabularMLP(
        input_features=bundle.input_features,
        hidden_dim=cfg.hidden_dim,
        output_features=cfg.output_features,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

  # Ensure output folder exists before saving checkpoints
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_path = cfg.output_dir / f"{cfg.run_name}_best.pt"

  # 3. Epoch loop
    for epoch in range(1, cfg.epochs + 1):
        # --- Training phase ---
        model.train()
        train_loss = 0.0

        for xb, yb in bundle.train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()  # Clear old gradients
            logits = model(xb)  # Forward pass
            loss = criterion(logits, yb)  # Compute loss
            loss.backward()  # Backpropagation (compute gradients)
            optimizer.step()  # Update weights

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(bundle.train_loader.dataset)

# --- Validation phase ---
    val_loss, val_auc = evaluate_loader(
        model, bundle.val_loader, criterion, cfg.device
    )

    # Save weights whenever validation loss improves
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), best_path)

    if epoch % 10 == 0 or epoch == 1:
      print(
          f"Epoch {epoch:02d}/{cfg.epochs} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val ROC-AUC: {val_auc:.4f}"
      )

    print(f"Done. Best Val Loss: {best_val_loss:.4f} | Saved to: {best_path}")


if __name__ == "__main__":
  train(TrainConfig())
