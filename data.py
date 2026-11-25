"""This scripts handles data preprocessing"""

from dataclasses import dataclass
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import TrainConfig

@dataclass
class DataBundle:
  """Container holding train, validation, and test loaders plus metadata."""

  train_loader: DataLoader
  val_loader: DataLoader
  test_loader: DataLoader
  input_features: int
  scaler: StandardScaler


def _make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
  """Convert feature and target arrays into a PyTorch DataLoader."""
  x_tensor = torch.from_numpy(x)
  y_tensor = torch.from_numpy(y).unsqueeze(1)  # (N,) -> (N, 1)
  dataset = TensorDataset(x_tensor, y_tensor)
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def prepare_dataloaders(cfg: TrainConfig) -> DataBundle:
  """Load, split, standardize, and wrap the breast cancer dataset."""
  dataset = load_breast_cancer()
  x = dataset.data.astype(np.float32)
  y = dataset.target.astype(np.float32)

  # 1. Hold out the test set first so it never influences splitting or scaling
  x_rem, x_test, y_rem, y_test = train_test_split(
      x,
      y,
      test_size=cfg.test_size,
      random_state=cfg.seed,
      stratify=y,
  )

  # 2. Split validation out of the remaining training pool
  val_ratio = cfg.val_size / (1.0 - cfg.test_size)
  x_train, x_val, y_train, y_val = train_test_split(
      x_rem,
      y_rem,
      test_size=val_ratio,
      random_state=cfg.seed,
      stratify=y_rem,
  )

  # 3. Fit StandardScaler strictly on training features to prevent data leakage
  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_val = scaler.transform(x_val)
  x_test = scaler.transform(x_test)

  # 4. Build DataLoaders (shuffle only the training set)
  train_loader = _make_loader(x_train, y_train, cfg.batch_size, shuffle=True)
  val_loader = _make_loader(x_val, y_val, cfg.batch_size, shuffle=False)
  test_loader = _make_loader(x_test, y_test, cfg.batch_size, shuffle=False)

  return DataBundle(
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader,
      input_features=x.shape[1],
      scaler=scaler,
  )


if __name__ == "__main__":
  config = TrainConfig()
  bundle = prepare_dataloaders(config)
  xb, yb = next(iter(bundle.train_loader))
  print("Input features:", bundle.input_features)
  print("Train batch X shape:", xb.shape)
  print("Train batch Y shape:", yb.shape)