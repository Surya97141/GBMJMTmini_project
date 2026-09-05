"""Supervised training loop for the episode diagnostic LSTM."""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from diagnostics.lstm_model import EpisodeDiagnosticLSTM

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_lstm(
    data_path: str = "data/edt_train.npz",
    save_path: str = "models/lstm/lstm_model.pth",
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_split: float = 0.2,
) -> EpisodeDiagnosticLSTM:
    data = np.load(data_path)
    X, y_failure, y_fix = data["X"], data["y_failure"], data["y_fix"]

    idx = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_split, stratify=y_failure, random_state=SEED
    )

    def make_loader(indices, shuffle):
        ds = TensorDataset(
            torch.tensor(X[indices], dtype=torch.float32),
            torch.tensor(y_failure[indices], dtype=torch.long),
            torch.tensor(y_fix[indices], dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)

    model = EpisodeDiagnosticLSTM().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for epoch in tqdm(range(1, epochs + 1), desc="Training LSTM"):
        model.train()
        total_loss = 0.0
        for xb, yf_b, yx_b in train_loader:
            xb, yf_b, yx_b = xb.to(DEVICE), yf_b.to(DEVICE), yx_b.to(DEVICE)
            optimizer.zero_grad()
            failure_logits, fix_logits = model(xb)
            loss = criterion(failure_logits, yf_b) + criterion(fix_logits, yx_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        scheduler.step()
        train_loss = total_loss / len(train_idx)

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yf_b, yx_b in val_loader:
                xb, yf_b = xb.to(DEVICE), yf_b.to(DEVICE)
                failure_logits, _ = model(xb)
                preds = torch.argmax(failure_logits, dim=-1)
                correct += (preds == yf_b).sum().item()
        val_acc = correct / len(val_idx) if len(val_idx) > 0 else 0.0

        print(f"[ARTEMIS] Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_failure_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), save_path)
    print(f"[ARTEMIS] Best val_failure_acc={best_val_acc:.4f} | saved to {save_path}")

    return model


if __name__ == "__main__":
    print("[ARTEMIS] Running diagnostics.train_lstm...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/edt_train.npz")
    parser.add_argument("--save_path", type=str, default="models/lstm/lstm_model.pth")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    train_lstm(
        data_path=args.data_path,
        save_path=args.save_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
