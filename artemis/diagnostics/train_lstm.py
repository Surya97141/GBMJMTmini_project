# Supervised training loop for the attention-augmented episode diagnostic LSTM
import os

# scikit-learn and PyTorch each bundle their own OpenMP runtime; loading both
# in one process on Windows causes a duplicate-runtime conflict that can
# segfault during the LSTM forward pass. These must be set before numpy/
# torch/sklearn are imported, since the native libraries read them at
# import/init time, not at first use.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from diagnostics.lstm_model import EpisodeDiagnosticLSTM

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def train_lstm(data_path="data/edt_train.npz", save_path="models/lstm/lstm_model.pth",
                epochs=150, lr=1e-3, batch_size=32, val_split=0.2):
    data = np.load(data_path)
    X, y_failure, y_fix = data["X"], data["y_failure"], data["y_fix"]

    X_tr, X_val, yf_tr, yf_val, yx_tr, yx_val = train_test_split(
        X, y_failure, y_fix, test_size=val_split, stratify=y_failure, random_state=SEED
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EpisodeDiagnosticLSTM().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    tr_X = torch.tensor(X_tr, dtype=torch.float32)
    tr_yf = torch.tensor(yf_tr, dtype=torch.long)
    tr_yx = torch.tensor(yx_tr, dtype=torch.long)
    val_X = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_yf = torch.tensor(yf_val, dtype=torch.long).to(device)

    dataset = torch.utils.data.TensorDataset(tr_X, tr_yf, tr_yx)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_acc=0.0
    best_state = None  # snapshot of weights from the best epoch so far, saved at the end

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yf_b, yx_b in loader:
            xb, yf_b, yx_b = xb.to(device), yf_b.to(device), yx_b.to(device)
            f_log, fix_log, _ = model(xb)
            loss = criterion(f_log, yf_b) + criterion(fix_log, yx_b)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            f_log_v, _, _ = model(val_X)
            preds = f_log_v.argmax(dim=-1)
            val_acc = float((preds == val_yf).float().mean())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: loss={total_loss / len(loader):.4f}  "
                  f"val_acc={val_acc:.3f}  best={best_val_acc:.3f}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(best_state, save_path)
    print(f"[ARTEMIS] Best val accuracy: {best_val_acc:.3f} -> saved to {save_path}")
    model.load_state_dict(best_state)
    return model


if __name__ == "__main__":
    print("[ARTEMIS] Running diagnostics/train_lstm.py...")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/edt_train.npz")
    parser.add_argument("--save_path", default="models/lstm/lstm_model.pth")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    train_lstm(data_path=args.data_path, save_path=args.save_path,
               epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
