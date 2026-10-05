# LSTM + multi-head self-attention that diagnoses WHY an episode failed and WHICH
# timesteps were responsible -- this is the XAI component of research-grade ARTEMIS
import numpy as np
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)

FAILURE_MODES = [
    "PROFITABLE",
    "PANIC_SELL",
    "HELD_TOO_LONG",
    "OVERTRADING",
    "MISSED_RALLY",
    "MAX_DRAWDOWN",
]

FIX_TYPES = [
    "NO_FIX_NEEDED",
    "FIX_PANIC_SELL",
    "FIX_HELD_TOO_LONG",
    "FIX_OVERTRADING",
    "FIX_MISSED_RALLY",
    "FIX_MAX_DRAWDOWN",
]

FAILURE_TO_FIX = {
    "PROFITABLE": "NO_FIX_NEEDED",
    "PANIC_SELL": "FIX_PANIC_SELL",
    "HELD_TOO_LONG": "FIX_HELD_TOO_LONG",
    "OVERTRADING": "FIX_OVERTRADING",
    "MISSED_RALLY": "FIX_MISSED_RALLY",
    "MAX_DRAWDOWN": "FIX_MAX_DRAWDOWN",
}

SEQ_LEN = 252
INPUT_SIZE = 6
STEP_KEYS = ["daily_return", "drawdown", "rsi", "position", "days_held", "reward"]


class EpisodeDiagnosticLSTM(nn.Module):
    # architecture:
    #   input (batch, T, 6)
    #   -> LSTM(hidden=64, layers=2)      outputs (batch, T, 64)
    #   -> MultiheadAttention(heads=4)    outputs (batch, T, 64) + weights
    #   -> LayerNorm + residual
    #   -> mean pooling over T            outputs (batch, 64)
    #   -> Dropout(0.3)
    #   -> failure_head: Linear(64, 6)
    #   -> fix_head:     Linear(64, 6)

    def __init__(self, input_size=INPUT_SIZE, hidden_size=64, num_layers=2, num_heads=4, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.failure_head = nn.Linear(hidden_size, len(FAILURE_MODES))
        self.fix_head = nn.Linear(hidden_size, len(FIX_TYPES))

    def forward(self, x: torch.Tensor):
        # x: (batch, T, 6) -- returns (failure_logits, fix_logits, attn_weights[batch,T,T])
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attended = self.layer_norm(lstm_out + attn_out)
        pooled = attended.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.failure_head(pooled), self.fix_head(pooled), attn_weights

    @staticmethod
    def steps_to_tensor(trajectory_steps: list, seq_len: int = SEQ_LEN) -> torch.Tensor:
        seq = [[s.get(k, 0.0) for k in STEP_KEYS] for s in trajectory_steps]
        if len(seq) < seq_len:
            seq = seq + [[0.0] * INPUT_SIZE] * (seq_len - len(seq))
        else:
            seq = seq[:seq_len]
        return torch.tensor([seq], dtype=torch.float32)

    def predict(self, trajectory_steps: list) -> dict:
        x = self.steps_to_tensor(trajectory_steps)
        device = next(self.parameters()).device
        x = x.to(device)
        self.eval()
        with torch.no_grad():
            f_logits, fix_logits, attn_weights = self.forward(x)
            f_probs = torch.softmax(f_logits, dim=-1)[0].cpu().numpy()
            fix_probs = torch.softmax(fix_logits, dim=-1)[0].cpu().numpy()
            attn = attn_weights[0].cpu().numpy()

        f_idx=int(f_probs.argmax())
        fix_idx = int(fix_probs.argmax())
        return {
            "failure_mode": FAILURE_MODES[f_idx],
            "fix_type": FIX_TYPES[fix_idx],
            "failure_confidence": float(f_probs[f_idx]),
            "fix_confidence": float(fix_probs[fix_idx]),
            "failure_probs": {m: float(p) for m, p in zip(FAILURE_MODES, f_probs)},
            "fix_probs": {ft: float(p) for ft, p in zip(FIX_TYPES, fix_probs)},
            "attention_weights": attn.mean(axis=0).tolist(),
        }


if __name__ == "__main__":
    print("[ARTEMIS] Running diagnostics/lstm_model.py...")
    model = EpisodeDiagnosticLSTM()
    dummy_steps = [{"daily_return": 0.0, "drawdown": 0.0, "rsi": 0.5, "position": 0.5,
                     "days_held": 0.0, "reward": 0.0} for _ in range(50)]
    result = model.predict(dummy_steps)
    print({k: v for k, v in result.items() if k != "attention_weights"})
    print("attention_weights length:", len(result["attention_weights"]))
