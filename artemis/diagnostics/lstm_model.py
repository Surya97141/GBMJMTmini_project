"""LSTM that reads an episode trajectory and diagnoses *why* the agent traded the way it did."""
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

STEP_FEATURES = ["daily_return", "drawdown", "rsi", "position", "days_held", "reward"]
SEQ_LEN = 252


class EpisodeDiagnosticLSTM(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.failure_head = nn.Linear(hidden_size, 6)
        self.fix_head = nn.Linear(hidden_size, 6)

    def forward(self, x: torch.Tensor):
        _, (h_n, _) = self.lstm(x)
        final_hidden = h_n[-1]
        final_hidden = self.dropout(final_hidden)
        failure_logits = self.failure_head(final_hidden)
        fix_logits = self.fix_head(final_hidden)
        return failure_logits, fix_logits

    @staticmethod
    def steps_to_tensor(trajectory_steps: list, seq_len: int = SEQ_LEN) -> torch.Tensor:
        arr = np.zeros((seq_len, len(STEP_FEATURES)), dtype=np.float32)
        n = min(len(trajectory_steps), seq_len)
        for i in range(n):
            step = trajectory_steps[i]
            arr[i] = [step[feat] for feat in STEP_FEATURES]
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

    def predict(self, trajectory_steps: list) -> dict:
        self.eval()
        x = self.steps_to_tensor(trajectory_steps)
        with torch.no_grad():
            failure_logits, fix_logits = self.forward(x)
            failure_probs = torch.softmax(failure_logits, dim=-1).squeeze(0)
            fix_probs = torch.softmax(fix_logits, dim=-1).squeeze(0)

        failure_idx = int(torch.argmax(failure_probs).item())
        fix_idx = int(torch.argmax(fix_probs).item())

        return {
            "failure_mode": FAILURE_MODES[failure_idx],
            "fix_type": FIX_TYPES[fix_idx],
            "failure_confidence": float(failure_probs[failure_idx].item()),
            "fix_confidence": float(fix_probs[fix_idx].item()),
            "failure_probs": {mode: float(failure_probs[i].item()) for i, mode in enumerate(FAILURE_MODES)},
        }


if __name__ == "__main__":
    print("[ARTEMIS] Running diagnostics.lstm_model...")
    model = EpisodeDiagnosticLSTM()
    dummy_steps = [{"daily_return": 0.0, "drawdown": 0.0, "rsi": 0.5, "position": 0.5,
                     "days_held": 0.0, "reward": 0.0} for _ in range(50)]
    result = model.predict(dummy_steps)
    print(result)
