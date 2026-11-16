import argparse
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from src.fi2010.train_fi2010_deeplob import (
    DeepLOB,
    WindowArrayDataset,
    evaluate_loader,
    load_fi2010_arrays,
    select_device,
    set_seed,
)

BASE = os.path.expanduser("~/sonar_track_c/data/fi2010")
PURGE = 100          # equals horizon k=100
LAGS = (10, 50, 99)  # tabular view: last snapshot + lagged deltas (all < T=100)
ECE_BINS = 15

def get_arrays(sonar_norm: bool):
    tr_f = sorted(glob.glob(os.path.join(BASE, "Train_*.txt")))
    te_f = sorted(glob.glob(os.path.join(BASE, "Test_*.txt")))
    x_tr, y_tr = load_fi2010_arrays(tr_f, 100, sonar_norm, False)
    x_te, y_te = load_fi2010_arrays(te_f, 100, sonar_norm, False)
    split = int(0.80 * len(x_tr))
    return {
        "x_train": x_tr[:split - PURGE], "y_train": y_tr[:split - PURGE],
        "x_val": x_tr[split:], "y_val": y_tr[split:],
        "x_test": x_te, "y_test": y_te,
    }

def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = ECE_BINS) -> float:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)

def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.05, 5.0, 496):
        logp = np.log(softmax_np(logits / t) + 1e-12)
        nll = -logp[np.arange(len(y)), y].mean()
        if nll < best_nll:
            best_t, best_nll = t, nll
    return float(best_t)

def calib_report(name: str, lv: np.ndarray, yv: np.ndarray, lt: np.ndarray, yt: np.ndarray, print_full: bool = False):
    t = fit_temperature(lv, yv)   # fit on VALIDATION only
    p_raw = softmax_np(lt)        # evaluate on TEST only
    p_cal = softmax_np(lt / t)
    pred = p_raw.argmax(axis=1)
    va_pred = lv.argmax(axis=1)
    va_f1 = float(f1_score(yv, va_pred, average="macro"))
    row = {
        "name": name, "T": t, "val_f1": va_f1,
        "ece_raw": ece(p_raw, yt), "ece_cal": ece(p_cal, yt),
        "f1": float(f1_score(yt, pred, average="macro")),
        "acc": float((pred == yt).mean()),
    }
    print(
        f"  {name}: Val F1={va_f1:.4f} | Temp={t:.2f} | ECE raw={row['ece_raw']:.4f} -> cal={row['ece_cal']:.4f} "
        f"| Test Macro-F1={row['f1']:.4f} | Acc={row['acc']:.4f}",
        flush=True,
    )
    if print_full:
        print(classification_report(yt, pred, target_names=["Up (0)", "Stationary (1)", "Down (2)"], digits=4))
        print("Confusion Matrix:\n", confusion_matrix(yt, pred), flush=True)
    return row

@torch.no_grad()
def torch_logits(model, loader, device):
    model.eval()
    L, Y = [], []
    for xb, yb in loader:
        L.append(model(xb.to(device)).float().cpu().numpy())
        Y.append(yb.numpy())
    return np.concatenate(L), np.concatenate(Y)

def tab_feats(x: np.ndarray, y: np.ndarray, T: int = 100):
    ends = np.arange(T - 1, len(x))
    cols = [x[ends]] + [x[ends] - x[ends - L] for L in LAGS]
    return np.concatenate(cols, axis=1).astype(np.float32), y[ends]

def run_xgb(arr, device):
    Xtr, ytr = tab_feats(arr["x_train"], arr["y_train"])
    Xva, yva = tab_feats(arr["x_val"], arr["y_val"])
    Xte, yte = tab_feats(arr["x_test"], arr["y_test"])
    dev_str = str(device) if device.type == "cuda" else "cpu"
    clf = xgb.XGBClassifier(
        n_estimators=2000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", device=dev_str,
        early_stopping_rounds=30, eval_metric="mlogloss", random_state=42,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    print(f"XGBoost best_iteration={clf.best_iteration} | features={Xtr.shape[1]}", flush=True)
    lv = np.log(np.clip(clf.predict_proba(Xva), 1e-12, 1.0))
    lt = np.log(np.clip(clf.predict_proba(Xte), 1e-12, 1.0))
    return calib_report("XGBoost (tabular, 1 seed)", lv, yva, lt, yte, print_full=True)

class LSTMAttention(nn.Module):
    def __init__(self, in_dim=40, hidden=128, layers=2, dropout=0.2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.score = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, 1, T, 40) -> (B, T, 40)
        h, _ = self.lstm(x)
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)
        ctx = (w.unsqueeze(-1) * h).sum(dim=1)
        return self.fc(self.drop(ctx))

def train_lstm(seed, tr_ds, va_ds, te_ds, device, pin, epochs, lr, bs):
    set_seed(seed)
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=pin)
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=pin)
    te_ld = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=pin)
    model = LSTMAttention().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_f1, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device, non_blocking=pin), yb.to(device, non_blocking=pin)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(yb)
        va_acc, va_f1, _, _ = evaluate_loader(model, va_ld, device, pin)
        print(
            f"[LSTM-ATT seed {seed}] Ep {ep:02d}/{epochs:02d} | Train Loss: {loss_sum/len(tr_ds):.4f} "
            f"| Val Acc: {va_acc:.4f} | Val Macro-F1: {va_f1:.4f}",
            flush=True,
        )
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"--> [LSTM-ATT seed {seed}] best Val F1={best_f1:.4f}", flush=True)
    lv, yv = torch_logits(model, va_ld, device)
    lt, yt = torch_logits(model, te_ld, device)
    return best_f1, best_state, lv, yv, lt, yt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--skip-lstm", action="store_true")
    ap.add_argument("--lstm-zscore", action="store_true")
    args = ap.parse_args()

    device, dev_name = select_device()
    pin = device.type == "cuda"
    print(f"Device: {device} ({dev_name})", flush=True)

    raw = get_arrays(False)
    son = get_arrays(True)
    rows, lstm_rows = [], []

    print("\n=== [1] DeepLOB checkpoints: temperature fit on val, ECE on test ===", flush=True)
    for name, T, arr, pt in [
        ("DeepLOB decpre T=100", 100, raw, "deeplob_fi2010_decpre_T100_k100.pt"),
        ("DeepLOB sonar_norm T=100", 100, son, "deeplob_fi2010_sonar_norm_T100_k100.pt"),
        ("DeepLOB sonar_norm T=600", 600, son, "deeplob_fi2010_sonar_norm_T600_k100.pt"),
    ]:
        model = DeepLOB(3).to(device)
        model.load_state_dict(torch.load(os.path.join(BASE, pt), map_location=device))
        va_ld = DataLoader(WindowArrayDataset(arr["x_val"], arr["y_val"], T, 1), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin)
        te_ld = DataLoader(WindowArrayDataset(arr["x_test"], arr["y_test"], T, 1), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin)
        lv, yv = torch_logits(model, va_ld, device)
        lt, yt = torch_logits(model, te_ld, device)
        rows.append(calib_report(name, lv, yv, lt, yt))

    print("\n=== [2] XGBoost tabular baseline (sonar_norm, T=100 view) ===", flush=True)
    rows.append(run_xgb(son, device))

    if not args.skip_lstm:
        print("\n=== [3] LSTM-Attention (sonar_norm, T=100) ===", flush=True)
        if args.lstm_zscore:
            mu = son["x_train"].mean(axis=0)
            sd = son["x_train"].std(axis=0) + 1e-8
            f32 = lambda a: ((a - mu) / sd).astype(np.float32)
        else:
            f32 = lambda a: a.astype(np.float32)
        tr_ds = WindowArrayDataset(f32(son["x_train"]), son["y_train"], 100, 1)
        va_ds = WindowArrayDataset(f32(son["x_val"]), son["y_val"], 100, 1)
        te_ds = WindowArrayDataset(f32(son["x_test"]), son["y_test"], 100, 1)
        global_best_vf1, best_lstm_pack = -1.0, None
        for seed in args.seeds:
            vf1, state, lv, yv, lt, yt = train_lstm(seed, tr_ds, va_ds, te_ds, device, pin, args.epochs, args.lr, args.batch_size)
            r = calib_report(f"LSTM-Attention seed {seed}", lv, yv, lt, yt)
            lstm_rows.append(r)
            if vf1 > global_best_vf1:
                global_best_vf1 = vf1
                best_lstm_pack = (seed, state, lv, yv, lt, yt)
        rows.extend(lstm_rows)
        b_seed, b_state, lv, yv, lt, yt = best_lstm_pack
        lstm_pt = os.path.join(BASE, "lstm_attention_fi2010_sonar_norm_T100_k100.pt")
        torch.save(b_state, lstm_pt)
        print(f"\n=== BEST-VAL LSTM-ATTENTION TEST REPORT (Seed {b_seed}) ===", flush=True)
        calib_report(f"LSTM-Attention Best-Val (Seed {b_seed})", lv, yv, lt, yt, print_full=True)
        print(f"Saved best-val LSTM-Attention weights -> {lstm_pt}", flush=True)

    print("\n=== SUMMARY: FI-2010 HELD-OUT TEST SPLIT ===")
    print(f"{'Model':<30}{'Val F1':>8}{'Temp':>7}{'ECE raw':>10}{'ECE cal':>10}{'Macro-F1':>11}{'Acc':>9}")
    for r in rows:
        print(f"{r['name']:<30}{r['val_f1']:>8.4f}{r['T']:>7.2f}{r['ece_raw']:>10.4f}{r['ece_cal']:>10.4f}{r['f1']:>11.4f}{r['acc']:>9.4f}")
    if len(lstm_rows) > 1:
        for key in ("f1", "acc", "ece_raw", "ece_cal"):
            vals = [r[key] for r in lstm_rows]
            print(f"LSTM-Attention {key}: {np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f} (sample std, ddof=1, n={len(vals)})")

if __name__ == "__main__":
    main()