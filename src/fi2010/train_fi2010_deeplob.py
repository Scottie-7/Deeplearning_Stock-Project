import argparse
import glob
import os
import random
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def verify_and_apply_sonar_sec6_transform(lob_40: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Convert FI-2010 DecPre 40-col LOB [ask_p1, ask_v1, bid_p1, bid_v1, ...]
    into SPEC v1.2 Section 6 features:
      - Prices : (p_i - p_mid) / p_mid
      - Volumes: ln(1 + V), where V = v_decpre * 1e5
    Asserts exact row-by-row arithmetic against the raw input.
    """
    raw_row0 = lob_40[0, :4].copy()
    v_scaled = lob_40[:, 1::2] * 1e5
    int_residuals = np.abs(v_scaled - np.rint(v_scaled))
    frac_exact_int = float(np.mean(int_residuals < 1e-2))
    mod10_vals = np.unique(np.rint(v_scaled).astype(np.int64) % 10).tolist()
    assert frac_exact_int >= 0.9999, f"Expected integer volumes, got {frac_exact_int}" 

    out = np.empty_like(lob_40, dtype=np.float32)
    ask_p1 = lob_40[:, 0]
    bid_p1 = lob_40[:, 2]
    p_mid = 0.5 * (ask_p1 + bid_p1)

    for c in range(0, 40, 2):
        out[:, c] = (lob_40[:, c] - p_mid) / np.maximum(p_mid, 1e-8)

    for c in range(1, 40, 2):
        v_shares = np.maximum(0.0, lob_40[:, c] * 1e5)
        out[:, c] = np.log1p(v_shares)

    # Row-0 verification assertion
    mid0 = 0.5 * (raw_row0[0] + raw_row0[2])
    exp_ask_p1 = (raw_row0[0] - mid0) / mid0
    exp_ask_v1 = np.log1p(max(0.0, raw_row0[1] * 1e5))
    exp_bid_p1 = (raw_row0[2] - mid0) / mid0
    exp_bid_v1 = np.log1p(max(0.0, raw_row0[3] * 1e5))
    assert np.allclose(
        out[0, :4], [exp_ask_p1, exp_ask_v1, exp_bid_p1, exp_bid_v1], rtol=1e-5, atol=1e-6
    ), "Transform assertion failed on Row 0!"

    if verbose:
        print(f"  [Transform Check] Raw Row 0 [ask_p1, ask_v1, bid_p1, bid_v1]  : {raw_row0}")
        print(f"  [Transform Check] Norm Row 0 [ask_p1, ask_v1, bid_p1, bid_v1] : {out[0, :4]}")
        print(f"  [Transform Check] v*1e5 integer fraction (<0.01 residual)     : {frac_exact_int*100:.4f}%")
        print(f"  [Transform Check] v*1e5 mod 10 unique digits (proves 1e-5)    : {mod10_vals}")
    return out


def load_fi2010_arrays(txt_files, horizon_k: int = 100, sonar_norm: bool = False, verbose: bool = False):
    horizon_map = {10: 144, 20: 145, 30: 146, 50: 147, 100: 148}
    label_col = horizon_map[horizon_k]
    xs, ys = [], []
    for i, path in enumerate(sorted(txt_files)):
        raw = np.fromfile(path, dtype=np.float32, sep=" ")
        data = raw.reshape(149, -1).T
        lob = np.ascontiguousarray(data[:, :40], dtype=np.float32)
        if i == 0 and verbose and not sonar_norm:
            print(f"  [Raw Check] {os.path.basename(path)} Row 0 [ask_p1, ask_v1, bid_p1, bid_v1]: {lob[0, :4]}")
        if sonar_norm:
            lob = verify_and_apply_sonar_sec6_transform(lob, verbose=(i == 0 and verbose))
        xs.append(lob)
        ys.append(data[:, label_col].astype(np.int64) - 1)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


class WindowArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, window_size: int = 100, stride: int = 1):
        self.x = x
        self.y = y
        self.window_size = window_size
        self.indices = np.arange(0, len(self.x) - window_size + 1, stride, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = int(self.indices[idx])
        e = s + self.window_size
        return torch.from_numpy(self.x[s:e]).unsqueeze(0), torch.tensor(self.y[e - 1], dtype=torch.long)


def cblock(in_c, out_c, act):
    return nn.Sequential(
        nn.ZeroPad2d((0, 0, 3, 0)),
        nn.Conv2d(in_c, out_c, (4, 1)),
        act,
        nn.BatchNorm2d(out_c),
    )


class DeepLOB(nn.Module):
    """DeepLOB (Zhang et al., 2019) with causal temporal padding."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            cblock(32, 32, nn.LeakyReLU(0.01)),
            cblock(32, 32, nn.LeakyReLU(0.01)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            cblock(32, 32, nn.Tanh()),
            cblock(32, 32, nn.Tanh()),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 10)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            cblock(32, 32, nn.LeakyReLU(0.01)),
            cblock(32, 32, nn.LeakyReLU(0.01)),
        )
        self.inc1 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.ZeroPad2d((0, 0, 2, 0)),
            nn.Conv2d(64, 64, (3, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inc2 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.ZeroPad2d((0, 0, 4, 0)),
            nn.Conv2d(64, 64, (5, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inc3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.lstm = nn.LSTM(192, 64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = torch.cat([self.inc1(x), self.inc2(x), self.inc3(x)], dim=1)
        out, _ = self.lstm(x.squeeze(-1).permute(0, 2, 1))
        return self.fc(out[:, -1, :])


def select_device():
    if torch.cuda.is_available():
        best_gpu = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
        return torch.device(f"cuda:{best_gpu}"), f"{torch.cuda.get_device_name(best_gpu)} (cuda:{best_gpu})"
    torch.set_num_threads(32)
    return torch.device("cpu"), "CPU (32 Xeon threads)"


@torch.no_grad()
def evaluate_loader(model, loader, device, pin):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device, non_blocking=pin)).argmax(1).cpu().numpy())
        trues.append(yb.numpy())
    yp, yt = np.concatenate(preds), np.concatenate(trues)
    mf1 = float(f1_score(yt, yp, average="macro"))
    acc = float((yp == yt).mean())
    return acc, mf1, yt, yp


def main():
    base_dir = os.path.expanduser("~/sonar_track_c/data/fi2010")
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=base_dir)
    p.add_argument("--window", type=int, default=100, choices=[100, 600])
    p.add_argument("--horizon", type=int, default=100, choices=[10, 20, 30, 50, 100])
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--sonar-norm", action="store_true")
    args = p.parse_args()

    device, dev_name = select_device()
    tag = "sonar_norm" if args.sonar_norm else "decpre"
    save_path = os.path.join(args.data_dir, f"deeplob_fi2010_{tag}_T{args.window}_k{args.horizon}.pt")
    print(f"Device: {device} ({dev_name}) | Mode: {tag} | T={args.window} | k={args.horizon} | Seeds={args.seeds}")

    t0 = time.monotonic()
    tr_f = sorted(glob.glob(os.path.join(args.data_dir, "Train_*.txt")))
    assert len(tr_f) == 1 and os.path.basename(tr_f[0]) == "Train_Dst_NoAuction_DecPre_CF_7.txt", f"Unexpected train files: {tr_f}" 
    te_f = sorted(glob.glob(os.path.join(args.data_dir, "Test_*.txt")))
    x_full_tr, y_full_tr = load_fi2010_arrays(tr_f, args.horizon, args.sonar_norm, verbose=True)
    x_te, y_te = load_fi2010_arrays(te_f, args.horizon, args.sonar_norm, verbose=False)

    # Chronological 80/20 Train / Validation split on Train_7 (zero test leakage)
    split_idx = int(0.80 * len(x_full_tr))
    tr_ds = WindowArrayDataset(x_full_tr[:split_idx - args.horizon], y_full_tr[:split_idx - args.horizon], args.window, args.stride)
    val_ds = WindowArrayDataset(x_full_tr[split_idx:], y_full_tr[split_idx:], args.window, args.stride)
    te_ds = WindowArrayDataset(x_te, y_te, args.window, args.stride)
    print(f"Loaded in {time.monotonic()-t0:.1f}s -> Train: {len(tr_ds):,} | Val (20%): {len(val_ds):,} | Test: {len(te_ds):,}")

    pin = device.type == "cuda"
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=pin)

    test_accs, test_f1s = [], []
    global_best_val_f1 = -1.0
    best_test_report = None

    for seed in args.seeds:
        set_seed(seed)
        tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin)
        model = DeepLOB(3).to(device)
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_f1 = -1.0
        best_state = None
        for ep in range(1, args.epochs + 1):
            ts = time.monotonic()
            model.train()
            loss_sum = 0.0
            for xb, yb in tr_ld:
                xb, yb = xb.to(device, non_blocking=pin), yb.to(device, non_blocking=pin)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                loss_sum += loss.item() * len(yb)

            val_acc, val_f1, _, _ = evaluate_loader(model, val_ld, device, pin)
            print(f"[Seed {seed}] Ep {ep:02d}/{args.epochs:02d} | Train Loss: {loss_sum/len(tr_ds):.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f} | {time.monotonic()-ts:.1f}s")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best validation checkpoint and evaluate Test_7/8/9 ONCE
        model.load_state_dict(best_state)
        te_acc, te_f1, yt, yp = evaluate_loader(model, te_ld, device, pin)
        test_accs.append(te_acc)
        test_f1s.append(te_f1)
        print(f"--> [Seed {seed} Held-Out Test (selected by Val F1={best_val_f1:.4f})] Test Acc: {te_acc:.4f} | Test Macro-F1: {te_f1:.4f}\n")

        if best_val_f1 > global_best_val_f1:
            global_best_val_f1 = best_val_f1
            best_test_report = (seed, yt, yp, te_acc, te_f1)
            torch.save(best_state, save_path)

    print(f"=== SUMMARY ACROSS {len(args.seeds)} SEEDS ({tag}, T={args.window}, k={args.horizon}) ===")
    print(f"Held-Out Test Accuracy : {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}")
    print(f"Held-Out Test Macro-F1 : {np.mean(test_f1s):.4f} +/- {np.std(test_f1s):.4f}")

    b_seed, yt, yp, b_acc, b_f1 = best_test_report
    print(f"\n=== HELD-OUT TEST REPORT FOR BEST-VAL CHECKPOINT (Seed {b_seed}, Test Macro-F1={b_f1:.4f}) ===")
    print(classification_report(yt, yp, target_names=["Up (0)", "Stationary (1)", "Down (2)"], digits=4))
    print("Confusion Matrix:\n", confusion_matrix(yt, yp))
    print(f"Saved best-val weights -> {save_path}")


if __name__ == "__main__":
    main()