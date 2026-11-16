import copy
import glob
import os
import time
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader

from src.fi2010.train_fi2010_deeplob import (
    DeepLOB,
    WindowArrayDataset,
    select_device,
    set_seed,
    verify_and_apply_sonar_sec6_transform,
)
from src.fi2010.baselines_fi2010 import LSTMAttention, calib_report

BASE = os.path.expanduser("~/sonar_track_c/data/fi2010")
SEEDS = (42, 43, 44)
PURGE = 100
T_WIN = 100
LAGS = (10, 50, 99)


def load_file_stock_blocks(path: str, sonar_norm: bool, horizon_col: int = 148, verbose: bool = False):
    raw = np.fromfile(path, dtype=np.float32, sep=" ")
    data = raw.reshape(149, -1).T
    lob = np.ascontiguousarray(data[:, :40], dtype=np.float32)
    y = data[:, horizon_col].astype(np.int64) - 1

    # Detect the 4 stock-boundary jumps (>5% mid-price jump) on raw prices BEFORE any transform
    mid = (lob[:, 0].astype(np.float64) + lob[:, 2].astype(np.float64)) / 2.0
    jumps = (np.where(np.abs(np.diff(mid)) / mid[:-1] > 0.05)[0] + 1).tolist()
    assert len(jumps) == 4, f"Expected 4 stock jumps in {os.path.basename(path)}, got {len(jumps)}: {jumps}"

    if sonar_norm:
        lob = verify_and_apply_sonar_sec6_transform(lob, verbose=verbose)

    bounds = [0] + jumps + [len(lob)]
    blocks = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        blocks.append((np.ascontiguousarray(lob[s:e], dtype=np.float32), np.ascontiguousarray(y[s:e], dtype=np.int64)))
    return blocks, bounds


def get_perstock_blocks(sonar_norm: bool, verbose: bool = False):
    tr_files = sorted(glob.glob(os.path.join(BASE, "Train_*.txt")))
    te_files = sorted(glob.glob(os.path.join(BASE, "Test_*.txt")))
    assert len(tr_files) == 1 and len(te_files) == 3, "Expected 1 Train and 3 Test FI-2010 files"

    tr_raw_blocks, tr_bounds = load_file_stock_blocks(tr_files[0], sonar_norm=sonar_norm, verbose=verbose)
    train_blocks, val_blocks = [], []
    for idx, (bx, by) in enumerate(tr_raw_blocks):
        sp = int(0.80 * len(bx))
        tr_end = sp - PURGE
        assert tr_end > T_WIN and (len(bx) - sp) > T_WIN, f"Stock block {idx} too short"
        train_blocks.append((bx[:tr_end], by[:tr_end]))
        val_blocks.append((bx[sp:], by[sp:]))

    test_blocks = []
    for tf in te_files:
        blks, _ = load_file_stock_blocks(tf, sonar_norm=sonar_norm, verbose=False)
        test_blocks.extend(blks)

    return train_blocks, val_blocks, test_blocks, tr_bounds


def build_window_datasets(train_blocks, val_blocks, test_blocks, T: int = T_WIN):
    tr_ds = ConcatDataset([WindowArrayDataset(bx, by, T, 1) for bx, by in train_blocks])
    va_ds = ConcatDataset([WindowArrayDataset(bx, by, T, 1) for bx, by in val_blocks])
    te_ds = ConcatDataset([WindowArrayDataset(bx, by, T, 1) for bx, by in test_blocks])
    return tr_ds, va_ds, te_ds


def build_tabular_from_blocks(blocks, T: int = T_WIN, lags=LAGS):
    Xs, Ys = [], []
    for bx, by in blocks:
        ends = np.arange(T - 1, len(bx))
        cols = [bx[ends]] + [bx[ends] - bx[ends - L] for L in lags]
        Xs.append(np.concatenate(cols, axis=1).astype(np.float32))
        Ys.append(by[ends])
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def print_perstock_summary(train_blocks, val_blocks, test_blocks, tr_bounds):
    print("=== [0] PER-STOCK 80/20 SPLIT & BOUNDARY-MASKED WINDOW AUDIT ===", flush=True)
    print(f"  Train_CF_7 Stock Block Boundaries: {tr_bounds}", flush=True)
    for idx, ((trx, _), (vax, _)) in enumerate(zip(train_blocks, val_blocks)):
        print(
            f"    Stock {idx + 1}: Train rows={len(trx):>6,} (windows={len(trx) - T_WIN + 1:>6,}) | "
            f"Purge={PURGE} | Val rows={len(vax):>6,} (windows={len(vax) - T_WIN + 1:>6,})",
            flush=True,
        )
    for name, blks in (("TRAIN (5 stocks)", train_blocks), ("VAL   (5 stocks)", val_blocks), ("TEST (15 blocks)", test_blocks)):
        raw_y = np.concatenate([by for _, by in blks])
        _, win_y = build_tabular_from_blocks(blks, T=T_WIN, lags=())
        rc = np.bincount(raw_y, minlength=3)
        rp = rc / len(raw_y) * 100.0
        wc = np.bincount(win_y, minlength=3)
        wp = wc / len(win_y) * 100.0
        print(
            f"  {name} | Raw Rows: {len(raw_y):>7,} [Up: {rc[0]:>6,} ({rp[0]:4.1f}%), Stat: {rc[1]:>6,} ({rp[1]:4.1f}%), Down: {rc[2]:>6,} ({rp[2]:4.1f}%)]",
            flush=True,
        )
        print(
            f"                   | T=100 Win: {len(win_y):>7,} [Up: {wc[0]:>6,} ({wp[0]:4.1f}%), Stat: {wc[1]:>6,} ({wp[1]:4.1f}%), Down: {wc[2]:>6,} ({wp[2]:4.1f}%)]",
            flush=True,
        )


def forward_model(model, xb):
    try:
        return model(xb)
    except RuntimeError:
        return model(xb.squeeze(1))


@torch.no_grad()
def eval_logits_targets(model, loader, device):
    model.eval()
    logits_list, targets_list = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        out = forward_model(model, xb)
        logits_list.append(out.detach().cpu().numpy())
        targets_list.append(yb.numpy())
    logits = np.concatenate(logits_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    preds = logits.argmax(axis=1)
    acc = float((preds == targets).mean())
    f1 = float(f1_score(targets, preds, average="macro"))
    return acc, f1, logits, targets


def summarize_3seed(label, rows):
    print(f"  --- 3-Seed Summary: {label} (sample std, ddof=1, n={len(rows)}) ---", flush=True)
    for key in ("val_f1", "f1", "acc", "T", "ece_raw", "ece_cal"):
        vals = [r[key] for r in rows]
        print(f"    {label} {key:<7}: {np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}", flush=True)


def run_tabular_suite(train_blocks, val_blocks, test_blocks, dev_str: str):
    Xtr, ytr = build_tabular_from_blocks(train_blocks, T=T_WIN, lags=LAGS)
    Xva, yva = build_tabular_from_blocks(val_blocks, T=T_WIN, lags=LAGS)
    Xte, yte = build_tabular_from_blocks(test_blocks, T=T_WIN, lags=LAGS)

    print("\n=== [1] Majority Class Baseline (Per-Stock Split, T=100) ===", flush=True)
    maj = DummyClassifier(strategy="prior")
    maj.fit(Xtr, ytr)
    yp_maj = maj.predict(Xte)
    print(f"  Majority Class: Test Macro-F1={f1_score(yte, yp_maj, average='macro'):.4f} | Acc={(yp_maj == yte).mean():.4f}", flush=True)
    print(classification_report(yte, yp_maj, target_names=["Up (0)", "Stationary (1)", "Down (2)"], digits=4, zero_division=0), flush=True)
    print("Confusion Matrix:\n", confusion_matrix(yte, yp_maj), flush=True)

    print("\n=== [2] Random Forest (150 trees, balanced, 160 feats, Seeds 42, 43, 44) ===", flush=True)
    rf_rows = []
    for s in SEEDS:
        rf = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=s, n_jobs=16)
        rf.fit(Xtr, ytr)
        lv = np.log(np.clip(rf.predict_proba(Xva), 1e-12, 1.0))
        lt = np.log(np.clip(rf.predict_proba(Xte), 1e-12, 1.0))
        rf_rows.append(calib_report(f"PerStock RF seed {s}", lv, yva, lt, yte, print_full=(s == 42)))
    summarize_3seed("PerStock Random Forest", rf_rows)

    print("\n=== [3] MLP (128, 64, standardized, max_iter=200, 160 feats, Seeds 42, 43, 44) ===", flush=True)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)
    mlp_rows = []
    for s in SEEDS:
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=200, early_stopping=True, validation_fraction=0.1, random_state=s
        )
        mlp.fit(Xtr_s, ytr)
        print(f"  MLP seed {s} converged in n_iter_={mlp.n_iter_} / 200", flush=True)
        lv = np.log(np.clip(mlp.predict_proba(Xva_s), 1e-12, 1.0))
        lt = np.log(np.clip(mlp.predict_proba(Xte_s), 1e-12, 1.0))
        mlp_rows.append(calib_report(f"PerStock MLP seed {s}", lv, yva, lt, yte, print_full=(s == 42)))
    summarize_3seed("PerStock MLP (128, 64)", mlp_rows)

    print("\n=== [4] XGBoost (160 feats, lags 10/50/99, 2000 cap, early_stopping=30, Seeds 42, 43, 44) ===", flush=True)
    xgb_rows = []
    for s in SEEDS:
        clf = xgb.XGBClassifier(
            n_estimators=2000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist", device=dev_str,
            early_stopping_rounds=30, eval_metric="mlogloss", random_state=s,
        )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        print(f"  XGBoost (160 feats) seed {s} best_iteration={clf.best_iteration}", flush=True)
        lv = np.log(np.clip(clf.predict_proba(Xva), 1e-12, 1.0))
        lt = np.log(np.clip(clf.predict_proba(Xte), 1e-12, 1.0))
        xgb_rows.append(calib_report(f"PerStock XGBoost (160f) seed {s}", lv, yva, lt, yte, print_full=(s == 42)))
    summarize_3seed("PerStock XGBoost (160f)", xgb_rows)

    print("\n=== [5] XGBoost Snapshot-Only Ablation (40 feats, LAGS=(), Seeds 42, 43, 44) ===", flush=True)
    Xtr40, ytr40 = build_tabular_from_blocks(train_blocks, T=T_WIN, lags=())
    Xva40, yva40 = build_tabular_from_blocks(val_blocks, T=T_WIN, lags=())
    Xte40, yte40 = build_tabular_from_blocks(test_blocks, T=T_WIN, lags=())
    xgb40_rows = []
    for s in SEEDS:
        clf40 = xgb.XGBClassifier(
            n_estimators=2000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist", device=dev_str,
            early_stopping_rounds=30, eval_metric="mlogloss", random_state=s,
        )
        clf40.fit(Xtr40, ytr40, eval_set=[(Xva40, yva40)], verbose=False)
        print(f"  XGBoost (40 feats, no lags) seed {s} best_iteration={clf40.best_iteration}", flush=True)
        lv40 = np.log(np.clip(clf40.predict_proba(Xva40), 1e-12, 1.0))
        lt40 = np.log(np.clip(clf40.predict_proba(Xte40), 1e-12, 1.0))
        xgb40_rows.append(calib_report(f"PerStock XGBoost (40f) seed {s}", lv40, yva40, lt40, yte40, print_full=(s == 42)))
    summarize_3seed("PerStock XGBoost (40f)", xgb40_rows)
    return rf_rows, mlp_rows, xgb_rows, xgb40_rows


def train_seq_3seed(
    section_title: str,
    short_name: str,
    model_builder,
    lr: float,
    epochs: int,
    train_blocks,
    val_blocks,
    test_blocks,
    device,
    ckpt_filename: str,
):
    print(f"\n=== {section_title} ===", flush=True)
    tr_ds, va_ds, te_ds = build_window_datasets(train_blocks, val_blocks, test_blocks, T=T_WIN)
    va_ld = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    te_ld = DataLoader(te_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    rows = []
    best_overall_val = -1.0
    best_overall_state = None
    best_overall_seed = None
    best_overall_ep = None

    for s in SEEDS:
        set_seed(s)
        g = torch.Generator()
        g.manual_seed(s)
        tr_ld = DataLoader(tr_ds, batch_size=128, shuffle=True, generator=g, num_workers=4, pin_memory=True)

        model = model_builder().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        best_val_f1 = -1.0
        best_state = None
        best_ep = 0

        for ep in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            loss_sum, n_samp = 0.0, 0
            for xb, yb in tr_ld:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = forward_model(model, xb)
                loss = crit(out, yb)
                loss.backward()
                opt.step()
                loss_sum += loss.item() * len(yb)
                n_samp += len(yb)

            va_acc, va_f1, _, _ = eval_logits_targets(model, va_ld, device)
            dt = time.time() - t0
            print(
                f"[{short_name} seed {s}] Ep {ep:02d}/{epochs:02d} | Train Loss: {loss_sum / n_samp:.4f} | "
                f"Val Acc: {va_acc:.4f} | Val Macro-F1: {va_f1:.4f} | {dt:.1f}s",
                flush=True,
            )
            if va_f1 > best_val_f1:
                best_val_f1 = va_f1
                best_state = copy.deepcopy(model.state_dict())
                best_ep = ep

        print(f"--> [{short_name} seed {s}] best Val F1={best_val_f1:.4f} at Ep {best_ep:02d}/{epochs:02d}", flush=True)
        model.load_state_dict(best_state)
        _, _, lv, yva = eval_logits_targets(model, va_ld, device)
        _, _, lt, yte = eval_logits_targets(model, te_ld, device)
        row = calib_report(f"{short_name} seed {s} (ep {best_ep})", lv, yva, lt, yte, print_full=(s == 42))
        row["best_ep"] = best_ep
        rows.append(row)

        if best_val_f1 > best_overall_val:
            best_overall_val = best_val_f1
            best_overall_state = copy.deepcopy(best_state)
            best_overall_seed = s
            best_overall_ep = best_ep

    ckpt_path = os.path.join(BASE, ckpt_filename)
    torch.save(best_overall_state, ckpt_path)
    print(
        f"Saved best-val {short_name} checkpoint (Seed {best_overall_seed}, Ep {best_overall_ep}, Val F1={best_overall_val:.4f}) -> {ckpt_path}",
        flush=True,
    )
    summarize_3seed(short_name, rows)
    return rows


def main():
    t_start = time.time()
    device, dev_name = select_device()
    dev_str = str(device) if device.type == "cuda" else "cpu"
    print(f"Device: {device} ({dev_name})", flush=True)

    son_tr, son_va, son_te, tr_bounds = get_perstock_blocks(sonar_norm=True, verbose=True)
    raw_tr, raw_va, raw_te, _ = get_perstock_blocks(sonar_norm=False, verbose=True)

    print_perstock_summary(son_tr, son_va, son_te, tr_bounds)

    # 1. Tabular baselines (Majority, RF, MLP, XGBoost 160f, XGBoost 40f ablation)
    rf_rows, mlp_rows, xgb_rows, xgb40_rows = run_tabular_suite(son_tr, son_va, son_te, dev_str)

    # 2. LSTM-Attention v3 (unscaled sonar_norm, 25 epochs, 3 seeds)
    lstm_rows = train_seq_3seed(
        section_title="[6] LSTM-Attention v3 (Per-Stock Split, unscaled sonar_norm, T=100, 25 ep, Seeds 42, 43, 44)",
        short_name="PerStock LSTM-Att v3",
        model_builder=lambda: LSTMAttention(),
        lr=1e-3,
        epochs=25,
        train_blocks=son_tr,
        val_blocks=son_va,
        test_blocks=son_te,
        device=device,
        ckpt_filename="perstock_lstm_attention_v3_sonar_norm_T100_k100.pt",
    )

    # 3. DeepLOB sonar_norm T=100 (25 epochs, 3 seeds)
    dl_son_rows = train_seq_3seed(
        section_title="[7] DeepLOB sonar_norm T=100 (Per-Stock Split, 25 ep, Seeds 42, 43, 44)",
        short_name="PerStock DeepLOB sonar_norm",
        model_builder=lambda: DeepLOB(3),
        lr=1e-4,
        epochs=25,
        train_blocks=son_tr,
        val_blocks=son_va,
        test_blocks=son_te,
        device=device,
        ckpt_filename="perstock_deeplob_sonar_norm_T100_k100.pt",
    )

    # 4. DeepLOB decpre T=100 (25 epochs, 3 seeds)
    dl_raw_rows = train_seq_3seed(
        section_title="[8] DeepLOB decpre T=100 (Per-Stock Split, 25 ep, Seeds 42, 43, 44)",
        short_name="PerStock DeepLOB decpre",
        model_builder=lambda: DeepLOB(3),
        lr=1e-4,
        epochs=25,
        train_blocks=raw_tr,
        val_blocks=raw_va,
        test_blocks=raw_te,
        device=device,
        ckpt_filename="perstock_deeplob_decpre_T100_k100.pt",
    )

    print("\n================ FINAL PER-STOCK 3-SEED MASTER SUMMARY (ddof=1, n=3) ================", flush=True)
    for label, rlist in [
        ("XGBoost (160f, lags 10/50/99)", xgb_rows),
        ("XGBoost Ablation (40f, no lags)", xgb40_rows),
        ("Random Forest (150 trees)", rf_rows),
        ("MLP (128, 64, standardized)", mlp_rows),
        ("LSTM-Attention v3 (unscaled, 25ep)", lstm_rows),
        ("DeepLOB sonar_norm (T=100, 25ep)", dl_son_rows),
        ("DeepLOB decpre (T=100, 25ep)", dl_raw_rows),
    ]:
        vf1 = [r["val_f1"] for r in rlist]
        tf1 = [r["f1"] for r in rlist]
        acc = [r["acc"] for r in rlist]
        er = [r["ece_raw"] for r in rlist]
        ec = [r["ece_cal"] for r in rlist]
        print(
            f"{label:<36} | Val F1: {np.mean(vf1):.4f}+/-{np.std(vf1, ddof=1):.4f} "
            f"| Test F1: {np.mean(tf1):.4f}+/-{np.std(tf1, ddof=1):.4f} "
            f"| Acc: {np.mean(acc):.4f}+/-{np.std(acc, ddof=1):.4f} "
            f"| ECE: {np.mean(er):.4f}->{np.mean(ec):.4f}",
            flush=True,
        )
    print(f"\nALL PER-STOCK RUNS COMPLETE IN {(time.time() - t_start) / 60.0:.1f} MINUTES.", flush=True)


if __name__ == "__main__":
    main()
