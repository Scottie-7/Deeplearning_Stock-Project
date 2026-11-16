import glob
import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from src.fi2010.train_fi2010_deeplob import (
    load_fi2010_arrays,
    WindowArrayDataset,
    DeepLOB,
    select_device,
    evaluate_loader,
)

def parse_log_details(log_path: str):
    text = open(log_path).read()
    matches = re.findall(
        r"--> \[Seed (\d+) Held-Out Test \(selected by Val F1=([0-9.]+)\)\] Test Acc: ([0-9.]+) \| Test Macro-F1: ([0-9.]+)",
        text,
    )
    assert len(matches) == 3, f"Expected 3 seed lines in {log_path}, found {len(matches)}"
    best = max(matches, key=lambda m: float(m[1]))
    accs = [float(m[2]) for m in matches]
    f1s = [float(m[3]) for m in matches]
    sum_match = re.search(
        r"(=== SUMMARY ACROSS 3 SEEDS.*?Held-Out Test Accuracy\s*:\s*[0-9.]+ \+/- [0-9.]+\nHeld-Out Test Macro-F1\s*:\s*[0-9.]+ \+/- [0-9.]+)",
        text,
        re.DOTALL,
    )
    assert sum_match is not None, f"Missing SUMMARY block in {log_path}"
    cm_match = re.search(r"Confusion Matrix:\n(\s*\[\[.*?\]\])", text, re.DOTALL)
    assert cm_match is not None, f"Missing Confusion Matrix in {log_path}"
    return best, accs, f1s, sum_match.group(1).strip(), cm_match.group(1).strip()

def main():
    device, dev_name = select_device()
    base = os.path.expanduser("~/sonar_track_c/data/fi2010")
    log_dir = os.path.expanduser("~/sonar_track_c/logs")
    tr_f = sorted(glob.glob(os.path.join(base, "Train_*.txt")))
    te_f = sorted(glob.glob(os.path.join(base, "Test_*.txt")))

    configs = [
        ("decpre", 100, False, "t100_decpre_25ep_purged.log"),
        ("sonar_norm", 100, True, "t100_sonar_norm_25ep_purged.log"),
        ("sonar_norm", 600, True, "t600_sonar_norm_purged.log"),
    ]

    for tag, T, sonar_norm, log_name in configs:
        pt_path = os.path.join(base, f"deeplob_fi2010_{tag}_T{T}_k100.pt")
        log_path = os.path.join(log_dir, log_name)
        assert os.path.exists(pt_path) and os.path.getsize(pt_path) > 0, f"Missing/empty {pt_path}"

        x_tr, y_tr = load_fi2010_arrays(tr_f, horizon_k=100, sonar_norm=sonar_norm, verbose=False)
        x_te, y_te = load_fi2010_arrays(te_f, horizon_k=100, sonar_norm=sonar_norm, verbose=False)
        split_idx = int(0.80 * len(x_tr))

        tr_ds = WindowArrayDataset(x_tr[:split_idx - 100], y_tr[:split_idx - 100], T, 1)
        val_ds = WindowArrayDataset(x_tr[split_idx:], y_tr[split_idx:], T, 1)
        te_ds = WindowArrayDataset(x_te, y_te, T, 1)

        val_ld = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        te_ld = DataLoader(te_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

        model = DeepLOB(3).to(device)
        model.load_state_dict(torch.load(pt_path, map_location=device))
        val_acc, val_f1, _, _ = evaluate_loader(model, val_ld, device, True)
        te_acc, te_f1, yt, yp = evaluate_loader(model, te_ld, device, True)

        (seed_s, log_vf1, log_tacc, log_tf1), accs, f1s, sum_str, log_cm_str = parse_log_details(log_path)
        cm_str = str(confusion_matrix(yt, yp)).strip()

        print(f"\n=== [EVAL VERIFICATION: {os.path.basename(pt_path)} | Log: {log_name} (Best-Val Seed {seed_s})] ===")
        print(f"Purged Counts -> Train: {len(tr_ds):,} | Val: {len(val_ds):,} | Test: {len(te_ds):,}")
        print(sum_str)
        print(f"Recomputed ddof=1 Sample Std -> Acc: {np.mean(accs):.4f} +/- {np.std(accs, ddof=1):.4f} | Macro-F1: {np.mean(f1s):.4f} +/- {np.std(f1s, ddof=1):.4f}")
        print(f"Checkpoint Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f} (Log: {log_vf1})")
        print(f"Held-Out Test Acc : {te_acc:.4f} (Log: {log_tacc}) | Test Macro-F1: {te_f1:.4f} (Log: {log_tf1})")

        assert f"{val_f1:.4f}" == log_vf1, f"Val F1 mismatch: {val_f1:.4f} != {log_vf1}"
        assert f"{te_acc:.4f}" == log_tacc, f"Test Acc mismatch: {te_acc:.4f} != {log_tacc}"
        assert f"{te_f1:.4f}" == log_tf1, f"Test F1 mismatch: {te_f1:.4f} != {log_tf1}"
        assert cm_str == log_cm_str, f"Confusion matrix mismatch!\nEval:\n{cm_str}\nLog:\n{log_cm_str}"
        print("ASSERTION PASSED: .pt checkpoint & Confusion Matrix match training log 100%.")
        print(classification_report(yt, yp, target_names=["Up (0)", "Stationary (1)", "Down (2)"], digits=4))
        print("Confusion Matrix:\n", cm_str)

if __name__ == "__main__":
    main()
