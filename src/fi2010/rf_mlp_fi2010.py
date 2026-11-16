import numpy as np
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.fi2010.train_fi2010_deeplob import select_device
from src.fi2010.baselines_fi2010 import calib_report, get_arrays, tab_feats

SEEDS = (42, 43, 44)

def print_split_stats(arr, T=100):
    print("=== [1] FI-2010 PURGED SPLIT ROW & WINDOW LABEL BALANCE (k=100) ===", flush=True)
    for split_name in ("train", "val", "test"):
        x = arr[f"x_{split_name}"]
        y = arr[f"y_{split_name}"]
        raw_cnt = np.bincount(y, minlength=3)
        raw_pct = raw_cnt / len(y) * 100.0
        _, y_win = tab_feats(x, y, T=T)
        win_cnt = np.bincount(y_win, minlength=3)
        win_pct = win_cnt / len(y_win) * 100.0
        print(
            f"  {split_name.upper():<5} | Raw Rows: {len(y):>7,} "
            f"[Up(0): {raw_cnt[0]:>6,} ({raw_pct[0]:4.1f}%), Stat(1): {raw_cnt[1]:>6,} ({raw_pct[1]:4.1f}%), Down(2): {raw_cnt[2]:>6,} ({raw_pct[2]:4.1f}%)]",
            flush=True,
        )
        print(
            f"        | T={T} Win: {len(y_win):>7,} "
            f"[Up(0): {win_cnt[0]:>6,} ({win_pct[0]:4.1f}%), Stat(1): {win_cnt[1]:>6,} ({win_pct[1]:4.1f}%), Down(2): {win_cnt[2]:>6,} ({win_pct[2]:4.1f}%)]",
            flush=True,
        )

def summarize_3seed(label, rows):
    for key in ("val_f1", "f1", "acc", "ece_raw", "ece_cal"):
        vals = [r[key] for r in rows]
        print(f"  {label} {key:<7}: {np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f} (ddof=1, n={len(vals)})", flush=True)

def main():
    device, dev_name = select_device()
    dev_str = str(device) if device.type == "cuda" else "cpu"
    print(f"Device: {device} ({dev_name})", flush=True)

    son = get_arrays(True)
    print_split_stats(son, T=100)

    Xtr, ytr = tab_feats(son["x_train"], son["y_train"], T=100)
    Xva, yva = tab_feats(son["x_val"], son["y_val"], T=100)
    Xte, yte = tab_feats(son["x_test"], son["y_test"], T=100)

    print("\n=== [2] Majority Class Baseline (FI-2010, T=100) ===", flush=True)
    maj = DummyClassifier(strategy="prior")
    maj.fit(Xtr, ytr)
    yp_maj = maj.predict(Xte)
    maj_f1 = float(f1_score(yte, yp_maj, average="macro"))
    maj_acc = float((yp_maj == yte).mean())
    print(f"  Majority Class: Test Macro-F1={maj_f1:.4f} | Acc={maj_acc:.4f}", flush=True)

    print("\n=== [3] Random Forest (150 trees, balanced, Seeds 42, 43, 44) ===", flush=True)
    rf_rows = []
    for s in SEEDS:
        rf = RandomForestClassifier(
            n_estimators=150, class_weight="balanced", random_state=s, n_jobs=16
        )
        rf.fit(Xtr, ytr)
        lv = np.log(np.clip(rf.predict_proba(Xva), 1e-12, 1.0))
        lt = np.log(np.clip(rf.predict_proba(Xte), 1e-12, 1.0))
        rf_rows.append(calib_report(f"Random Forest seed {s}", lv, yva, lt, yte, print_full=(s == 42)))
    summarize_3seed("Random Forest", rf_rows)

    print("\n=== [4] MLP (128, 64, standardized, max_iter=200, Seeds 42, 43, 44) ===", flush=True)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)
    mlp_rows = []
    for s in SEEDS:
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=s,
        )
        mlp.fit(Xtr_s, ytr)
        print(f"  MLP seed {s} converged in n_iter_={mlp.n_iter_} / 200", flush=True)
        lv = np.log(np.clip(mlp.predict_proba(Xva_s), 1e-12, 1.0))
        lt = np.log(np.clip(mlp.predict_proba(Xte_s), 1e-12, 1.0))
        mlp_rows.append(calib_report(f"MLP (128, 64) seed {s}", lv, yva, lt, yte, print_full=(s == 42)))
    summarize_3seed("MLP (128, 64)", mlp_rows)

    print("\n=== [5] XGBoost (2000 cap, early_stopping=30, Seeds 42, 43, 44) ===", flush=True)
    xgb_rows = []
    for s in SEEDS:
        clf = xgb.XGBClassifier(
            n_estimators=2000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist", device=dev_str,
            early_stopping_rounds=30, eval_metric="mlogloss", random_state=s,
        )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        print(f"  XGBoost seed {s} best_iteration={clf.best_iteration}", flush=True)
        lv = np.log(np.clip(clf.predict_proba(Xva), 1e-12, 1.0))
        lt = np.log(np.clip(clf.predict_proba(Xte), 1e-12, 1.0))
        xgb_rows.append(calib_report(f"XGBoost seed {s}", lv, yva, lt, yte, print_full=(s == 42)))
    summarize_3seed("XGBoost", xgb_rows)

if __name__ == "__main__":
    main()
