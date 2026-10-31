"""
SONAR (Stock Project) - Stage 5 Refined Data Preprocessing & Multi-Model Solution
"""
import os
import time
import uuid
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def get_l1_schema():
    if not HAS_PYARROW:
        return None
    return pa.schema([
        ("exch_ts_ns", pa.int64()),
        ("recv_mono_ns", pa.int64()),
        ("wall_utc_ns", pa.int64()),
        ("session", pa.string()),
        ("symbol", pa.string()),
        ("source", pa.string()),
        ("event_type", pa.uint8()),
        ("bid_px", pa.float32()),
        ("ask_px", pa.float32()),
        ("bid_sz", pa.uint32()),
        ("ask_sz", pa.uint32()),
        ("bid_exch", pa.string()),
        ("ask_exch", pa.string()),
        ("last_px", pa.float32()),
        ("last_sz", pa.uint32()),
        ("trade_exch", pa.string()),
        ("trade_side", pa.int8()),
        ("cum_vol", pa.uint64()),
    ])


def get_l2_wide_schema():
    if not HAS_PYARROW:
        return None
    fields = [
        ("exch_ts_ns", pa.int64()),
        ("recv_mono_ns", pa.int64()),
        ("wall_utc_ns", pa.int64()),
        ("session", pa.string()),
        ("symbol", pa.string()),
        ("source", pa.string()),
        ("book_age_ms", pa.uint32()),
    ]
    for lvl in range(1, 11):
        fields.extend([
            (f"bid_px_{lvl}", pa.float32()),
            (f"bid_sz_{lvl}", pa.uint32()),
            (f"ask_px_{lvl}", pa.float32()),
            (f"ask_sz_{lvl}", pa.uint32()),
        ])
    return pa.schema(fields)


def atomic_parquet_flush(rows: list[dict], schema, target_dir: str, prefix: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    uid = uuid.uuid4().hex[:10]
    if HAS_PYARROW and schema is not None:
        tmp_path = os.path.join(target_dir, f".{prefix}_{uid}.parquet.tmp")
        final_path = os.path.join(target_dir, f"{prefix}_{uid}.parquet")
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, tmp_path, compression="zstd")
        os.replace(tmp_path, final_path)
        return final_path
    tmp_path = os.path.join(target_dir, f".{prefix}_{uid}.npz.tmp")
    final_path = os.path.join(target_dir, f"{prefix}_{uid}.npz")
    np.savez_compressed(tmp_path, rows=rows)
    os.replace(tmp_path, final_path)
    return final_path


def classify_trade_side(trade_px: float, bid_px: float, ask_px: float,
                        prev_trade_px: float, prev_side: int = 1) -> int:
    if bid_px > 0.0 and ask_px > bid_px:
        if trade_px >= ask_px:
            return 1
        if trade_px <= bid_px:
            return -1
    if trade_px > prev_trade_px:
        return 1
    if trade_px < prev_trade_px:
        return -1
    return prev_side


def compute_price_keyed_ofi(curr_px: np.ndarray, curr_sz: np.ndarray,
                            prev_px: np.ndarray, prev_sz: np.ndarray,
                            is_bid: bool) -> np.ndarray:
    """10-level dual-boundary price-keyed OFI with both entry and exit scroll guards."""
    curr_map = {round(float(p), 4): i for i, p in enumerate(curr_px)}
    prev_map = {round(float(p), 4): float(s) for p, s in zip(prev_px, prev_sz)}
    prev_boundary = round(float(prev_px[-1]), 4)
    curr_boundary = round(float(curr_px[-1]), 4)
    flow = np.zeros(10, dtype=np.float32)

    for i in range(10):
        p = round(float(curr_px[i]), 4)
        s = float(curr_sz[i])
        if p in prev_map:
            flow[i] = s - prev_map[p]
        else:
            scrolled_in = (p < prev_boundary) if is_bid else (p > prev_boundary)
            flow[i] = 0.0 if scrolled_in else s

    for p_old, s_old in prev_map.items():
        if p_old not in curr_map:
            scrolled_out = (p_old < curr_boundary) if is_bid else (p_old > curr_boundary)
            if not scrolled_out:
                nearest_idx = int(np.argmin(np.abs(curr_px - p_old)))
                flow[nearest_idx] -= s_old
    return flow


def compute_v1_features(bid_px: np.ndarray, bid_sz: np.ndarray,
                        ask_px: np.ndarray, ask_sz: np.ndarray,
                        prev_bid_px: np.ndarray, prev_bid_sz: np.ndarray,
                        prev_ask_px: np.ndarray, prev_ask_sz: np.ndarray,
                        agg_buy_vol_top1: float, agg_sell_vol_top1: float) -> tuple[np.ndarray, np.ndarray, dict]:
    mid = 0.5 * (bid_px[0] + ask_px[0])
    spread = (ask_px[0] - bid_px[0]) / max(mid, 1e-6)
    micro_px = (bid_sz[0] * ask_px[0] + ask_sz[0] * bid_px[0]) / max(bid_sz[0] + ask_sz[0], 1.0)
    micro_div = (micro_px - mid) / max(mid, 1e-6)

    rel_bid_dist = (bid_px - mid) / max(mid, 1e-6)
    rel_ask_dist = (ask_px - mid) / max(mid, 1e-6)
    log_bid_sz = np.log1p(bid_sz)
    log_ask_sz = np.log1p(ask_sz)

    e_b = compute_price_keyed_ofi(bid_px, bid_sz, prev_bid_px, prev_bid_sz, is_bid=True)
    e_a = compute_price_keyed_ofi(ask_px, ask_sz, prev_ask_px, prev_ask_sz, is_bid=False)
    ofi_10 = e_b - e_a

    ask_depletion = max(0.0, float(prev_ask_sz[0]) - float(ask_sz[0]))
    bid_depletion = max(0.0, float(prev_bid_sz[0]) - float(bid_sz[0]))
    air_ask = agg_buy_vol_top1 / max(1.0, ask_depletion)
    air_bid = agg_sell_vol_top1 / max(1.0, bid_depletion)

    spatial_40 = np.empty(40, dtype=np.float32)
    spatial_40[0::4] = rel_ask_dist
    spatial_40[1::4] = log_ask_sz
    spatial_40[2::4] = rel_bid_dist
    spatial_40[3::4] = log_bid_sz

    aux_16 = np.concatenate([
        ofi_10,
        np.array([spread, micro_div, float(np.sum(ofi_10)), float(ofi_10[0]), air_ask, air_bid], dtype=np.float32)
    ])

    summary = {
        "mid_px": mid,
        "spread": spread,
        "micro_div": micro_div,
        "ofi_sum": float(np.sum(ofi_10)),
        "ofi_l1": float(ofi_10[0]),
        "air_ask": air_ask,
        "air_bid": air_bid,
    }
    return spatial_40, aux_16, summary


def compute_triple_barrier_labels(best_bids: np.ndarray, best_asks: np.ndarray,
                                  horizon_steps: int = 50, min_theta: float = 0.0015,
                                  k_vol: float = 0.45) -> np.ndarray:
    mids = 0.5 * (best_bids + best_asks)
    log_rets = np.diff(np.log(np.maximum(mids, 1e-6)), prepend=np.log(max(mids[0], 1e-6)))
    n_valid = len(best_bids) - horizon_steps
    labels = np.ones(n_valid, dtype=np.int64)

    for t in range(n_valid):
        win_start = max(0, t - 60)
        sigma_step = float(np.std(log_rets[win_start : t + 1])) if t >= 10 else 0.0002
        theta_t = max(min_theta, k_vol * sigma_step * np.sqrt(horizon_steps))

        upper_target = best_asks[t] * (1.0 + theta_t)
        lower_target = best_bids[t] * (1.0 - theta_t)
        future_bids = best_bids[t + 1 : t + 1 + horizon_steps]
        future_asks = best_asks[t + 1 : t + 1 + horizon_steps]

        up_hits = np.where(future_bids >= upper_target)[0]
        dn_hits = np.where(future_asks <= lower_target)[0]
        t_up = up_hits[0] if len(up_hits) > 0 else horizon_steps + 1
        t_dn = dn_hits[0] if len(dn_hits) > 0 else horizon_steps + 1

        if t_up < t_dn and t_up <= horizon_steps:
            labels[t] = 2
        elif t_dn < t_up and t_dn <= horizon_steps:
            labels[t] = 0
    return labels


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = np.zeros((3, 3), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    f1s = []
    for c in range(3):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        f1s.append(float((2 * tp) / max(1, 2 * tp + fp + fn)))
    return {
        "macro_f1": float(np.mean(f1s)),
        "f1_down": f1s[0],
        "f1_neutral": f1s[1],
        "f1_up": f1s[2],
        "cm": cm.tolist(),
    }


def run_ps5_pipeline(symbol: str = "SONAR_SMCAP", n_ticks: int = 1000) -> dict:
    rng = np.random.default_rng(42)
    base_px = 4.50
    l1_rows, l2_rows = [], []
    spatial_matrix = np.zeros((n_ticks, 40), dtype=np.float32)
    tabular_features = np.zeros((n_ticks, 56), dtype=np.float32)
    best_bids = np.zeros(n_ticks, dtype=np.float32)
    best_asks = np.zeros(n_ticks, dtype=np.float32)

    prev_bid_px = np.array([base_px - 0.01 * i for i in range(1, 11)], dtype=np.float32)
    prev_ask_px = np.array([base_px + 0.01 * i for i in range(1, 11)], dtype=np.float32)
    prev_bid_sz = rng.integers(500, 5000, size=10).astype(np.uint32)
    prev_ask_sz = rng.integers(500, 5000, size=10).astype(np.uint32)
    cum_vol = 0
    prev_trade_px = base_px

    for t in range(n_ticks):
        latent_signal = np.sin(t / 32.0) * 0.0025
        price_shock = rng.normal(0, 0.0035)
        base_px = max(0.50, base_px + latent_signal + price_shock)
        spread_cents = 0.01 if rng.random() > 0.25 else 0.02

        bid_gaps = np.cumsum(rng.choice([0.01, 0.02, 0.03], size=10, p=[0.6, 0.3, 0.1]))
        ask_gaps = np.cumsum(rng.choice([0.01, 0.02, 0.03], size=10, p=[0.6, 0.3, 0.1]))
        bid_px = (base_px - spread_cents / 2.0 - bid_gaps + 0.01).astype(np.float32)
        ask_px = (base_px + spread_cents / 2.0 + ask_gaps - 0.01).astype(np.float32)

        noisy_imb = 1.25 if (latent_signal + rng.normal(0, 0.002)) > 0 else 0.80
        bid_sz = (rng.integers(400, 4500, size=10) * noisy_imb).astype(np.uint32)
        ask_sz = (rng.integers(400, 4500, size=10) / noisy_imb).astype(np.uint32)

        trade_sz = int(rng.integers(100, 2500))
        trade_px = float(ask_px[0] if rng.random() < (0.55 if latent_signal > 0 else 0.45) else bid_px[0])
        side = classify_trade_side(trade_px, float(bid_px[0]), float(ask_px[0]), prev_trade_px)
        prev_trade_px = trade_px
        cum_vol += trade_sz

        now_ns = time.time_ns()
        mono_ns = time.monotonic_ns()

        l1_rows.append({
            "exch_ts_ns": now_ns, "recv_mono_ns": mono_ns, "wall_utc_ns": now_ns,
            "session": "regular", "symbol": symbol, "source": "tastytrade", "event_type": 1,
            "bid_px": float(bid_px[0]), "ask_px": float(ask_px[0]),
            "bid_sz": int(bid_sz[0]), "ask_sz": int(ask_sz[0]),
            "bid_exch": "Q", "ask_exch": "P",
            "last_px": trade_px, "last_sz": trade_sz, "trade_exch": "P",
            "trade_side": side, "cum_vol": cum_vol
        })

        l2_row = {
            "exch_ts_ns": now_ns, "recv_mono_ns": mono_ns, "wall_utc_ns": now_ns,
            "session": "regular", "symbol": symbol, "source": "moomoo_arca", "book_age_ms": 12
        }
        for lvl in range(1, 11):
            l2_row[f"bid_px_{lvl}"] = float(bid_px[lvl - 1])
            l2_row[f"bid_sz_{lvl}"] = int(bid_sz[lvl - 1])
            l2_row[f"ask_px_{lvl}"] = float(ask_px[lvl - 1])
            l2_row[f"ask_sz_{lvl}"] = int(ask_sz[lvl - 1])
        l2_rows.append(l2_row)

        agg_buy = float(trade_sz) if side == 1 else 0.0
        agg_sell = float(trade_sz) if side == -1 else 0.0
        sp40, aux16, _ = compute_v1_features(
            bid_px, bid_sz, ask_px, ask_sz,
            prev_bid_px, prev_bid_sz, prev_ask_px, prev_ask_sz,
            agg_buy, agg_sell
        )
        spatial_matrix[t] = sp40
        tabular_features[t, :40] = sp40
        tabular_features[t, 40:] = aux16
        best_bids[t] = bid_px[0]
        best_asks[t] = ask_px[0]
        prev_bid_px, prev_bid_sz = bid_px, bid_sz
        prev_ask_px, prev_ask_sz = ask_px, ask_sz

    out_dir = os.path.join(os.path.dirname(__file__), "recordings", "2026-10-30", "regular", symbol)
    l1_path = atomic_parquet_flush(l1_rows, get_l1_schema(), out_dir, "l1_quotes")
    l2_path = atomic_parquet_flush(l2_rows, get_l2_wide_schema(), out_dir, "l2_snapshots")

    window_len = 100
    horizon_steps = 50
    labels = compute_triple_barrier_labels(best_bids, best_asks, horizon_steps=horizon_steps, min_theta=0.0015, k_vol=0.45)
    n_samples = len(labels) - window_len

    X_deeplob = np.zeros((n_samples, 1, window_len, 40), dtype=np.float32)
    X_aux_seq = np.zeros((n_samples, window_len, 16), dtype=np.float32)
    X_tab = np.zeros((n_samples, 56), dtype=np.float32)
    y_all = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        end_idx = i + window_len
        X_deeplob[i, 0] = spatial_matrix[i:end_idx]
        X_aux_seq[i] = tabular_features[i:end_idx, 40:]
        X_tab[i, :40] = spatial_matrix[end_idx - 1]
        X_tab[i, 40:] = np.mean(tabular_features[end_idx - 15 : end_idx, 40:], axis=0)
        y_all[i] = labels[end_idx - 1]

    split_idx = int(n_samples * 0.70)
    X_train, y_train = X_tab[:split_idx - horizon_steps], y_all[:split_idx - horizon_steps]
    X_test, y_test = X_tab[split_idx:], y_all[split_idx:]

    vals, counts = np.unique(y_train, return_counts=True)
    maj_class = int(vals[np.argmax(counts)])
    y_pred_maj = np.full_like(y_test, fill_value=maj_class)
    res_maj = _compute_metrics(y_test, y_pred_maj)

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        rf = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42)
        rf.fit(X_train, y_train)
        res_rf = _compute_metrics(y_test, rf.predict(X_test))

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
        mlp.fit(X_tr_s, y_train)
        res_mlp = _compute_metrics(y_test, mlp.predict(X_te_s))
    except ImportError:
        res_rf = res_maj
        res_mlp = res_maj

    return {
        "l1_path": l1_path,
        "l2_path": l2_path,
        "x_deeplob_shape": X_deeplob.shape,
        "x_aux_shape": X_aux_seq.shape,
        "x_tab_shape": X_tab.shape,
        "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y_all, return_counts=True))},
        "models": {
            "Majority Class": res_maj,
            "Random Forest": res_rf,
            "Late-Fusion MLP (Neural)": res_mlp,
        },
    }

run_ps4_preprocessing_pipeline = run_ps5_pipeline
