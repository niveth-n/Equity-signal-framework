import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from .features import add_basic_features
from .model import make_xy, train_model, FEATURES

# ----------------------- data -----------------------

def make_synth(n_assets=12, start="2024-01-02", end="2025-06-30", seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    frames = []
    for i in range(n_assets):
        mu = rng.normal(0.0002, 0.00015)
        vol = rng.uniform(0.008, 0.02)
        r = rng.normal(mu, vol, len(dates))
        px = 100 * (1 + pd.Series(r, index=dates)).cumprod().values
        frames.append(pd.DataFrame({"date": dates, "asset": f"A{i:02d}", "close": px}))
    return pd.concat(frames, ignore_index=True)

def load_or_synth(start: str, end: str, seed: int) -> pd.DataFrame:
    """If data/prices.csv exists (date,asset,close), use it; else generate synthetic."""
    csv = Path("data/prices.csv")
    if csv.exists():
        df = pd.read_csv(csv, parse_dates=["date"])
        mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
        return df.loc[mask, ["date", "asset", "close"]].copy()
    return make_synth(start=start, end=end, seed=seed)

# --------------------- utils -----------------------

def ensure_dirs():
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("evidence").mkdir(parents=True, exist_ok=True)

def max_drawdown(curve: pd.Series) -> float:
    peak = curve.cummax()
    dd = (curve / peak - 1.0).min()
    return float(abs(dd) * 100.0)

def save_plots(eq: pd.Series, eq_bh: pd.Series, y_true, y_pred):
    plt.figure(figsize=(8, 4))
    eq.plot(label="Strategy")
    eq_bh.plot(label="Buy&Hold")
    plt.legend(); plt.title("Equity Curve (OOS)")
    plt.tight_layout(); plt.savefig("results/plots/equity_curve.png"); plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (OOS)")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout(); plt.savefig("results/plots/confusion_matrix.png"); plt.close()

# ---------------------- core -----------------------

def run(cfg_path: str = "evidence/backtest_config.yaml"):
    ensure_dirs()
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    seed = int(cfg.get("seed", 42))
    fees_bps = float(cfg.get("fees_bps", 2))
    slippage_bps = float(cfg.get("slippage_bps", 5))
    top_k = int(cfg.get("top_k", 5))

    oos_start = pd.to_datetime(cfg["oos_start"])
    oos_end = pd.to_datetime(cfg["oos_end"])

    # Training history: ~1y before OOS
    start_all = (oos_start - pd.offsets.BDay(260)).date()
    df = load_or_synth(str(start_all), str(oos_end.date()), seed=seed)

    feats = add_basic_features(df)

    train = feats[feats["date"] < oos_start].dropna(subset=FEATURES + ["y"])
    test = feats[(feats["date"] >= oos_start) & (feats["date"] <= oos_end)].dropna(subset=FEATURES + ["y"])

    X_tr, y_tr = make_xy(train)
    X_te, y_te = make_xy(test)

    model = train_model(X_tr, y_tr, seed=seed)
    proba = pd.Series(model.predict_proba(X_te)[:, 1], index=test.index, name="proba")
    test = test.assign(proba=proba.values)

    # daily long top-k by probability
    test["rank"] = test.groupby("date")["proba"].rank(ascending=False, method="first")
    test["w"] = (test["rank"] <= top_k).astype(float) / top_k

    w = test.pivot(index="date", columns="asset", values="w").fillna(0.0)
    fwd = test.pivot(index="date", columns="asset", values="fwd_ret").fillna(0.0)

    gross = (w * fwd).sum(axis=1)

    # simple costs
    bps_cost = (fees_bps + slippage_bps) / 1e4
    turnover = w.diff().abs().sum(axis=1) / 2.0
    cost = turnover * bps_cost
    net = gross - cost

    # benchmark: equal-weight buy & hold
    bh = fwd.mean(axis=1)

    # panel classification metrics
    y_hat = (test["proba"] > 0.5).astype(int)
    acc = accuracy_score(y_te, y_hat)
    prec = precision_score(y_te, y_hat, zero_division=0)
    rec = recall_score(y_te, y_hat, zero_division=0)
    f1 = f1_score(y_te, y_hat, zero_division=0)

    # strategy metrics
    eq = (1 + net).cumprod()
    eq_bh = (1 + bh).cumprod()
    sharpe = float(np.sqrt(252) * (net.mean() / (net.std(ddof=1) + 1e-12)))
    mdd = max_drawdown(eq)
    strat_ret = float((eq.iloc[-1] - 1) * 100)
    bh_ret = float((eq_bh.iloc[-1] - 1) * 100)
    avg_turnover = float(turnover.mean())

    save_plots(eq, eq_bh, y_te, y_hat)

    out = pd.DataFrame([{
        "start_date": str(oos_start.date()),
        "end_date": str(oos_end.date()),
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(mdd, 2),
        "strat_return": round(strat_ret, 2),
        "buyhold_return": round(bh_ret, 2),
        "turnover": round(avg_turnover, 2),
    }])
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/oos_metrics.csv", index=False)
    print("\nOOS metrics\n", out.to_string(index=False))

# ---------------------- cli -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="evidence/backtest_config.yaml")
    args = ap.parse_args()
    run(args.config)

if __name__ == "__main__":
    main()

