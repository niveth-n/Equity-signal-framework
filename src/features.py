import pandas as pd
import numpy as np

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: ['date','asset','close'] (business days).
    Returns df with simple features + targets: y (next-day up/down), fwd_ret.
    """
    x = df.sort_values(["asset", "date"]).copy()
    g = x.groupby("asset", group_keys=False)

    r1 = g["close"].pct_change()
    x["ret_1"] = r1
    x["mom_5"] = g["close"].pct_change(5)
    x["vol_20"] = r1.rolling(20).std()
    x["ma_ratio_10"] = x["close"] / g["close"].transform(lambda s: s.rolling(10).mean())

    # cross-sectional z-score of 5-day momentum per date
    x["mom_5_cs"] = x.groupby("date")["mom_5"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    )

    # targets
    x["fwd_ret"] = g["close"].pct_change(-1)
    x["y"] = (x["fwd_ret"] > 0).astype(int)

    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    return x
