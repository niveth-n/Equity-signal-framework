from typing import Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES = ["ret_1", "mom_5", "vol_20", "ma_ratio_10", "mom_5_cs"]

def make_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES].fillna(0.0)
    y = df["y"].astype(int)
    return X, y

def train_model(X, y, seed: int = 42):
    clf = LogisticRegression(max_iter=400, C=1.0, solver="lbfgs", random_state=seed)
    clf.fit(X, y)
    return clf
