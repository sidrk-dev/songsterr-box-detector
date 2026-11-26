"""Rung-probability model: P(index at S > strike | features), monotone in strike distance k.

A single gradient-boosted classifier is trained on (event, rung) rows. Because the
relative strike k = strike - i0 is an input with a monotone-decreasing constraint, the
model outputs a valid (non-increasing) ladder of probabilities for every event - a
generalisation of the article's "ladder model" that permits skewed distributions.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

FEATURES = ["k", "city", "lead", "di1", "di2", "di3", "di6", "yday_delta", "yday_level",
            "metar_minus_i0", "metar_age", "metar_trend", "m_spread", "m_n", "m_dewdep", "m_wind",
            "m_sky", "m_ceil", "m_precip", "sky_trend", "sun_s", "sun_d", "hour_sin", "hour_cos", "doy_sin"]

PARAMS = dict(n_estimators=400, max_depth=5, learning_rate=0.04, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=20, reg_lambda=2.0,
              objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=4)


def informative(df, lo=-6, hi=6):
    """Training rows: drop rungs so far from i0 they are trivially certain."""
    return df[(df.k > lo) & (df.k < hi)]


class LadderModel:
    def __init__(self, **kw):
        p = dict(PARAMS); p.update(kw)
        mono = tuple(-1 if f == "k" else 0 for f in FEATURES)
        self.clf = xgb.XGBClassifier(monotone_constraints=mono, **p)

    def fit(self, df):
        d = informative(df)
        self.clf.fit(d[FEATURES], d["y"])
        return self

    def predict(self, df):
        p = self.clf.predict_proba(df[FEATURES])[:, 1]
        # rungs far outside the training window are effectively certain
        p = np.where(df.k <= -6, 0.999, np.where(df.k >= 6, 0.001, p))
        return np.clip(p, 0.001, 0.999)

    def save(self, path):
        self.clf.save_model(path)

    @classmethod
    def load(cls, path):
        m = cls(); m.clf.load_model(path); return m


def brier(p, y): return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
