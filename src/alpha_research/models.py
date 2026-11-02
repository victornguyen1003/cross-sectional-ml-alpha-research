"""Training-only purged date CV; a fixed fit is used for each held-out period."""

import itertools

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

MODEL_ORDER = ["momentum", "ols", "ridge", "lasso", "elasticnet", "lightgbm"]


def mean_rank_ic(frame):
    correlations = []
    for _, group in frame.dropna(subset=["realized_return", "prediction"]).groupby("date"):
        if len(group) >= 3 and group.prediction.nunique() > 1 and group.realized_return.nunique() > 1:
            correlations.append(group.prediction.rank().corr(group.realized_return.rank()))
    return float(np.mean(correlations)) if correlations else float("nan")


def purged_folds(samples, n_splits=3):
    dates = np.array(sorted(samples.date.unique()))
    if len(dates) < n_splits + 1:
        raise ValueError("Too few training signal dates for chronological CV")
    blocks = np.array_split(dates, n_splits + 1)
    for block in blocks[1:]:
        start = pd.Timestamp(block[0])
        # No rows from a validation cross-section may enter training. Labels must
        # have matured strictly before that cross-section's feature timestamp.
        train = np.flatnonzero((samples.date < start) & (samples.label_end < start))
        valid = np.flatnonzero(samples.date.isin(block))
        if len(train) and len(valid):
            yield train, valid


def make_model(name, params, seed):
    if name == "ols":
        estimator = LinearRegression()
    elif name == "ridge":
        estimator = Ridge(**params)
    elif name == "lasso":
        estimator = Lasso(**params, max_iter=20000, random_state=seed)
    elif name == "elasticnet":
        estimator = ElasticNet(**params, max_iter=20000, random_state=seed)
    elif name == "lightgbm":
        estimator = LGBMRegressor(**params, random_state=seed, deterministic=True,
                                  force_col_wise=True, n_jobs=1, verbosity=-1)
    else:
        raise ValueError(f"Unknown model {name}")
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), estimator)


def parameter_grid(name, settings, seed):
    if name == "ols":
        return [{}]
    if name in {"ridge", "lasso"}:
        return [{"alpha": alpha} for alpha in settings[f"{name}_alpha"]]
    if name == "elasticnet":
        return [{"alpha": alpha, "l1_ratio": ratio} for alpha, ratio in itertools.product(
            settings["elasticnet_alpha"], settings["elasticnet_l1_ratio"])]
    candidates = [dict(zip(["num_leaves", "learning_rate", "n_estimators"], values))
                  for values in itertools.product(settings["lgbm_num_leaves"],
                                                  settings["lgbm_learning_rate"],
                                                  settings["lgbm_n_estimators"])]
    if settings["search"] == "random":
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(len(candidates), min(len(candidates), settings["random_candidates"]),
                                    replace=False))
        candidates = [candidates[index] for index in indices]
    return candidates


def tune_models(train, settings, seed):
    train = train[train.label_available].reset_index(drop=True)
    folds = list(purged_folds(train, settings["cv_splits"]))
    if not folds:
        raise ValueError("No nonempty purged training folds")
    chosen, scores = {}, []
    for name in MODEL_ORDER[1:]:
        best, best_score = None, -np.inf
        for params in parameter_grid(name, settings, seed):
            predictions = []
            for train_i, valid_i in folds:
                model = make_model(name, params, seed)
                model.fit(train.iloc[train_i][FEATURES], train.iloc[train_i].target)
                frame = train.iloc[valid_i][["date", "realized_return"]].copy()
                frame["prediction"] = model.predict(train.iloc[valid_i][FEATURES])
                predictions.append(frame)
            score = mean_rank_ic(pd.concat(predictions))
            scores.append({"model": name, "params": params, "cv_rank_ic": score})
            if best is None or (np.isfinite(score) and score > best_score):
                best, best_score = params, score if np.isfinite(score) else -np.inf
        chosen[name] = best
    return chosen, scores


def fit_predict(name, params, train, evaluation, seed):
    predictions = evaluation.copy()
    if name == "momentum":
        predictions["prediction"] = predictions.momentum
        return None, predictions
    train = train[train.label_available]
    if train.empty:
        raise ValueError("No mature training labels")
    model = make_model(name, params, seed)
    model.fit(train[FEATURES], train.target)
    predictions["prediction"] = model.predict(predictions[FEATURES])
    return model, predictions
