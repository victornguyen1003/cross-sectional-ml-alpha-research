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


def parameter_grid(name, settings):
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
    return candidates


def tune_models(train, settings, seed):
    train = train[train.label_available].reset_index(drop=True)
    folds = list(purged_folds(train, settings["cv_splits"]))
    if not folds:
        raise ValueError("No nonempty purged training folds")
    chosen, scores = {}, []
    for name in MODEL_ORDER[1:]:
        best, best_score = None, -np.inf
        for params in parameter_grid(name, settings):
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


def fit_period(samples, research, settings, frozen=None):
    """Select on validation, or refit one frozen specification for the final test."""
    r = research
    final_test = frozen is not None
    validation_start, test_start = pd.Timestamp(r["validation_start"]), pd.Timestamp(r["test_start"])
    train = samples[(samples.date >= pd.Timestamp(r["train_start"])) &
                    (samples.date < validation_start) & (samples.label_end < validation_start) &
                    samples.label_available].copy()
    if train.empty:
        raise ValueError("No mature initial training samples")
    fitted, validation_scores = {}, []
    if final_test:
        selected = frozen["model"]
        chosen_params = frozen["params"]
        fitting = samples[(samples.date >= pd.Timestamp(r["train_start"])) &
                          (samples.date < test_start) & (samples.label_end < test_start) &
                          samples.label_available]
        evaluation = samples[samples.date >= test_start].copy()
        model, predictions = fit_predict(selected, chosen_params, fitting, evaluation, r["seed"])
        fitted[selected] = model
        cv_scores = []
        period = "test"
    else:
        parameters, cv_scores = tune_models(train, settings, r["seed"])
        evaluation = samples[(samples.date >= validation_start) & (samples.date < test_start) &
                             (samples.entry_date < test_start)].copy()
        # Do not evaluate labels that extend into the final-test period.
        boundary = evaluation.label_end >= test_start
        evaluation.loc[boundary, ["realized_return", "target"]] = np.nan
        evaluation.loc[boundary, "label_available"] = False
        if evaluation.empty:
            raise ValueError("No validation samples")
        all_predictions = {}
        for model_name in MODEL_ORDER:
            model, predicted = fit_predict(model_name, parameters.get(model_name, {}), train,
                                           evaluation, r["seed"])
            fitted[model_name] = model
            all_predictions[model_name] = predicted
            validation_scores.append({"model": model_name, "rank_ic": mean_rank_ic(predicted)})
        finite = [score for score in validation_scores if np.isfinite(score["rank_ic"])]
        if not finite:
            raise ValueError("No usable validation rank correlations; cannot freeze selection")
        best_score = max(score["rank_ic"] for score in finite)
        selected = next(score["model"] for score in finite if score["rank_ic"] >= best_score - 1e-12)
        chosen_params = parameters.get(selected, {})
        predictions = all_predictions[selected]
        period = "validation"
    if evaluation.empty:
        raise ValueError("No evaluation samples")
    return {
        "period": period, "model": selected, "params": chosen_params,
        "predictions": predictions, "models": fitted,
        "model_predictions": {selected: predictions} if final_test else all_predictions,
        "cv_scores": cv_scores, "validation_scores": validation_scores,
    }
