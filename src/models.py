from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

@dataclass
class ModelResult:
    name: str
    degree: int
    best_params: Dict[str, Any]
    train_scores: Dict[str, float]
    test_scores: Dict[str, float]
    estimator: Any

def _metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-8, np.abs(y_true)))) * 100.0
    r2 = r2_score(y_true, y_pred)
    return {"R2": r2, "MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE(%)": mape}

def train_for_degree(Xtr, Xte, ytr, yte, degree: int, cv_splits: int = 5, seed: int = 42):
    results = []

    # Linear Regression
    lin = LinearRegression(n_jobs=None)
    lin.fit(Xtr, ytr)
    yhat_tr = lin.predict(Xtr)
    yhat_te = lin.predict(Xte)
    results.append(ModelResult(
        name="Linear",
        degree=degree,
        best_params={},
        train_scores=_metrics(ytr, yhat_tr),
        test_scores=_metrics(yte, yhat_te),
        estimator=lin
    ))

    # Ridge
    ridge = Ridge()
    ridge_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    ridge_cv = GridSearchCV(ridge, ridge_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    ridge_cv.fit(Xtr, ytr)
    rbest = ridge_cv.best_estimator_
    results.append(ModelResult(
        name="Ridge",
        degree=degree,
        best_params={"alpha": ridge_cv.best_params_["alpha"]},
        train_scores=_metrics(ytr, rbest.predict(Xtr)),
        test_scores=_metrics(yte, rbest.predict(Xte)),
        estimator=rbest
    ))

    # Lasso
    lasso = Lasso(max_iter=10000)
    lasso_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
    lasso_cv = GridSearchCV(lasso, lasso_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    lasso_cv.fit(Xtr, ytr)
    lbest = lasso_cv.best_estimator_
    results.append(ModelResult(
        name="Lasso",
        degree=degree,
        best_params={"alpha": lasso_cv.best_params_["alpha"]},
        train_scores=_metrics(ytr, lbest.predict(Xtr)),
        test_scores=_metrics(yte, lbest.predict(Xte)),
        estimator=lbest
    ))

    return results

def select_best(results):
    # Choose model with lowest Test RMSE; tiebreaker: highest Test R2
    best = None
    for r in results:
        if best is None:
            best = r
        else:
            if (r.test_scores["RMSE"] < best.test_scores["RMSE"]) or (
                np.isclose(r.test_scores["RMSE"], best.test_scores["RMSE"]) and r.test_scores["R2"] > best.test_scores["R2"]
            ):
                best = r
    return best
