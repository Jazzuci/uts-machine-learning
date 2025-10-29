from __future__ import annotations

import os
import json
from dataclasses import asdict
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from src.data import generate_synthetic, split_and_impute
from src.features import transform_features
from src.models import train_for_degree, select_best
from src.evaluate import (
    save_histograms, save_scatter_vs_target, save_corr_heatmap,
    save_residual_plot, save_pred_vs_true, save_learning_curve,
    results_table_to_markdown
)
from src.importance import save_coeff_importance
from src.predict import make_new_samples

FIG_DIR = "figures"
MODEL_DIR = "models"

def log(msg: str):
    print(f"[INFO] {msg}")

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) Generate synthetic data (+missing +outliers), split, impute
    log("Generating synthetic dataset (n=400) with light missingness & outliers...")
    df = generate_synthetic(n_samples=400, seed=42)
    df.to_csv("data/synthetic_raw.csv", index=False)
    log("Splitting and imputing (median)...")
    splits = split_and_impute(df)
    Xtr, Xte, ytr, yte = splits.X_train, splits.X_test, splits.y_train, splits.y_test

    # EDA plots on the IMPUTED train set merged with y
    log("Saving EDA figures to ./figures ...")
    train_df_for_eda = pd.concat([Xtr.reset_index(drop=True), ytr.reset_index(drop=True)], axis=1)
    train_df_for_eda.columns = list(Xtr.columns) + ["Harga"]
    save_histograms(train_df_for_eda, target_col="Harga", outdir=FIG_DIR)
    save_scatter_vs_target(train_df_for_eda, target_col="Harga", outdir=FIG_DIR)
    save_corr_heatmap(train_df_for_eda, outdir=FIG_DIR)

    # 2) Iterate degree 1..5, train Linear/Ridge/Lasso (with alpha grid), 5-fold CV
    all_results = []
    best_global = None
    best_bundle = None
    best_feature_names = None

    for deg in range(1, 6):
        log(f"Training models for degree={deg} ...")
        bundle = transform_features(Xtr, Xte, degree=deg)
        res = train_for_degree(bundle.X_train, bundle.X_test, ytr, yte, degree=deg, cv_splits=5, seed=42)
        all_results.extend(res)

        # choose local best for plots
        local_best = max(res, key=lambda r: r.test_scores['R2'])

        # plots for the local best
        yhat_tr = local_best.estimator.predict(bundle.X_train)
        yhat_te = local_best.estimator.predict(bundle.X_test)
        save_residual_plot(ytr, yhat_tr, FIG_DIR, name=f"{local_best.name}_deg{deg}_train")
        save_residual_plot(yte, yhat_te, FIG_DIR, name=f"{local_best.name}_deg{deg}_test")
        save_pred_vs_true(ytr, yhat_tr, FIG_DIR, name=f"{local_best.name}_deg{deg}_train")
        save_pred_vs_true(yte, yhat_te, FIG_DIR, name=f"{local_best.name}_deg{deg}_test")
        save_learning_curve(local_best.estimator, bundle.X_train, ytr, FIG_DIR, name=f"{local_best.name}_deg{deg}")

        # save coeff importance if available
        # construct feature names from polynomial transformer
        poly = bundle.pipeline.named_steps['poly']
        feature_names = poly.get_feature_names_out(input_features=Xtr.columns)
        save_coeff_importance(local_best.estimator, feature_names, FIG_DIR, name=f"{local_best.name}_deg{deg}")

        # update global best by Test RMSE
        cand = local_best if best_global is None else min([best_global, local_best], key=lambda r: r.test_scores["RMSE"])
        if cand is local_best:
            best_global = local_best
            best_bundle = bundle
            best_feature_names = feature_names

    # 3) Summarize results
    rows = []
    for r in all_results:
        rows.append({
            "Model": r.name,
            "Degree": r.degree,
            "BestParams": json.dumps(r.best_params),
            "Train_R2": round(r.train_scores["R2"], 4),
            "Test_R2": round(r.test_scores["R2"], 4),
            "Test_RMSE": round(r.test_scores["RMSE"], 3),
            "Test_MAE": round(r.test_scores["MAE"], 3),
            "Test_MAPE%": round(r.test_scores["MAPE(%)"], 3),
        })
    md = results_table_to_markdown(rows)
    open("results_summary.md","w", encoding="utf-8").write(md)
    log("Saved results table -> results_summary.md")

    # 4) Persist best model & its polynomial+scaler pipeline
    log(f"Best model: {best_global.name} (degree={best_global.degree}, params={best_global.best_params})")
    dump(best_global.estimator, os.path.join(MODEL_DIR, "best_model.joblib"))
    dump(best_bundle.pipeline, os.path.join(MODEL_DIR, "poly_scaler_pipeline.joblib"))

    # 5) Make 5 new predictions
    newX = make_new_samples(n=5, seed=7)
    newX_trans = best_bundle.pipeline.transform(newX)
    preds = best_global.estimator.predict(newX_trans)
    out_df = newX.copy()
    out_df["PrediksiHarga"] = preds
    out_df.to_csv("prediksi_5_data_baru.csv", index=False)
    log("Saved predictions -> prediksi_5_data_baru.csv")

    # 6) Save a quick JSON report 
    report = {
        "best_model": {
            "name": best_global.name,
            "degree": best_global.degree,
            "best_params": best_global.best_params,
            "test_scores": best_global.test_scores
        },
        "artifacts": {
            "best_model": "models/best_model.joblib",
            "poly_scaler_pipeline": "models/poly_scaler_pipeline.joblib",
            "figures_dir": "figures"
        }
    }
    open("quick_report.json","w", encoding="utf-8").write(json.dumps(report, indent=2))

    log("All done. Check ./figures for PNGs and results_summary.md for the table.")

if __name__ == "__main__":
    main()
