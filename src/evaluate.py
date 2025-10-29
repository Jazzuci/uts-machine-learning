from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict
from sklearn.model_selection import learning_curve

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_histograms(df: pd.DataFrame, target_col: str, outdir: str):
    ensure_dir(outdir)
    for col in df.columns:
        plt.figure()
        df[col].hist(bins=30)
        plt.title(f"Histogram {col}")
        plt.xlabel(col); plt.ylabel("Frekuensi")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_{col}.png"))
        plt.close()

def save_scatter_vs_target(df: pd.DataFrame, target_col: str, outdir: str):
    ensure_dir(outdir)
    for col in df.drop(columns=[target_col]).columns:
        plt.figure()
        plt.scatter(df[col], df[target_col], s=16)
        plt.title(f"{col} vs {target_col}")
        plt.xlabel(col); plt.ylabel(target_col)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"scatter_{col}_vs_{target_col}.png"))
        plt.close()

def save_corr_heatmap(df: pd.DataFrame, outdir: str):
    ensure_dir(outdir)
    corr = df.corr(numeric_only=True)
    plt.figure()
    plt.imshow(corr, interpolation='nearest')
    plt.title("Correlation Heatmap")
    plt.colorbar()
    ticks = range(len(corr.columns))
    plt.xticks(ticks, corr.columns, rotation=45, ha='right')
    plt.yticks(ticks, corr.columns)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "corr_heatmap.png"))
    plt.close()

def save_residual_plot(y_true, y_pred, outdir: str, name: str):
    ensure_dir(outdir)
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, s=12)
    plt.axhline(0, linestyle='--')
    plt.title(f"Residuals vs Predicted - {name}")
    plt.xlabel("Predicted"); plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"residuals_{name}.png"))
    plt.close()

def save_pred_vs_true(y_true, y_pred, outdir: str, name: str):
    ensure_dir(outdir)
    plt.figure()
    plt.scatter(y_true, y_pred, s=12)
    m = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(m, m, linestyle='--')
    plt.title(f"Predicted vs True - {name}")
    plt.xlabel("True"); plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"pred_vs_true_{name}.png"))
    plt.close()

def save_learning_curve(estimator, X, y, outdir: str, name: str):
    ensure_dir(outdir)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring="r2", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8), shuffle=True, random_state=42
    )
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), marker='o', label="Train R2")
    plt.plot(train_sizes, test_scores.mean(axis=1), marker='o', label="CV R2")
    plt.xlabel("Train size"); plt.ylabel("R2")
    plt.title(f"Learning Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"learning_curve_{name}.png"))
    plt.close()

def results_table_to_markdown(rows: List[Dict]) -> str:
    import pandas as pd
    df = pd.DataFrame(rows)
    return df.to_markdown(index=False)
