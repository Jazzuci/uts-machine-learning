from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

def save_coeff_importance(estimator, feature_names, outdir: str, name: str, topk: int = 20):
    os.makedirs(outdir, exist_ok=True)
    if not hasattr(estimator, "coef_"):
        return
    coefs = estimator.coef_.ravel()
    idx = np.argsort(np.abs(coefs))[::-1][:topk]
    plt.figure()
    plt.bar(range(len(idx)), coefs[idx])
    labels = [feature_names[i] for i in idx]
    plt.xticks(range(len(idx)), labels, rotation=45, ha='right')
    plt.title(f"Top-{topk} Coeff Importance - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"coeff_importance_{name}.png"))
    plt.close()
