from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

RANDOM_SEED = 42

@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

def generate_synthetic(n_samples: int = 400, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Base features
    luas = rng.normal(70, 20, size=n_samples).clip(25, 200)  # m^2
    kamar = rng.integers(1, 6, size=n_samples)               # 1..5
    km = rng.integers(1, 4, size=n_samples)                  # 1..3
    jarak_pusat = rng.normal(8, 4, size=n_samples).clip(0.5, 30)  # km
    usia_bangunan = rng.normal(10, 6, size=n_samples).clip(0, 40)  # years
    akses_transport = rng.normal(0.0, 1.0, size=n_samples)         # z-score of accessibility

    # True function (nonlinear) + noise
    noise = rng.normal(0, 100, size=n_samples)
    price = (
        500 + 25*luas + 80*kamar + 60*km - 40*jarak_pusat - 5*usia_bangunan
        + 120*akses_transport + 0.08*(luas**2) - 2.0*(kamar*jarak_pusat)
    ) + noise

    df = pd.DataFrame({
        "Luas": luas,
        "Kamar": kamar,
        "KamarMandi": km,
        "JarakPusat": jarak_pusat,
        "UsiaBangunan": usia_bangunan,
        "AksesTransport": akses_transport,
        "Harga": price
    })

    # Inject light outliers (1.5% rows)
    n_out = max(1, int(0.015 * n_samples))
    idx_out = rng.choice(df.index, size=n_out, replace=False)
    df.loc[idx_out, "Harga"] *= rng.uniform(1.8, 2.5, size=n_out)

    # Inject small missingness (~2% per feature, except target)
    for col in df.columns[:-1]:
        mask = rng.uniform(0, 1, size=n_samples) < 0.02
        df.loc[mask, col] = np.nan

    return df

def split_and_impute(df: pd.DataFrame, test_size: float = 0.3, seed: int = RANDOM_SEED) -> DatasetSplits:
    X = df.drop(columns=["Harga"])
    y = df["Harga"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Simple imputer (median for numeric)
    imp = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_test.columns, index=X_test.index)

    return DatasetSplits(X_train_imp, X_test_imp, y_train, y_test)
