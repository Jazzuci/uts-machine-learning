from __future__ import annotations

import numpy as np
import pandas as pd

def make_new_samples(n: int = 5, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Luas": rng.normal(75, 15, size=n).clip(30, 180),
        "Kamar": rng.integers(1, 6, size=n),
        "KamarMandi": rng.integers(1, 4, size=n),
        "JarakPusat": rng.normal(7, 3, size=n).clip(0.5, 25),
        "UsiaBangunan": rng.normal(8, 4, size=n).clip(0, 35),
        "AksesTransport": rng.normal(0.0, 1.0, size=n),
    })
