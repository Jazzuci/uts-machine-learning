from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

@dataclass
class FeatureBundle:
    pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame

def build_poly_scaler(degree: int) -> Pipeline:
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler())
    ])

def transform_features(X_train: pd.DataFrame, X_test: pd.DataFrame, degree: int) -> FeatureBundle:
    pipe = build_poly_scaler(degree)
    Xtr = pipe.fit_transform(X_train)
    Xte = pipe.transform(X_test)
    return FeatureBundle(pipe, Xtr, Xte)
