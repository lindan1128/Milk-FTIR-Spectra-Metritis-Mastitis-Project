#!/usr/bin/env python
# coding: utf-8

__author__ = 'Dan Lin'


import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

from typing import Tuple, List, Dict


def normalize(data: pd.DataFrame) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.values)


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape: Tuple[int, int], units: int = 20, epochs: int = 200, batch_size: int = 32):
        self.input_shape = input_shape
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential([
            LSTM(self.units, activation='relu', input_shape=self.input_shape),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int).flatten()


def model_evaluation(X: np.ndarray, y: np.ndarray, model: BaseEstimator, n_samples: int = 50, n_splits: int = 3) -> pd.DataFrame:
    metrics_summary: Dict[str, List[float]] = {'accuracy': [], 'sensitivity': [], 'specificity': []}
    cv = StratifiedKFold(n_splits=n_splits)

    for _ in range(n_samples):
        X_sampled, y_sampled = resample_data(X, y)
        acc_scores, sens_scores, spec_scores = perform_cross_validation(X_sampled, y_sampled, model, cv)
        
        metrics_summary['accuracy'].append(np.mean(acc_scores))
        metrics_summary['sensitivity'].append(np.mean(sens_scores))
        metrics_summary['specificity'].append(np.mean(spec_scores))

    return pd.DataFrame(metrics_summary)


def resample_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    minority_class_size = np.sum(y == 1)
    minority_indices = np.random.choice(np.where(y == 0)[0], size=minority_class_size, replace=True)
    X_sampled = np.vstack([X[minority_indices], X[y == 1]])
    y_sampled = np.array([0] * minority_class_size + [1] * minority_class_size)
    return X_sampled, y_sampled


def perform_cross_validation(X: np.ndarray, y: np.ndarray, model: BaseEstimator, cv: StratifiedKFold) -> Tuple[List[float], List[float], List[float]]:
    acc_scores, sens_scores, spec_scores = [], [], []

    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        acc_scores.append(accuracy_score(y[test_idx], y_pred))
        sens_scores.append(recall_score(y[test_idx], y_pred))
        tn, fp, fn, tp = confusion_matrix(y[test_idx], y_pred).ravel()
        spec_scores.append(tn / (tn + fp))

    return acc_scores, sens_scores, spec_scores


def main():
    df = pd.read_csv('/path/to/data/spc.csv').dropna(subset=['milkweightlbs', 'cells'])
    df_h = df[(df['disease'] == 0) & (df['dim'] <= 7)]
    df_met = df[~df['dim_met'].isnull() & df['dim_mast'].isnull() & df['dim_da'].isnull() & df['dim_ket'].isnull() & (df['dim'] <= 7)]
    met = pd.concat([df_h, df_met])
    y = np.array([0] * len(df_h) + [1] * len(df_met))

    models = {
        'PLS': PLSRegression(n_components=2),
        'RF': RandomForestClassifier(max_depth=2, n_estimators=100, random_state=42, class_weight='balanced'),
        'LSTM': LSTMClassifier(input_shape=(1, met.shape[1]))
    }

    results = {}
    for name, model in models.items():
        X = normalize(met)
        results[name] = model_evaluation(X, y, model)

    results_df = pd.concat(results)
    print(results_df)


if __name__ == "__main__":
    main()
