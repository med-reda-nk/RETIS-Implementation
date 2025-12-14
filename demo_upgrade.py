import time
import numpy as np
from retis import RETIS

from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV


def run_demo(n_samples=500, n_features=10):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=int(n_features*0.6), noise=10, random_state=42)

    model = RETIS(max_depth=6, min_samples_split=10, min_samples_leaf=5, use_fast_solver=True)

    t0 = time.time()
    model.fit(X, y)
    fit_time = time.time() - t0

    t0 = time.time()
    preds = model.predict(X)
    pred_time = time.time() - t0

    print(f"fit_time={fit_time:.4f}s, pred_time={pred_time:.6f}s for n={n_samples}, d={n_features}")

    # Try GridSearchCV
    try:
        gs = GridSearchCV(RETIS(), {'max_depth':[3,5], 'use_fast_solver':[True]}, cv=2)
        gs.fit(X, y)
        print("GridSearchCV run OK | best_params=", gs.best_params_)
    except Exception as e:
        print("GridSearchCV error:", type(e).__name__, str(e))


if __name__ == '__main__':
    for n in [200, 500, 1000]:
        run_demo(n_samples=n, n_features=8)
