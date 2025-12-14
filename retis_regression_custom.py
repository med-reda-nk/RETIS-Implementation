import numpy as np
from typing import Dict, List, Tuple
from custom_metrics import CustomMetrics
import urllib.request


class RETISRegressionEvaluatorCustom:

    def __init__(self, model = None, X_train: np.ndarray = None, y_train: np.ndarray = None,
                 X_test: np.ndarray = None, y_test: np.ndarray = None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_model(self, model = None) -> Dict:
        if model is None:
            model = self.model

        if model is None:
            raise ValueError("No model provided")

        # Get predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # Calculate all metrics
        train_metrics = self._calculate_all_metrics(self.y_train, y_train_pred)
        test_metrics = self._calculate_all_metrics(self.y_test, y_test_pred)

        # Print results
        self._print_evaluation_results(train_metrics, test_metrics)

        return {'train': train_metrics, 'test': test_metrics}

    def _calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = {}

        metrics['mse'] = CustomMetrics.mse(y_true, y_pred)
        metrics['rmse'] = CustomMetrics.rmse(y_true, y_pred)
        metrics['mae'] = CustomMetrics.mae(y_true, y_pred)
        metrics['r2'] = CustomMetrics.r2_score(y_true, y_pred)
        metrics['explained_variance'] = CustomMetrics.explained_variance_score(y_true, y_pred)
        metrics['mape'] = CustomMetrics.mean_absolute_percentage_error(y_true, y_pred)
        metrics['median_ae'] = CustomMetrics.median_absolute_error(y_true, y_pred)
        metrics['relative_rmse'] = CustomMetrics.relative_rmse(y_true, y_pred)
        metrics['relative_mae'] = CustomMetrics.relative_mae(y_true, y_pred)

        return metrics

    def _print_evaluation_results(self, train_metrics: Dict, test_metrics: Dict):
        print("\n" + "="*70)
        print("🎯 RETIS REGRESSION MODEL EVALUATION")
        print("="*70)

        print(f"\n📊 Training Set Performance:")
        self._print_metrics(train_metrics)

        print(f"\n📈 Test Set Performance:")
        self._print_metrics(test_metrics)

        # Overfitting analysis
        overfitting = train_metrics['r2'] - test_metrics['r2']
        print(f"\n⚠️  Overfitting Analysis:")
        print(f"   R² difference (train - test): {overfitting:.4f}")

        if overfitting > 0.1:
            print("   → Significant overfitting detected")
        elif overfitting < -0.1:
            print("   → Model may be underfitting")
        else:
            print("   → Good generalization balance")

    def _print_metrics(self, metrics: Dict):
        print(f"   MSE:           {metrics['mse']:.4f}")
        print(f"   RMSE:          {metrics['rmse']:.4f}")
        print(f"   MAE:           {metrics['mae']:.4f}")
        print(f"   R²:            {metrics['r2']:.4f}")
        print(f"   Explained Var: {metrics['explained_variance']:.4f}")
        print(f"   MAPE:          {metrics['mape']:.4f}")
        print(f"   Median AE:     {metrics['median_ae']:.4f}")
        print(f"   Rel. RMSE:     {metrics['relative_rmse']:.4f}")

    def cross_validate(self, cv: int = 5) -> Dict[str, float]:
        print(f"\n🔄 Performing {cv}-fold cross-validation...")

        # Split data into folds
        n_samples = len(self.y_train)
        fold_size = max(1, n_samples // cv)  # Ensure at least 1 sample per fold

        cv_scores = []

        for i in range(cv):
            # Create validation set
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < cv - 1 else n_samples

            if val_start >= n_samples:
                continue

            val_indices = np.arange(val_start, min(val_end, n_samples))
            
            # Create train indices (everything except validation)
            train_indices = np.concatenate([np.arange(0, val_start), 
                                          np.arange(val_end, n_samples)])

            if len(train_indices) == 0:
                continue

            X_tr = self.X_train[train_indices]
            y_tr = self.y_train[train_indices]
            X_val = self.X_train[val_indices]
            y_val = self.y_train[val_indices]

            # Train model
            from retis import RETIS
            model_copy = RETIS(**self.model.get_params())
            model_copy.fit(X_tr, y_tr)

            # Evaluate
            y_val_pred = model_copy.predict(X_val)
            r2 = CustomMetrics.r2_score(y_val, y_val_pred)
            cv_scores.append(r2)

        cv_results = {
            'scores': np.array(cv_scores),
            'mean': float(np.mean(cv_scores)) if cv_scores else 0.0,
            'std': float(np.std(cv_scores)) if cv_scores else 0.0,
            'metric': 'R²'
        }

        print(f"   R² scores: {cv_scores}")
        print(f"   Mean: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")

        return cv_results

    def compare_with_baselines(self) -> Dict:
        retis_pred = self.model.predict(self.X_test)
        retis_r2 = CustomMetrics.r2_score(self.y_test, retis_pred)
        retis_rmse = CustomMetrics.rmse(self.y_test, retis_pred)

        # Baseline 1: Mean predictor
        mean_pred = np.full_like(self.y_test, np.mean(self.y_train), dtype=float)
        mean_r2 = CustomMetrics.r2_score(self.y_test, mean_pred)
        mean_rmse = CustomMetrics.rmse(self.y_test, mean_pred)

        # Baseline 2: Median predictor
        median_pred = np.full_like(self.y_test, np.median(self.y_train), dtype=float)
        median_r2 = CustomMetrics.r2_score(self.y_test, median_pred)
        median_rmse = CustomMetrics.rmse(self.y_test, median_pred)

        results = {
            'RETIS': {'R2': retis_r2, 'RMSE': retis_rmse},
            'Mean_Baseline': {'R2': mean_r2, 'RMSE': mean_rmse},
            'Median_Baseline': {'R2': median_r2, 'RMSE': median_rmse}
        }

        print("\n🏁 Model Comparison:")
        for model_name, metrics in results.items():
            print(f"   {model_name}: R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.4f}")

        return results

    def residuals_analysis(self) -> Dict:
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        train_residuals = self.y_train - y_train_pred
        test_residuals = self.y_test - y_test_pred

        analysis = {
            'train_residuals_mean': float(np.mean(train_residuals)),
            'train_residuals_std': float(np.std(train_residuals)),
            'train_residuals_min': float(np.min(train_residuals)),
            'train_residuals_max': float(np.max(train_residuals)),
            'test_residuals_mean': float(np.mean(test_residuals)),
            'test_residuals_std': float(np.std(test_residuals)),
            'test_residuals_min': float(np.min(test_residuals)),
            'test_residuals_max': float(np.max(test_residuals))
        }

        print("\n📊 Residuals Analysis:")
        print(f"\n   Training Residuals:")
        print(f"      Mean: {analysis['train_residuals_mean']:.4f}")
        print(f"      Std:  {analysis['train_residuals_std']:.4f}")
        print(f"      Min:  {analysis['train_residuals_min']:.4f}")
        print(f"      Max:  {analysis['train_residuals_max']:.4f}")

        print(f"\n   Test Residuals:")
        print(f"      Mean: {analysis['test_residuals_mean']:.4f}")
        print(f"      Std:  {analysis['test_residuals_std']:.4f}")
        print(f"      Min:  {analysis['test_residuals_min']:.4f}")
        print(f"      Max:  {analysis['test_residuals_max']:.4f}")

        return analysis


def generate_synthetic_regression_data(n_samples: int = 500, n_features: int = 5,
                                       noise: float = 10.0, random_state: int = 42) -> Tuple:
    np.random.seed(random_state)

    X = np.random.randn(n_samples, n_features)

    # Create target with known pattern
    true_coef = np.random.randn(n_features)
    y = X @ true_coef + noise * np.random.randn(n_samples)

    return X, y


def generate_synthetic_classification_data(n_samples: int = 500, n_features: int = 5,
                                          n_classes: int = 3, random_state: int = 42) -> Tuple:
    np.random.seed(random_state)

    X = np.random.randn(n_samples, n_features)

    # Create target with simple linear boundary
    scores = X @ np.random.randn(n_features, n_classes)
    y = np.argmax(scores, axis=1)

    return X, y


def custom_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                           random_state: int = None) -> Tuple:
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_size))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def load_california_housing() -> Tuple[np.ndarray, np.ndarray]:
    
    print("[+] Loading California Housing dataset...")
    
    try:
        # Download from pandas/sklearn default location
        url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
        response = urllib.request.urlopen(url, timeout=10)
        data = response.read().decode('utf-8')
        
        # Parse CSV
        lines = data.strip().split('\n')
        if len(lines) < 2:
            raise ValueError("Dataset too small")
        
        X_list = []
        y_list = []
        
        for line in lines[1:]:  # Skip header
            try:
                values = line.split(',')
                # Extract first 8 numeric features (skip ocean_proximity)
                row = []
                for i in range(8):
                    row.append(float(values[i]))
                X_list.append(row)
                # 9th column is target (median_house_value)
                y_list.append(float(values[8]))
            except:
                continue
        
        if len(X_list) == 0:
            raise ValueError("No valid data rows found")
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32) / 100000.0  # Normalize
        
        print(f"[+] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    except Exception as e:
        print(f"[!] Failed to load California Housing: {e}")
        print("[+] Using synthetic California Housing-like data...")
        # Generate synthetic data with similar characteristics
        np.random.seed(42)
        n_samples = 20640
        n_features = 8
        X = np.random.randn(n_samples, n_features) * np.array([4, 3.5, 32, 1000, 100, 40, 12, 8])
        # Add non-linear relationship
        y = (2*X[:, 0] - X[:, 1] + 0.05*X[:, 2] + 0.001*X[:, 3] + 
             0.01*X[:, 4] + np.random.randn(n_samples) * 0.2)
        print(f"[+] Synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y

