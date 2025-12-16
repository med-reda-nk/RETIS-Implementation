import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
from retis import RETIS

warnings.filterwarnings('ignore')


class StandardScalerCustom:
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X: np.ndarray) -> 'StandardScalerCustom':
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1  # Avoid division by zero
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class RETISOptimizer:

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.scaler = StandardScalerCustom()
        self.best_model = None
        self.best_params = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def preprocess_data(self, scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        if scale:
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            print("[+] Data scaled using custom StandardScaler")
            return self.X_train_scaled, self.X_test_scaled
        return self.X_train, self.X_test

    def grid_search_optimization(self, use_scaled: bool = True, verbose: bool = True) -> Tuple[RETIS, Dict]:
      
        if verbose:
            print("\n" + "=" * 60)
            print("GRID SEARCH OPTIMIZATION")
            print("=" * 60)

        X_train = self.X_train_scaled if use_scaled else self.X_train

        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10],
            'm_estimate': [0.5, 1.0],
            'min_mse_reduction': [0.0, 0.001, 0.01],
            'max_threshold_candidates': [10, 25, 50],
        }

        best_score = float('inf')
        best_params = None
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        
        if verbose:
            print(f"Searching {total_combinations} parameter combinations...\n")

        # Generate all parameter combinations
        def generate_combinations(params_dict):
            keys = list(params_dict.keys())
            values = list(params_dict.values())
            
            def recurse(idx, current):
                if idx == len(keys):
                    yield current.copy()
                else:
                    for value in values[idx]:
                        current[keys[idx]] = value
                        yield from recurse(idx + 1, current)
            
            yield from recurse(0, {})

        combo_count = 0
        for params in generate_combinations(param_grid):
            combo_count += 1
            
            # 5-fold cross-validation
            cv_scores = []
            fold_size = len(X_train) // 5
            
            for fold in range(5):
                # Create fold
                test_start = fold * fold_size
                test_end = test_start + fold_size
                
                if fold == 4:  # Last fold gets remaining samples
                    test_end = len(X_train)
                
                X_cv_test = X_train[test_start:test_end]
                y_cv_test = self.y_train[test_start:test_end]
                X_cv_train = np.vstack([X_train[:test_start], X_train[test_end:]])
                y_cv_train = np.concatenate([self.y_train[:test_start], self.y_train[test_end:]])
                
                # Train model
                retis = RETIS(**params)
                retis.fit(X_cv_train, y_cv_train)
                
                # Evaluate
                y_pred = retis.predict(X_cv_test)
                mse = np.mean((y_cv_test - y_pred) ** 2)
                cv_scores.append(mse)
            
            mean_cv_score = np.mean(cv_scores)
            
            if mean_cv_score < best_score:
                best_score = mean_cv_score
                best_params = params.copy()
        
        # Train final model with best parameters
        self.best_params = best_params
        self.best_model = RETIS(**best_params)
        self.best_model.fit(X_train, self.y_train)

        if verbose:
            print(f"\n[+] Best parameters found (CV MSE: {best_score:.4f}):")
            for param, value in sorted(best_params.items()):
                print(f"  {param}: {value}")

        return self.best_model, self.best_params

    def manual_optimization_guide(self) -> Dict:
        print("\n" + "=" * 60)
        print("MANUAL OPTIMIZATION GUIDE")
        print("=" * 60)

        n_samples = len(self.y_train)
        n_features = self.X_train.shape[1]
        target_variance = np.var(self.y_train)
        
        print(f"\n[Dataset Characteristics]")
        print(f"  Samples: {n_samples}, Features: {n_features}")
        print(f"  Target variance: {target_variance:.4f}")
        
        # Generate recommendations
        recommendations = {
            'max_depth': min(15, max(3, int(np.log2(n_samples)))),
            'min_samples_split': max(2, int(np.sqrt(n_samples))),
            'min_samples_leaf': max(1, int(np.sqrt(n_samples) / 2)),
            'm_estimate': 2.0 if target_variance > 100 else 1.0,
            'min_mse_reduction': 0.001 if n_samples > 1000 else 0.01,
        }
        
        print(f"\n[Recommendations]")
        for param, value in sorted(recommendations.items()):
            print(f"  {param}: {value}")
        
        return recommendations

    def compare_configurations(self, configs: List[Dict], cv: int = 5, run_baselines: bool = False) -> List[Dict]:

        print("\n" + "=" * 60)
        print("CONFIGURATION COMPARISON")
        print("=" * 60)

        X_train = self.X_train_scaled if self.X_train_scaled is not None else self.X_train
        results = []

        baseline_mean_mse = None
        if run_baselines:
            # Simple baseline: predict the training mean for every sample
            mean_target = np.mean(self.y_train)
            baseline_preds = np.full_like(self.y_train, fill_value=mean_target, dtype=float)
            baseline_mean_mse = np.mean((self.y_train - baseline_preds) ** 2)
            print(f"[+] Baseline (mean) MSE on training data: {baseline_mean_mse:.4f}")

        # Use shuffled KFold indices for more stable CV estimates
        indices = np.arange(len(X_train))
        rng = np.random.RandomState(42)
        rng.shuffle(indices)

        fold_sizes = (len(indices) // cv) * np.ones(cv, dtype=int)
        fold_sizes[: len(indices) % cv] += 1
        folds = []
        cur = 0
        for fs in fold_sizes:
            folds.append(indices[cur:cur + fs])
            cur += fs

        for i, config in enumerate(configs):
            print(f"\n🔍 Configuration {i + 1}: {config}")
            
            cv_scores = []
            
            for fold in range(cv):
                test_idx = folds[fold]
                train_idx = np.concatenate([folds[j] for j in range(cv) if j != fold])

                X_cv_train = X_train[train_idx]
                y_cv_train = self.y_train[train_idx]
                X_cv_test = X_train[test_idx]
                y_cv_test = self.y_train[test_idx]
                
                retis = RETIS(**config)
                retis.fit(X_cv_train, y_cv_train)
                y_pred = retis.predict(X_cv_test)
                mse = np.mean((y_cv_test - y_pred) ** 2)
                cv_scores.append(mse)
            
            mean_cv = np.mean(cv_scores)
            std_cv = np.std(cv_scores)
            
            results.append({
                'config': config,
                'mean_cv_mse': mean_cv,
                'std_cv_mse': std_cv,
                'cv_scores': cv_scores
            })
            if baseline_mean_mse is not None:
                results[-1]['baseline_mean_mse'] = baseline_mean_mse
            # Train on full training data and evaluate on provided test set
            try:
                final_model = RETIS(**config)
                final_model.fit(X_train, self.y_train)
                y_test_pred = final_model.predict(self.X_test if self.X_test_scaled is None else self.X_test_scaled)
                test_mse = float(np.mean((self.y_test - y_test_pred) ** 2))
            except Exception:
                test_mse = float('inf')

            results[-1]['test_mse'] = test_mse
            if 'test_mse' in results[-1]:
                results[-1]['test_rmse'] = float(np.sqrt(results[-1]['test_mse']))
            
            print(f"  CV MSE: {mean_cv:.4f} ± {std_cv:.4f}")

        return results

    def evaluate_model(self, model: Optional[RETIS] = None, use_scaled: bool = True) -> Dict:
        if model is None:
            model = self.best_model
        if model is None:
            raise ValueError("No model to evaluate. Train a model first.")

        X_train = self.X_train_scaled if use_scaled else self.X_train
        X_test = self.X_test_scaled if use_scaled else self.X_test

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_mse = np.mean((self.y_train - y_pred_train) ** 2)
        test_mse = np.mean((self.y_test - y_pred_test) ** 2)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mae = np.mean(np.abs(self.y_train - y_pred_train))
        test_mae = np.mean(np.abs(self.y_test - y_pred_test))
        
        # R² score
        def calc_r2(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        train_r2 = calc_r2(self.y_train, y_pred_train)
        test_r2 = calc_r2(self.y_test, y_pred_test)

        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        print("\n📈 Training Set:")
        print(f"  MSE:  {train_mse:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  MAE:  {train_mae:.4f}")
        print(f"  R²:   {train_r2:.4f}")

        print("\n📉 Test Set:")
        print(f"  MSE:  {test_mse:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  R²:   {test_r2:.4f}")

        overfit = train_r2 - test_r2
        print(f"\n⚠️  Overfitting gap: {overfit:.4f}")
        if overfit > 0.1:
            print("   → Consider more regularization")
        elif overfit < 0:
            print("   → Model may be underfitting")
        else:
            print("   → Good generalization!")
        
        return {
            'train_mse': train_mse, 'test_mse': test_mse,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_r2': train_r2, 'test_r2': test_r2,
            'overfit_gap': overfit,
            'n_leaves': model.get_n_leaves(),
            'tree_depth': model.get_depth(),
        }


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("RETIS OPTIMIZATION DEMO - Custom Implementation")
    print("=" * 60)
    
    # Generate synthetic data from scratch
    np.random.seed(42)
    n_train = 300
    n_test = 100
    n_features = 5
    
    X_train = np.random.randn(n_train, n_features)
    y_train = (2*X_train[:, 0] - 3*X_train[:, 1] + X_train[:, 2]**2 + 
               0.5*X_train[:, 3] + np.random.randn(n_train) * 10)
    
    X_test = np.random.randn(n_test, n_features)
    y_test = (2*X_test[:, 0] - 3*X_test[:, 1] + X_test[:, 2]**2 + 
              0.5*X_test[:, 3] + np.random.randn(n_test) * 10)
    
    # Initialize optimizer
    optimizer = RETISOptimizer(X_train, y_train, X_test, y_test)
    
    # Preprocess data
    optimizer.preprocess_data(scale=True)
    
    # Get optimization guide
    recommendations = optimizer.manual_optimization_guide()
    
    # Train model with recommendations
    retis = RETIS(**recommendations)
    retis.fit(optimizer.X_train_scaled, y_train)
    
    # Evaluate
    optimizer.best_model = retis
    metrics = optimizer.evaluate_model()
    
    # Compare configurations
    configs = [
        {'max_depth': 3, 'min_samples_split': 20, 'min_samples_leaf': 10, 'm_estimate': 3.0},
        {'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'm_estimate': 2.0},
        {'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2, 'm_estimate': 1.0},
    ]
    
    print("\n[+] Comparing configurations...")
    results = optimizer.compare_configurations(configs, cv=5)
    
    print("\n" + "=" * 60)
    print("Optimization complete! All built from scratch with numpy.")
    print("=" * 60)
    
