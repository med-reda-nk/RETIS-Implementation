import numpy as np
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class LinearRegressionCustom:
    
    def __init__(self):
        self.coefficients = None  # [intercept, coef_1, coef_2, ...]
        self.intercept_ = None
        self.coef_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionCustom':
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Augment X with intercept column
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Solve normal equations: (X^T X) β = X^T y
        # Using lstsq for numerical stability
        coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        
        self.coefficients = coeffs.reshape(-1)
        self.intercept_ = self.coefficients[0]
        self.coef_ = self.coefficients[1:]
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X)
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        return (X_aug @ self.coefficients).reshape(-1, 1)


class BaseEstimator:
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        from inspect import signature
        sig = signature(self.__init__)
        return {p: getattr(self, p, None) for p in sig.parameters 
                if p != 'self'}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class RegressorMixin:
    pass


class RETISNode:

    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.model = None
        self.is_leaf = False
        self.n_samples = 0
        self.mse = 0.0


class RETIS(BaseEstimator, RegressorMixin):
 
    def __init__(self,
                 max_depth: int = 10,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 5,
                 m_estimate: float = 2.0,
                 min_mse_reduction: float = 0.01,
                 max_threshold_candidates: int = 25,
                 use_fast_solver: bool = True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.m_estimate = m_estimate
        self.min_mse_reduction = min_mse_reduction
        # Maximum number of candidate thresholds to evaluate per feature.
        # Limiting this reduces work while keeping good split coverage.
        self.max_threshold_candidates = max_threshold_candidates
        # Use a lightweight numpy least-squares solver for leaf models to reduce overhead
        # (avoids repeated sklearn object construction). Can be turned off for debugging.
        self.use_fast_solver = use_fast_solver
        self.root = None
        self.n_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RETIS':
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        self.n_features = X.shape[1]

        # Grow the tree
        self.root = self._grow_tree(X, y, depth=0)

        # Prune the tree using Bayesian m-estimate
        self._prune_tree(self.root, X, y)

        return self

    def _fit_linear_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional['LinearRegressionCustom'], float]:
        if X.shape[0] < 2:
            # Not enough samples for linear regression
            mean_pred = np.mean(y)
            mse = np.mean((y - mean_pred) ** 2)
            return None, mse

        # Use custom numpy least-squares solver
        try:
            # augment X with intercept column
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
            coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            preds = (X_aug @ coeffs).reshape(-1, 1)
            mse = float(np.mean((y - preds) ** 2))

            # Create a simple model object to store coefficients
            model = LinearRegressionCustom()
            model.coefficients = coeffs.reshape(-1)
            model.intercept_ = model.coefficients[0]
            model.coef_ = model.coefficients[1:]
            return model, mse
        except Exception:
            mean_pred = float(np.mean(y))
            mse = float(np.mean((y - mean_pred) ** 2))
            return None, mse

    def _calculate_split_mse(self, X: np.ndarray, y: np.ndarray,
                             feature_idx: int, threshold: float) -> Tuple[float, np.ndarray, np.ndarray]:
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            return float('inf'), left_mask, right_mask

        # Fit linear models for both splits
        _, mse_left = self._fit_linear_model(X_left, y_left)
        _, mse_right = self._fit_linear_model(X_right, y_right)

        # Weighted MSE
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        weighted_mse = (n_left * mse_left + n_right * mse_right) / n_total

        return weighted_mse, left_mask, right_mask

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        # Current MSE with linear model
        _, current_mse = self._fit_linear_model(X, y)

        for feature_idx in range(self.n_features):
            # Get unique values for this feature
            unique_vals = np.unique(X[:, feature_idx])

            # if too many unique values, limit candidate thresholds using quantiles
            if len(unique_vals) > self.max_threshold_candidates:
                # compute candidate thresholds as quantiles (skip extreme endpoints)
                q = np.linspace(0, 1, self.max_threshold_candidates + 2)[1:-1]
                # use quantile centers to propose split thresholds
                thresholds = np.quantile(unique_vals, q)
            else:
                # Consider midpoints between consecutive unique values
                if len(unique_vals) > 1:
                    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
                else:
                    thresholds = unique_vals

            for threshold in thresholds:
                mse, _, _ = self._calculate_split_mse(X, y, feature_idx, threshold)

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_idx
                    best_threshold = threshold

        # Check if split provides sufficient MSE reduction
        mse_reduction = current_mse - best_mse
        if mse_reduction < self.min_mse_reduction:
            return None, None, current_mse

        return best_feature, best_threshold, best_mse

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> RETISNode:
        node = RETISNode()
        node.n_samples = len(y)

        # Calculate MSE for this node (without fitting a permanent model)
        # This is used for pruning and split evaluation
        if len(y) >= 2:
            _, mse = self._fit_linear_model(X, y)
        else:
            mse = 0.0  # No variance with single sample
        node.mse = mse

        # Check stopping criteria
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                len(np.unique(y)) == 1):
            node.is_leaf = True
            # Only fit model for leaf nodes (original RETIS behavior)
            model, _ = self._fit_linear_model(X, y)
            node.model = model if model is not None else float(np.mean(y))
            return node

        # Find best split
        best_feature, best_threshold, split_mse = self._find_best_split(X, y)

        if best_feature is None:
            # No good split found - make it a leaf
            node.is_leaf = True
            # Only fit model for leaf nodes (original RETIS behavior)
            model, _ = self._fit_linear_model(X, y)
            node.model = model if model is not None else float(np.mean(y))
            return node

        # Create split (internal node - no model fitted)
        node.feature_idx = best_feature
        node.threshold = best_threshold

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Recursively grow children
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _estimate_error_m_estimate(self, node: RETISNode, X: np.ndarray, y: np.ndarray) -> float:
       
        n = node.n_samples
        if n == 0:
            return float('inf')

        # Bayesian adjustment for small sample sizes
        adjusted_error = node.mse * (n + self.m_estimate) / n
        return adjusted_error

    def _prune_tree(self, node: RETISNode, X: np.ndarray, y: np.ndarray) -> float:
        if node.is_leaf:
            return self._estimate_error_m_estimate(node, X, y)

        # Get masks for left and right splits
        left_mask = X[:, node.feature_idx] <= node.threshold
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        # Recursively prune children
        left_error = self._prune_tree(node.left, X_left, y_left)
        right_error = self._prune_tree(node.right, X_right, y_right)

        # Weighted error of subtrees
        n_left = len(y_left)
        n_right = len(y_right)
        subtree_error = (n_left * left_error + n_right * right_error) / (n_left + n_right)

        # Error if we convert this node to a leaf
        model, mse = self._fit_linear_model(X, y)
        node.mse = mse  # Update MSE for error estimation
        leaf_error = self._estimate_error_m_estimate(node, X, y)

        # Prune if converting to leaf reduces estimated error
        if leaf_error <= subtree_error:
            node.is_leaf = True
            node.model = model if model is not None else np.mean(y)
            node.left = None
            node.right = None
            return leaf_error

        return subtree_error

    def _predict_node(self, x: np.ndarray, node: RETISNode) -> float:
        if node.is_leaf:
            # model can be either:
            #  - LinearRegressionCustom instance
            #  - numeric constant
            if isinstance(node.model, LinearRegressionCustom):
                coeffs = node.model.coefficients
                x_aug = np.concatenate([[1.0], x])
                return float(np.dot(x_aug, coeffs))
            elif isinstance(node.model, (np.ndarray, list)):
                # assume coefficients with intercept first
                coeffs = np.array(node.model).reshape(-1)
                x_aug = np.concatenate([[1.0], x])
                return float(np.dot(x_aug, coeffs))
            else:
                # numeric constant (mean value)
                return float(node.model)

        if x[node.feature_idx] <= node.threshold:
            return self._predict_node(x, node.left)
        else:
            return self._predict_node(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X)
        # iterative traversal is slightly faster than recursive list-comprehension for many samples
        preds = np.zeros(X.shape[0], dtype=float)
        for i, x in enumerate(X):
            preds[i] = self._predict_node(x, self.root)
        predictions = preds
        return predictions

    # sklearn estimator compatibility
    def get_params(self, deep: bool = True):
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'm_estimate': self.m_estimate,
            'min_mse_reduction': self.min_mse_reduction,
            'max_threshold_candidates': self.max_threshold_candidates,
            'use_fast_solver': self.use_fast_solver,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)
        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def get_n_leaves(self) -> int:

        def count_leaves(node):
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return count_leaves(node.left) + count_leaves(node.right)

        return count_leaves(self.root)

    def get_depth(self) -> int:

        def calc_depth(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(calc_depth(node.left), calc_depth(node.right))

        return calc_depth(self.root)

"""
# Example usage and demonstration
if __name__ == "__main__":
    print("RETIS Regression Tree - Karalic (1992)")
    print("=" * 50)
    
    # Generate synthetic data from scratch
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    # Create a realistic target with signal and noise
    y = (2*X[:, 0] - 3*X[:, 1] + X[:, 2]**2 + 0.5*X[:, 3] + 
         np.random.randn(n_samples) * 10)
    
    # Manual train-test split (without sklearn)
    split_idx = int(0.7 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Fit RETIS model
    retis = RETIS(max_depth=5, min_samples_split=10,
                  min_samples_leaf=5, m_estimate=2.0)
    retis.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = retis.predict(X_train)
    y_pred_test = retis.predict(X_test)
    
    # Calculate metrics from scratch
    train_mse = np.mean((y_train - y_pred_train) ** 2)
    test_mse = np.mean((y_test - y_pred_test) ** 2)
    
    train_ss_res = np.sum((y_train - y_pred_train) ** 2)
    train_ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    train_r2 = 1 - (train_ss_res / train_ss_tot) if train_ss_tot != 0 else 0
    
    test_ss_res = np.sum((y_test - y_pred_test) ** 2)
    test_ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    test_r2 = 1 - (test_ss_res / test_ss_tot) if test_ss_tot != 0 else 0
    
    print(f"\nTree Statistics:")
    print(f"  Number of leaves: {retis.get_n_leaves()}")
    print(f"  Tree depth: {retis.get_depth()}")
    print(f"\nTraining Performance:")
    print(f"  MSE: {train_mse:.4f}")
    print(f"  R²: {train_r2:.4f}")
    print(f"\nTest Performance:")
    print(f"  MSE: {test_mse:.4f}")
    print(f"  R²: {test_r2:.4f}")
    
    print("\n" + "=" * 50)
    print("RETIS uses linear regression in leaves, making it")
    print("more powerful than standard regression trees that")
    print("use only constant predictions.")
    print("\nFully implemented from scratch with numpy only!")
"""