import numpy as np
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
import inspect

# sklearn compatibility classes (minimal implementation)
class BaseEstimator:
    def get_params(self, deep=True):
        params = {}
        try:
            sig = inspect.signature(self.__class__.__init__)
            for name, param in sig.parameters.items():
                if name == 'self':
                   continue
                if hasattr(self, name):
                    val = getattr(self, name)
                    if deep and hasattr(val, 'get_params'):
                        params[name] = val.get_params(deep=deep)
                    else:
                        params[name] = val
        except Exception:
                # Fallback: conservative behavior, return only simple attributes
            for key, value in self.__dict__.items():
                if not key.startswith('_') and isinstance(value, (int, float, bool, str, type(None))):
                    params[key] = value
        return params
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

class RegressorMixin:
    def score(self, X, y, sample_weight=None):
        # Custom R² calculation (no sklearn dependency)
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

class LinearRegressionCustom:
    def __init__(self):
        self.coefficients = None  # [intercept, coef_1, coef_2, ...]
        self.intercept_ = None
        self.coef_ = None
        self.residual_variance = None  # Added for statistical tests
        
    def fit(self, X: np.ndarray, y: np.ndarray, ridge_alpha: float = 0.0) -> 'LinearRegressionCustom':
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Augment X with intercept column
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Solve (regularized) normal equations: (X^T X + alpha * D) β = X^T y
        # where D is identity but we typically do not regularize intercept term.
        try:
            XtX = X_aug.T @ X_aug
            Xty = X_aug.T @ y

            if ridge_alpha and ridge_alpha > 0.0:
                # Do not regularize intercept term: add alpha to diagonal starting at index 1
                reg = np.zeros_like(XtX)
                diag = np.arange(XtX.shape[0])
                reg[diag, diag] = 1.0
                reg[0, 0] = 0.0
                XtX_reg = XtX + ridge_alpha * reg
            else:
                XtX_reg = XtX

            coeffs = np.linalg.solve(XtX_reg, Xty)
            coeffs = coeffs.reshape(-1, 1)

            self.coefficients = coeffs.reshape(-1)
            self.intercept_ = self.coefficients[0]
            self.coef_ = self.coefficients[1:]

            # Calculate residual variance for statistical tests
            predictions = X_aug @ coeffs
            residuals_vec = y - predictions
            df = len(y) - len(coeffs)  # degrees of freedom
            if df > 0:
                self.residual_variance = float(np.sum(residuals_vec ** 2) / df)
            else:
                self.residual_variance = 0.0
        except Exception:
            # Fallback to mean if singular matrix or other numerical issue
            self.coefficients = np.array([float(np.mean(y))] + [0.0] * X.shape[1])
            self.intercept_ = float(np.mean(y))
            self.coef_ = np.zeros(X.shape[1])
            self.residual_variance = float(np.var(y))
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X)
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        return (X_aug @ self.coefficients).reshape(-1, 1)


class RETISNode:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.model = None  # Linear model at this node
        self.is_leaf = False
        self.n_samples = 0
        self.error = 0.0  # Resubstitution error
        self.variance = 0.0  # Residual variance for statistical tests
        self.df = 0  # Degrees of freedom


class RETIS(BaseEstimator, RegressorMixin):
        
    def __init__(self, 
                 max_depth: int = 6,
                 min_samples_split: int = 20,
                 min_samples_leaf: int = 10,
                 significance_level: float = 0.10,
                 min_error_reduction: float = 0.005,
                 account_for_split_cost: bool = True,
                 use_variable_selection: bool = False,
                 variable_selection_threshold: float = 0.05,
                 # Compatibility / convenience parameters used across the codebase
                 # Mild ridge regularization to prevent coefficient explosion
                 m_estimate: float = 1.0,
                 min_mse_reduction: Optional[float] = None,
                 max_threshold_candidates: Optional[int] = 50,
                 use_fast_solver: bool = True):
       
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.significance_level = significance_level
        self.min_error_reduction = min_error_reduction
        self.account_for_split_cost = account_for_split_cost
        self.use_variable_selection = use_variable_selection
        self.variable_selection_threshold = variable_selection_threshold
        # Compatibility aliases (some callers use different names)
        self.m_estimate = m_estimate
        # Allow callers to pass `min_mse_reduction`; prefer explicit param if provided
        if min_mse_reduction is not None:
            self.min_error_reduction = min_mse_reduction
        self.max_threshold_candidates = max_threshold_candidates
        self.use_fast_solver = use_fast_solver
        self.root = None
        self.n_features = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RETIS':
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        self.n_features = X.shape[1]
        
        # Phase 1: Grow full tree with linear models at every node
        self.root = self._grow_tree(X, y, depth=0)
        
        # Phase 2: Prune using statistical significance test (F-test)
        self._prune_tree(self.root, X, y)
        
        return self
    
    def _fit_linear_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegressionCustom, float, float, int]:
       
        n = len(y)
        
        if n < 2:
            # Not enough samples for linear regression
            mean_val = float(np.mean(y))
            model = LinearRegressionCustom()
            model.coefficients = np.array([mean_val] + [0.0] * X.shape[1])
            model.intercept_ = mean_val
            model.coef_ = np.zeros(X.shape[1])
            model.residual_variance = 0.0
            error = 0.0
            return model, error, 0.0, 0
        
        try:
            # If variable selection is enabled and we have enough samples
            if self.use_variable_selection and n > X.shape[1] + 5:
                model, error, variance, df = self._fit_with_variable_selection(X, y)
            else:
                # Standard: Fit linear model with all features
                model = LinearRegressionCustom()
                # Use m_estimate as a ridge regularization parameter (alpha)
                ridge_alpha = getattr(self, 'm_estimate', 0.0) if getattr(self, 'm_estimate', None) is not None else 0.0
                model.fit(X, y, ridge_alpha=ridge_alpha)
                
                # Calculate error (sum of squared residuals)
                X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
                predictions = X_aug @ model.coefficients
                residuals = y.reshape(-1) - predictions
                error = float(np.sum(residuals ** 2))
                
                # Degrees of freedom
                df = n - len(model.coefficients)
                
                # Residual variance
                if df > 0:
                    variance = error / df
                else:
                    variance = 0.0
                
            return model, error, variance, df
            
        except Exception:
            # Fallback to mean (singular matrix or other numerical issue)
            # This is a practical necessity not explicitly in Karalič 1992
            mean_val = float(np.mean(y))
            model = LinearRegressionCustom()
            model.coefficients = np.array([mean_val] + [0.0] * X.shape[1])
            model.intercept_ = mean_val
            model.coef_ = np.zeros(X.shape[1])
            model.residual_variance = float(np.var(y))
            error = float(np.sum((y - mean_val) ** 2))
            variance = model.residual_variance
            df = n - 1
            return model, error, variance, df
    
    def _fit_with_variable_selection(self, X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegressionCustom, float, float, int]:
        
        from scipy import stats
        
        n = len(y)
        # Start with all features
        selected_features = list(range(X.shape[1]))
        
        while len(selected_features) > 0:
            # Fit model with current feature set
            X_selected = X[:, selected_features]
            X_aug = np.hstack([np.ones((n, 1)), X_selected])
            
            try:
                coeffs, *_ = np.linalg.lstsq(X_aug, y.reshape(-1, 1), rcond=None)
                predictions = X_aug @ coeffs
                residuals = y.reshape(-1) - predictions.reshape(-1)
                mse = np.sum(residuals ** 2) / (n - len(coeffs))
                
                # Calculate t-statistics for each coefficient (except intercept)
                if mse > 0:
                    # Variance-covariance matrix
                    XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
                    se = np.sqrt(np.diag(XtX_inv) * mse)
                    t_stats = np.abs(coeffs.reshape(-1) / se)
                    
                    # Find least significant variable (excluding intercept)
                    if len(t_stats) > 1:
                        min_t_idx = np.argmin(t_stats[1:]) + 1  # +1 to skip intercept
                        p_value = 2 * (1 - stats.t.cdf(t_stats[min_t_idx], n - len(coeffs)))
                        
                        # Remove if not significant
                        if p_value > self.variable_selection_threshold:
                            removed_idx = selected_features[min_t_idx - 1]
                            selected_features.remove(removed_idx)
                            continue
                
                # All remaining variables are significant
                break
            except:
                # Numerical issues, stop selection
                break
        
        # Fit final model with selected features
        if len(selected_features) > 0:
            X_final = X[:, selected_features]
        else:
            # No features selected, use intercept only
            X_final = np.zeros((n, 0))
        
        X_aug = np.hstack([np.ones((n, 1)), X_final])

        # Use ridge regularization consistent with node settings
        ridge_alpha = getattr(self, 'm_estimate', 0.0) if getattr(self, 'm_estimate', None) is not None else 0.0

        # Build normal equations and solve with optional regularization
        try:
            XtX = X_aug.T @ X_aug
            Xty = X_aug.T @ y.reshape(-1, 1)
            if ridge_alpha and ridge_alpha > 0.0:
                reg = np.zeros_like(XtX)
                diag = np.arange(XtX.shape[0])
                reg[diag, diag] = 1.0
                reg[0, 0] = 0.0
                XtX_reg = XtX + ridge_alpha * reg
            else:
                XtX_reg = XtX

            coeffs = np.linalg.solve(XtX_reg, Xty)
        except Exception:
            coeffs, *_ = np.linalg.lstsq(X_aug, y.reshape(-1, 1), rcond=None)

        # Create full coefficient vector (with zeros for non-selected features)
        full_coeffs = np.zeros(X.shape[1] + 1)
        try:
            full_coeffs[0] = float(coeffs[0])  # intercept
            for i, feat_idx in enumerate(selected_features):
                full_coeffs[feat_idx + 1] = float(coeffs[i + 1])
        except Exception:
            # Fallback
            full_coeffs[0] = float(np.mean(y))

        # Create model
        model = LinearRegressionCustom()
        model.coefficients = full_coeffs
        model.intercept_ = full_coeffs[0]
        model.coef_ = full_coeffs[1:]

        # Calculate error and variance
        try:
            predictions = X_aug @ coeffs
            residuals = y.reshape(-1) - predictions.reshape(-1)
            error = float(np.sum(residuals ** 2))
            df = n - len(coeffs)
            variance = error / df if df > 0 else 0.0
        except Exception:
            error = float(np.sum((y - np.mean(y)) ** 2))
            df = n - 1
            variance = error / df if df > 0 else 0.0

        return model, error, variance, df
    
    def _calculate_split_error(self, X: np.ndarray, y: np.ndarray, 
                               feature_idx: int, threshold: float) -> Tuple[float, np.ndarray, np.ndarray]:
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            return float('inf'), left_mask, right_mask
        
        # Fit linear models for both splits and get their errors
        _, error_left, _, _ = self._fit_linear_model(X_left, y_left)
        _, error_right, _, _ = self._fit_linear_model(X_right, y_right)
        
        total_error = error_left + error_right
        
        return total_error, left_mask, right_mask
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray, 
                        current_error: float) -> Tuple[Optional[int], Optional[float], float]:
        
        best_error = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(self.n_features):
            # Get unique values for this feature
            unique_vals = np.unique(X[:, feature_idx])
            
            if len(unique_vals) <= 1:
                continue
            
            # Consider midpoints between consecutive unique values
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

            # If user requested a maximum number of threshold candidates,
            # downsample the candidate thresholds uniformly to limit compute.
            if self.max_threshold_candidates is not None and len(thresholds) > self.max_threshold_candidates:
                # Uniformly sample indices
                idxs = np.linspace(0, len(thresholds) - 1, self.max_threshold_candidates, dtype=int)
                thresholds = thresholds[idxs]
            
            for threshold in thresholds:
                error, _, _ = self._calculate_split_error(X, y, feature_idx, threshold)
                
                if error < best_error:
                    best_error = error
                    best_feature = feature_idx
                    best_threshold = threshold
        
        # Check if split provides sufficient error reduction
        if best_feature is not None:
            error_reduction = (current_error - best_error) / (current_error + 1e-10)
            if error_reduction < self.min_error_reduction:
                return None, None, current_error
        
        return best_feature, best_threshold, best_error
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> RETISNode:
        
        node = RETISNode()
        node.n_samples = len(y)
        
        # Fit linear model at this node (CRITICAL: done for ALL nodes)
        model, error, variance, df = self._fit_linear_model(X, y)
        node.model = model
        node.error = error
        node.variance = variance
        node.df = df
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(np.unique(y)) == 1 or
            variance < 1e-10):  # No point splitting if no variance
            node.is_leaf = True
            return node
        
        # Find best split
        best_feature, best_threshold, split_error = self._find_best_split(X, y, error)
        
        if best_feature is None:
            # No good split found
            node.is_leaf = True
            return node
        
        # Create split
        node.feature_idx = best_feature
        node.threshold = best_threshold
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively grow children
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def _f_test_prune(self, node: RETISNode, X: np.ndarray, y: np.ndarray) -> bool:
        
        if node.is_leaf or node.left is None or node.right is None:
            return False
        
        # Get splits
        left_mask = X[:, node.feature_idx] <= node.threshold
        right_mask = ~left_mask
        
        # Error at parent (single linear model)
        error_parent = node.error
        df_parent = node.df
        
        # Error at children (two linear models)
        error_children = node.left.error + node.right.error
        df_children = node.left.df + node.right.df
        
        # Degrees of freedom for the test
        # Standard: One model has (p+1) params, two models have 2(p+1) params
        # The difference is (p+1) additional parameters
        # 
        # Karalič 1992 Academic Note:
        # The paper suggests that finding the best split point itself consumes
        # a degree of freedom. Setting account_for_split_cost=True adds +1 to
        # the numerator for more conservative pruning.
        df_numerator = df_parent - df_children
        if self.account_for_split_cost:
            df_numerator += 1  # Account for split-point search
        
        df_denominator = df_children
        
        if df_numerator <= 0 or df_denominator <= 0:
            # Cannot perform test, keep split
            return False
        
        # F-statistic: tests if reduction in error is significant
        # F = [(SSE_parent - SSE_children) / df_num] / [SSE_children / df_denom]
        error_reduction = error_parent - error_children
        
        if error_reduction <= 0 or error_children <= 0:
            # Split doesn't reduce error or no error in children
            return True  # Prune
        
        f_statistic = (error_reduction / df_numerator) / (error_children / df_denominator)
        
        # Critical F-value (approximation using significance level)
        # For simplicity, use a threshold-based approach
        # f_critical for typical cases (α=0.05) is around 2-4
        from scipy import stats
        try:
            f_critical = stats.f.ppf(1 - self.significance_level, df_numerator, df_denominator)
        except:
            # Fallback if scipy not available
            f_critical = 3.84  # Approximate for many common cases
        
        # Prune if F-statistic is less than critical value
        # (i.e., improvement is not statistically significant)
        return f_statistic < f_critical
    
    def _prune_tree(self, node: RETISNode, X: np.ndarray, y: np.ndarray):
        
        if node.is_leaf or node.left is None or node.right is None:
            return
        
        # Get masks for children
        left_mask = X[:, node.feature_idx] <= node.threshold
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # Recursively prune children first (bottom-up)
        self._prune_tree(node.left, X_left, y_left)
        self._prune_tree(node.right, X_right, y_right)
        
        # Now test if this split should be pruned
        should_prune = self._f_test_prune(node, X, y)
        
        if should_prune:
            # Convert to leaf: keep the linear model already fitted at this node
            node.is_leaf = True
            node.left = None
            node.right = None
    
    def _predict_node(self, x: np.ndarray, node: RETISNode) -> float:
        if node.is_leaf:
            # Use the linear model at this leaf
            x_aug = np.concatenate([[1.0], x])
            return float(np.dot(x_aug, node.model.coefficients))
        
        # Traverse to appropriate child
        if x[node.feature_idx] <= node.threshold:
            return self._predict_node(x, node.left)
        else:
            return self._predict_node(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X)
        predictions = np.array([self._predict_node(x, self.root) for x in X])
        return predictions
    
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
# Example usage
if __name__ == "__main__":
    print("RETIS - Correct Implementation (Karalic 1992)")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    # Create target with different regimes
    y = np.where(X[:, 0] > 0,
                 2*X[:, 0] + 3*X[:, 1] + np.random.randn(n_samples) * 0.5,
                 -X[:, 0] + 2*X[:, 2] + np.random.randn(n_samples) * 0.5)
    
    # Train-test split
    split_idx = int(0.7 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Fit RETIS model
    print("\nTraining RETIS model...")
    retis = RETIS(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        significance_level=0.05
    )
    retis.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = retis.predict(X_train)
    y_pred_test = retis.predict(X_test)
    
    # Metrics
    train_mse = np.mean((y_train - y_pred_train) ** 2)
    test_mse = np.mean((y_test - y_pred_test) ** 2)
    train_r2 = retis.score(X_train, y_train)
    test_r2 = retis.score(X_test, y_test)
    
    print(f"\n{'Tree Structure':─^60}")
    print(f"  Number of leaves: {retis.get_n_leaves()}")
    print(f"  Tree depth: {retis.get_depth()}")
    
    print(f"\n{'Training Performance':─^60}")
    print(f"  MSE: {train_mse:.4f}")
    print(f"  R²:  {train_r2:.4f}")
    
    print(f"\n{'Test Performance':─^60}")
    print(f"  MSE: {test_mse:.4f}")
    print(f"  R²:  {test_r2:.4f}")
    
    print("\n" + "=" * 60)
    print("Key RETIS Features Implemented:")
    print("  ✓ Linear models at ALL nodes (not just leaves)")
    print("  ✓ F-test based statistical pruning")
    print("  ✓ Two-phase approach: grow then prune")
    print("  ✓ Splits based on error reduction")
    print("\nAcademic Completeness Options:")
    print("  • account_for_split_cost: Adds +1 df for threshold search")
    print("  • use_variable_selection: Stepwise selection in leaves")
    print("  • Mean fallback: Handles singular matrices (practical)")
    print("=" * 60)
    
    # Demonstrate the academic options
    print("\n" + "=" * 60)
    print("Testing Academic Options:")
    print("=" * 60)
    
    # Test with split cost accounting
    retis_conservative = RETIS(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        significance_level=0.05,
        account_for_split_cost=True  # More conservative pruning
    )
    retis_conservative.fit(X_train, y_train)
    print(f"\nWith split-cost accounting:")
    print(f"  Leaves: {retis_conservative.get_n_leaves()} (more pruning expected)")
    print(f"  Test R²: {retis_conservative.score(X_test, y_test):.4f}")
    
    # Test with variable selection (if scipy available)
    try:
        retis_varsel = RETIS(
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            significance_level=0.05,
            use_variable_selection=True  # Stepwise variable selection
        )
        retis_varsel.fit(X_train, y_train)
        print(f"\nWith variable selection:")
        print(f"  Leaves: {retis_varsel.get_n_leaves()}")
        print(f"  Test R²: {retis_varsel.score(X_test, y_test):.4f}")
    except:
        print("\nVariable selection requires scipy (skipped)")
    
    print("\n" + "=" * 60)
"""
