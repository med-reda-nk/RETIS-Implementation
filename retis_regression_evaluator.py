import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')

# Import custom components
from retis import RETIS
from retis_optimizer import RETISOptimizer
from custom_metrics import CustomMetrics


class RETISRegressionEvaluator:

    def __init__(self, model: RETIS = None, X_train: np.ndarray = None, y_train: np.ndarray = None,
                 X_test: np.ndarray = None, y_test: np.ndarray = None):
  
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.metrics_train = {}
        self.metrics_test = {}
        self.cv_scores = {}

    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
       
        metrics = {}

        # Basic metrics using CustomMetrics
        metrics['mse'] = CustomMetrics.mse(y_true, y_pred)
        metrics['rmse'] = CustomMetrics.rmse(y_true, y_pred)
        metrics['mae'] = CustomMetrics.mae(y_true, y_pred)
        metrics['r2'] = CustomMetrics.r2_score(y_true, y_pred)
        metrics['explained_variance'] = CustomMetrics.explained_variance_score(y_true, y_pred)

        # Additional metrics using CustomMetrics
        metrics['mape'] = CustomMetrics.mean_absolute_percentage_error(y_true, y_pred)
        metrics['median_absolute_error'] = CustomMetrics.median_absolute_error(y_true, y_pred)

        # Relative metrics using CustomMetrics
        metrics['relative_rmse'] = CustomMetrics.relative_rmse(y_true, y_pred)
        metrics['relative_mae'] = CustomMetrics.relative_mae(y_true, y_pred)

        return metrics

    def evaluate_model(self, model: RETIS = None, plot: bool = True) -> Dict[str, Dict[str, float]]:
        
        if model is None:
            model = self.model

        if model is None:
            raise ValueError("No model provided for evaluation")

        # Get predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # Calculate metrics
        self.metrics_train = self.calculate_all_metrics(self.y_train, y_train_pred)
        self.metrics_test = self.calculate_all_metrics(self.y_test, y_test_pred)

        # Print results
        self._print_evaluation_results()

        # Create plots if requested
        if plot:
            self._create_evaluation_plots(y_train_pred, y_test_pred)

        return {
            'train': self.metrics_train,
            'test': self.metrics_test
        }

    def _print_evaluation_results(self):
        print("\n" + "="*70)
        print("🎯 RETIS REGRESSION MODEL EVALUATION")
        print("="*70)

        print(f"\n📊 Training Set Performance:")
        self._print_metrics(self.metrics_train)

        print(f"\n📈 Test Set Performance:")
        self._print_metrics(self.metrics_test)

        # Calculate overfitting indicators
        overfitting = self.metrics_train['r2'] - self.metrics_test['r2']
        print(f"\n⚠️  Overfitting Analysis:")
        print(f"   R² difference (train - test): {overfitting:.4f}")

        if overfitting > 0.1:
            print("   → Significant overfitting detected")
        elif overfitting < -0.1:
            print("   → Model may be underfitting")
        else:
            print("   → Good generalization balance")

    def _print_metrics(self, metrics: Dict[str, float]):
        print(f"   MSE:           {metrics['mse']:.4f}")
        print(f"   RMSE:          {metrics['rmse']:.4f}")
        print(f"   MAE:           {metrics['mae']:.4f}")
        print(f"   R²:            {metrics['r2']:.4f}")
        print(f"   Explained Var: {metrics['explained_variance']:.4f}")
        print(f"   MAPE:          {metrics['mape']:.4f}")
        print(f"   Rel. RMSE:     {metrics['relative_rmse']:.4f}")
        print(f"   Rel. MAE:      {metrics['relative_mae']:.4f}")

    def _create_evaluation_plots(self, y_train_pred: np.ndarray, y_test_pred: np.ndarray):
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('RETIS Regression Model Evaluation', fontsize=16)

            # Training set: Predicted vs Actual
            axes[0, 0].scatter(self.y_train, y_train_pred, alpha=0.6, color='blue')
            axes[0, 0].plot([self.y_train.min(), self.y_train.max()],
                           [self.y_train.min(), self.y_train.max()],
                           'r--', linewidth=2)
            axes[0, 0].set_xlabel('Actual Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title('Training Set: Predicted vs Actual')
            axes[0, 0].grid(True, alpha=0.3)

            # Test set: Predicted vs Actual
            axes[0, 1].scatter(self.y_test, y_test_pred, alpha=0.6, color='green')
            axes[0, 1].plot([self.y_test.min(), self.y_test.max()],
                           [self.y_test.min(), self.y_test.max()],
                           'r--', linewidth=2)
            axes[0, 1].set_xlabel('Actual Values')
            axes[0, 1].set_ylabel('Predicted Values')
            axes[0, 1].set_title('Test Set: Predicted vs Actual')
            axes[0, 1].grid(True, alpha=0.3)

            # Residuals plot
            train_residuals = self.y_train - y_train_pred
            test_residuals = self.y_test - y_test_pred

            axes[1, 0].scatter(y_train_pred, train_residuals, alpha=0.6, color='blue', label='Train')
            axes[1, 0].scatter(y_test_pred, test_residuals, alpha=0.6, color='green', label='Test')
            axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[1, 0].set_xlabel('Predicted Values')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title('Residuals Plot')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Prediction error distribution
            axes[1, 1].hist(train_residuals, bins=30, alpha=0.7, color='blue', label='Train', density=True)
            axes[1, 1].hist(test_residuals, bins=30, alpha=0.7, color='green', label='Test', density=True)
            axes[1, 1].set_xlabel('Prediction Error')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Prediction Error Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('retis_regression_evaluation.png', dpi=300, bbox_inches='tight')
            print("📊 Evaluation plots saved as 'retis_regression_evaluation.png'")
            plt.show()

        except ImportError:
            print("\n⚠️  Matplotlib not available for plotting")
        except Exception as e:
            print(f"\n⚠️  Error creating plots: {e}")

    def cross_validate(self, model: RETIS = None, cv: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict[str, float]:
        if model is None:
            model = self.model

        print(f"\n🔄 Performing {cv}-fold cross-validation...")

        # Get CV scores
        cv_scores = cross_val_score(model, self.X_train, self.y_train,
                                   cv=cv, scoring=scoring, n_jobs=-1)

        if scoring == 'neg_mean_squared_error':
            # Convert negative MSE to positive
            cv_scores = -cv_scores
            metric_name = 'MSE'
        elif scoring == 'r2':
            metric_name = 'R²'
        else:
            metric_name = scoring

        self.cv_scores = {
            'scores': cv_scores,
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores),
            'metric': metric_name
        }

        print(f"   {metric_name} scores: {cv_scores}")
        print(".4f",
              ".4f")
        return self.cv_scores

    def compare_with_baselines(self, baselines: List[Dict] = None) -> pd.DataFrame:
        
        if baselines is None:
            baselines = [
                {'name': 'Mean Predictor', 'train_pred': np.full_like(self.y_train, np.mean(self.y_train)),
                 'test_pred': np.full_like(self.y_test, np.mean(self.y_train))},
                {'name': 'Median Predictor', 'train_pred': np.full_like(self.y_train, np.median(self.y_train)),
                 'test_pred': np.full_like(self.y_test, np.median(self.y_train))}
            ]

        # Get RETIS predictions
        retis_train_pred = self.model.predict(self.X_train)
        retis_test_pred = self.model.predict(self.X_test)

        results = []

        # RETIS results
        retis_metrics = self.calculate_all_metrics(self.y_test, retis_test_pred)
        results.append({
            'Model': 'RETIS',
            'Test_MSE': retis_metrics['mse'],
            'Test_RMSE': retis_metrics['rmse'],
            'Test_MAE': retis_metrics['mae'],
            'Test_R2': retis_metrics['r2'],
            'Test_MAPE': retis_metrics['mape']
        })

        # Baseline results
        for baseline in baselines:
            baseline_metrics = self.calculate_all_metrics(self.y_test, baseline['test_pred'])
            results.append({
                'Model': baseline['name'],
                'Test_MSE': baseline_metrics['mse'],
                'Test_RMSE': baseline_metrics['rmse'],
                'Test_MAE': baseline_metrics['mae'],
                'Test_R2': baseline_metrics['r2'],
                'Test_MAPE': baseline_metrics['mape']
            })

        df = pd.DataFrame(results)
        print("\n🏁 Model Comparison:")
        print(df.to_string(index=False, float_format='%.4f'))

        return df

    def get_model_info(self) -> Dict[str, Union[int, float]]:
        if self.model is None:
            return {}

        info = {
            'n_leaves': self.model.get_n_leaves(),
            'depth': self.model.get_depth(),
            'n_features': self.model.n_features
        }

        # Get tree statistics
        def get_tree_stats(node):
            if node is None:
                return 0, 0, 0
            if node.is_leaf:
                return 1, 0, 1  # leaves, internal, total

            left_leaves, left_internal, left_total = get_tree_stats(node.left)
            right_leaves, right_internal, right_total = get_tree_stats(node.right)

            return left_leaves + right_leaves, left_internal + right_internal + 1, left_total + right_total + 1

        leaves, internal, total = get_tree_stats(self.model.root)
        info.update({
            'internal_nodes': internal,
            'total_nodes': total,
            'avg_samples_per_leaf': len(self.y_train) / leaves if leaves > 0 else 0
        })

        return info


def run_comprehensive_evaluation(X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               model_configs: List[Dict] = None) -> Dict:

    print("="*80)
    print("🔬 COMPREHENSIVE RETIS REGRESSION EVALUATION")
    print("="*80)

    if model_configs is None:
        model_configs = [
            {'max_depth': 5, 'min_samples_split': 20, 'm_estimate': 2.0},
            {'max_depth': 8, 'min_samples_split': 10, 'm_estimate': 1.5},
            {'max_depth': 12, 'min_samples_split': 5, 'm_estimate': 1.0}
        ]

    # Initialize optimizer
    optimizer = RETISOptimizer(X_train, y_train, X_test, y_test)

    # Compare configurations
    config_results = optimizer.compare_configurations(model_configs, cv=5, run_baselines=True)

    # Get best configuration
    best_config = config_results[np.argmin([r['test_mse'] for r in config_results])]['config']

    # Train best model
    best_model = RETIS(**best_config)
    best_model.fit(X_train, y_train)

    # Comprehensive evaluation
    evaluator = RETISRegressionEvaluator(best_model, X_train, y_train, X_test, y_test)
    eval_results = evaluator.evaluate_model(plot=True)
    cv_results = evaluator.cross_validate(cv=5)
    comparison_df = evaluator.compare_with_baselines()
    model_info = evaluator.get_model_info()

    # Summary
    print(f"\n🎯 FINAL RESULTS SUMMARY:")
    print(f"   Best Configuration: {best_config}")
    print(f"   Test R²: {eval_results['test']['r2']:.4f}")
    print(f"   Test RMSE: {eval_results['test']['rmse']:.4f}")
    print(f"   CV Score: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
    print(f"   Tree Depth: {model_info['depth']}")
    print(f"   Number of Leaves: {model_info['n_leaves']}")

    return {
        'config_results': config_results,
        'best_config': best_config,
        'evaluation': eval_results,
        'cv_results': cv_results,
        'comparison': comparison_df,
        'model_info': model_info,
        'best_model': best_model
    }


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=10, n_informative=7,
                          noise=20, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Run comprehensive evaluation
    results = run_comprehensive_evaluation(X_train, y_train, X_test, y_test)

    print("\n✅ Comprehensive evaluation completed!")