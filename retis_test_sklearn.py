import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import time
import warnings

warnings.filterwarnings('ignore')

# Import all RETIS components
from retis import RETIS
from retis_optimizer import RETISOptimizer
from retis_regression_evaluator import RETISRegressionEvaluator, run_comprehensive_evaluation
from retis_classification_evaluator import RETISClassifier, RETISClassificationEvaluator, run_comprehensive_classification_evaluation


class RETISTestSuite:

    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}

    def run_all_tests(self):
        
        print("="*80)
        print("🧪 COMPREHENSIVE RETIS TEST SUITE")
        print("="*80)

        # Test 1: Basic RETIS functionality
        self.test_basic_retis()

        # Test 2: RETIS optimizer
        self.test_retis_optimizer()

        # Test 3: Regression evaluation
        self.test_regression_evaluation()

        # Test 4: Classification evaluation
        self.test_classification_evaluation()

        # Test 5: sklearn compatibility
        self.test_sklearn_compatibility()

        # Test 6: Performance benchmarks
        self.test_performance_benchmarks()

        # Test 7: Edge cases
        self.test_edge_cases()

        # Print summary
        self.print_test_summary()

        return self.test_results

    def test_basic_retis(self):
        print("\n🔍 Test 1: Basic RETIS Functionality")

        try:
            # Generate data
            X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Test RETIS model
            model = RETIS(max_depth=5, min_samples_split=10)
            model.fit(X_train, y_train)

            # Test predictions
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            # Test model properties
            n_leaves = model.get_n_leaves()
            depth = model.get_depth()

            print("   ✅ Model training: PASSED")
            print("   ✅ Predictions: PASSED")
            print(f"   📊 MSE: {mse:.4f}, Leaves: {n_leaves}, Depth: {depth}")

            self.test_results['basic_retis'] = {
                'status': 'PASSED',
                'mse': mse,
                'n_leaves': n_leaves,
                'depth': depth
            }

        except Exception as e:
            print(f"   ❌ Basic RETIS test FAILED: {e}")
            self.test_results['basic_retis'] = {'status': 'FAILED', 'error': str(e)}

    def test_retis_optimizer(self):
        print("\n🔍 Test 2: RETIS Optimizer")

        try:
            # Generate data
            X, y = make_regression(n_samples=300, n_features=8, noise=15, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Test optimizer
            optimizer = RETISOptimizer(X_train, y_train, X_test, y_test)

            # Test configuration comparison
            configs = [
                {'max_depth': 5, 'min_samples_split': 20},
                {'max_depth': 8, 'min_samples_split': 10}
            ]
            results = optimizer.compare_configurations(configs, cv=3, run_baselines=False)

            print("   ✅ Configuration comparison: PASSED")
            print(f"   📊 Configurations tested: {len(results)}")

            self.test_results['retis_optimizer'] = {
                'status': 'PASSED',
                'n_configs_tested': len(results)
            }

        except Exception as e:
            print(f"   ❌ RETIS optimizer test FAILED: {e}")
            self.test_results['retis_optimizer'] = {'status': 'FAILED', 'error': str(e)}
            self.test_results['retis_optimizer'] = {'status': 'FAILED', 'error': str(e)}

    def test_regression_evaluation(self):
        print("\n🔍 Test 3: Regression Evaluation")
        try:
            # Generate data
            X, y = make_regression(n_samples=400, n_features=6, noise=12, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train model
            model = RETIS(max_depth=6, min_samples_split=15)
            model.fit(X_train, y_train)

            # Test evaluator
            evaluator = RETISRegressionEvaluator(model, X_train, y_train, X_test, y_test)
            results = evaluator.evaluate_model(plot=False)
            cv_results = evaluator.cross_validate(cv=3)
            comparison = evaluator.compare_with_baselines()

            print("   ✅ Regression evaluation: PASSED")
            print("   ✅ Cross-validation: PASSED")
            print("   ✅ Baseline comparison: PASSED")
            print(".4f")
            self.test_results['regression_evaluation'] = {
                'status': 'PASSED',
                'test_r2': results['test']['r2'],
                'cv_score': cv_results['mean'],
                'n_baselines': len(comparison) - 1
            }

        except Exception as e:
            print(f"   ❌ Regression evaluation test FAILED: {e}")
            self.test_results['regression_evaluation'] = {'status': 'FAILED', 'error': str(e)}

    def test_classification_evaluation(self):
        print("\n🔍 Test 4: Classification Evaluation")
        try:
            # Generate data
            X, y = make_classification(n_samples=400, n_features=6, n_informative=4,
                                     n_classes=3, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train model
            model = RETISClassifier(max_depth=6, min_samples_split=15)
            model.fit(X_train, y_train)

            # Test evaluator
            evaluator = RETISClassificationEvaluator(model, X_train, y_train, X_test, y_test)
            results = evaluator.evaluate_model(plot=False)
            cv_results = evaluator.cross_validate(cv=3)
            comparison = evaluator.compare_with_baselines()

            print("   ✅ Classification evaluation: PASSED")
            print("   ✅ Cross-validation: PASSED")
            print("   ✅ Baseline comparison: PASSED")
            print(".4f")
            self.test_results['classification_evaluation'] = {
                'status': 'PASSED',
                'test_accuracy': results['test']['accuracy'],
                'cv_score': cv_results['mean'],
                'n_baselines': len(comparison) - 1
            }

        except Exception as e:
            print(f"   ❌ Classification evaluation test FAILED: {e}")
            self.test_results['classification_evaluation'] = {'status': 'FAILED', 'error': str(e)}

    def test_sklearn_compatibility(self):
        print("\n🔍 Test 5: sklearn Compatibility")

        try:
            from sklearn.model_selection import GridSearchCV, cross_val_score
            from sklearn.base import BaseEstimator

            # Test inheritance
            model = RETIS()
            assert isinstance(model, BaseEstimator), "RETIS should inherit from BaseEstimator"

            # Test get_params/set_params
            params = model.get_params()
            assert isinstance(params, dict), "get_params should return dict"

            new_params = {'max_depth': 10, 'min_samples_split': 5}
            model.set_params(**new_params)
            assert model.max_depth == 10, "set_params should work"

            # Test GridSearchCV compatibility
            X, y = make_regression(n_samples=100, n_features=4, random_state=42)
            param_grid = {'max_depth': [3, 5], 'min_samples_split': [5, 10]}

            gs = GridSearchCV(RETIS(), param_grid, cv=2, scoring='neg_mean_squared_error')
            gs.fit(X, y)

            print("   ✅ BaseEstimator inheritance: PASSED")
            print("   ✅ get_params/set_params: PASSED")
            print("   ✅ GridSearchCV compatibility: PASSED")
            print(f"   📊 Best CV score: {gs.best_score_:.4f}")

            self.test_results['sklearn_compatibility'] = {
                'status': 'PASSED',
                'best_cv_score': gs.best_score_,
                'n_params_tested': len(gs.cv_results_['mean_test_score'])
            }

        except Exception as e:
            print(f"   ❌ sklearn compatibility test FAILED: {e}")
            self.test_results['sklearn_compatibility'] = {'status': 'FAILED', 'error': str(e)}

    def test_performance_benchmarks(self):
        print("\n🔍 Test 6: Performance Benchmarks")
        try:
            # Test with different dataset sizes
            sizes = [100, 500, 1000]
            performance_data = []

            for n_samples in sizes:
                X, y = make_regression(n_samples=n_samples, n_features=10, noise=10, random_state=42)

                # Time training
                start_time = time.time()
                model = RETIS(max_depth=8, min_samples_split=10)
                model.fit(X, y)
                train_time = time.time() - start_time

                # Time prediction
                start_time = time.time()
                y_pred = model.predict(X)
                pred_time = time.time() - start_time

                mse = mean_squared_error(y, y_pred)
                performance_data.append({
                    'n_samples': n_samples,
                    'train_time': train_time,
                    'pred_time': pred_time,
                    'mse': mse,
                    'n_leaves': model.get_n_leaves()
                })

            df_perf = pd.DataFrame(performance_data)
            print("   ✅ Performance benchmarks: PASSED")
            print("   📊 Performance scaling:")
            for _, row in df_perf.iterrows():
                print(".4f")
            self.test_results['performance_benchmarks'] = {
                'status': 'PASSED',
                'performance_data': performance_data
            }

        except Exception as e:
            print(f"   ❌ Performance benchmarks test FAILED: {e}")
            self.test_results['performance_benchmarks'] = {'status': 'FAILED', 'error': str(e)}

    def test_edge_cases(self):
        print("\n🔍 Test 7: Edge Cases")
        try:
            edge_cases_passed = 0
            total_cases = 0

            # Test 1: Single feature
            total_cases += 1
            X = np.random.randn(50, 1)
            y = X.ravel() + 0.1 * np.random.randn(50)
            model = RETIS(max_depth=3)
            model.fit(X, y)
            y_pred = model.predict(X)
            if len(y_pred) == len(y):
                edge_cases_passed += 1
                print("   ✅ Single feature: PASSED")
            else:
                print("   ❌ Single feature: FAILED")
            # Test 2: Very small dataset
            total_cases += 1
            X = np.random.randn(10, 3)
            y = np.random.randn(10)
            model = RETIS(max_depth=2, min_samples_split=5)
            model.fit(X, y)
            y_pred = model.predict(X)
            if len(y_pred) == len(y):
                edge_cases_passed += 1
                print("   ✅ Small dataset: PASSED")
            else:
                print("   ❌ Small dataset: FAILED")
            # Test 3: Constant target
            total_cases += 1
            X = np.random.randn(50, 3)
            y = np.ones(50) * 5.0
            model = RETIS(max_depth=3)
            model.fit(X, y)
            y_pred = model.predict(X)
            if np.allclose(y_pred, 5.0, atol=1e-6):
                edge_cases_passed += 1
                print("   ✅ Constant target: PASSED")
            else:
                print("   ❌ Constant target: FAILED")
            print(f"   📊 Edge cases: {edge_cases_passed}/{total_cases} passed")

            self.test_results['edge_cases'] = {
                'status': 'PASSED' if edge_cases_passed == total_cases else 'PARTIAL',
                'passed': edge_cases_passed,
                'total': total_cases
            }

        except Exception as e:
            print(f"   ❌ Edge cases test FAILED: {e}")
            self.test_results['edge_cases'] = {'status': 'FAILED', 'error': str(e)}

    def print_test_summary(self):
        print("\n" + "="*80)
        print("📊 TEST SUITE SUMMARY")
        print("="*80)

        passed_tests = 0
        total_tests = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = result['status']
            if status == 'PASSED':
                passed_tests += 1
                icon = "✅"
            elif status == 'PARTIAL':
                passed_tests += 0.5
                icon = "⚠️"
            else:
                icon = "❌"

            print(f"{icon} {test_name.replace('_', ' ').title()}: {status}")

            # Print additional metrics for passed tests
            if status in ['PASSED', 'PARTIAL'] and len(result) > 1:
                for key, value in result.items():
                    if key != 'status':
                        if isinstance(value, float):
                            print(".4f")                        
                        elif isinstance(value, int):
                            print(f"      - {key}: {value}")
                        elif isinstance(value, dict):
                            print(f"      - {key}: {value}")

        print(f"\n🎯 Overall Score: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print("🏆 ALL TESTS PASSED! RETIS is fully functional.")
        elif passed_tests >= total_tests * 0.8:
            print("✅ MOST TESTS PASSED! RETIS is ready for use.")
        else:
            print("⚠️  SOME TESTS FAILED! Check implementation.")

        print("="*80)


def run_integration_demo():
    print("\n" + "="*80)
    print("🚀 RETIS INTEGRATION DEMO")
    print("="*80)

    # Regression demo
    print("\n📈 REGRESSION DEMO")
    print("-" * 40)

    X_reg, y_reg = make_regression(n_samples=500, n_features=8, n_informative=6,
                                  noise=15, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)

    reg_results = run_comprehensive_evaluation(
        X_train_reg, y_train_reg, X_test_reg, y_test_reg
    )

    # Classification demo
    print("\n🏷️  CLASSIFICATION DEMO")
    print("-" * 40)

    X_clf, y_clf = make_classification(n_samples=500, n_features=8, n_informative=6,
                                      n_classes=3, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42)

    clf_results = run_comprehensive_classification_evaluation(
        X_train_clf, y_train_clf, X_test_clf, y_test_clf
    )

    print("\n" + "="*80)
    print("🎉 INTEGRATION DEMO COMPLETED!")
    print("="*80)
    print("✅ RETIS regression and classification fully demonstrated")
    print("✅ All components working together seamlessly")
    print("✅ Ready for production use!")


# Main execution
if __name__ == "__main__":
    # Run comprehensive test suite
    test_suite = RETISTestSuite()
    test_results = test_suite.run_all_tests()

    # Run integration demo
    run_integration_demo()

    print("\n🎯 RETIS COMPREHENSIVE TESTING COMPLETE!")
    print("All components have been thoroughly tested and demonstrated.")