# RETIS Project - Karalic 1992 Implementation

[Presentation](https://www.canva.com/design/DAG7sJbv9q0/VeGo8GmWHBytwUHZeiOXZA/edit?utm_content=DAG7sJbv9q0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## Project Summary
RETIS (Regression Tree with Instance-based Selection) is a **complete implementation** of the Karalic 1992 algorithm, built entirely from scratch using only NumPy. The project emphasizes **algorithmic fidelity** while incorporating modern regularization techniques to prevent overfitting.

## Key Achievements
- **Karalic 1992 Fidelity**: Accurate implementation with linear models in leaves
- **Strong Generalization**: Minimal train-test gaps (regression ΔR²=-0.03, classification Δaccuracy=0.01)
- **Anti-Overfitting Pipeline**: Ridge regularization, F-test pruning, shuffled CV
- **From-Scratch Implementation**: Zero sklearn dependencies in core algorithms
- **Complete ML Pipeline**: 19+ custom metrics, cross-validation, evaluators

## Performance Results (After Regularization)
- **Classification**: 85.8% test accuracy with only 0.95% train-test gap
- **Regression**: Test R² = 0.151 with good generalization (ΔR² = -0.03)
- **Validation**: Consistent 5-fold CV performance (low variance)
- **Baseline Comparison**: Outperforms mean/median predictors significantly
- **Tree Structure**: Compact trees (3-5 leaves, depth 2-4) favor interpretability

## Core Algorithmic Features (Karalic 1992 + Regularization)
- **Linear Models in Leaves**: Fits linear models at leaf nodes for local approximation
- **F-Test Pruning**: Statistically-motivated post-hoc pruning using Fisher's F-test
- **Ridge Regularization (L2)**: Integrated into linear models to control coefficient magnitude
- **Shuffled Cross-Validation**: Uses stratified fold splitting for stable CV estimates
- **Conservative Defaults**: Balanced hyperparameters (max_depth=8, min_samples_split=20, m_estimate=2.0)
- **Threshold Capping**: Limits candidate split thresholds to 25 to reduce noisy splits
- **Split Cost Accounting**: Adjusts degrees of freedom for threshold search cost

###  Core Implementation Files

#### `retis.py` (700+ lines) - **Karalic 1992 RETIS with Ridge Regularization**
**Purpose**: Core implementation of RETIS algorithm with anti-overfitting mechanisms
- `LinearRegressionCustom`: Custom linear regression with optional Ridge (L2) regularization
  - `fit(X, y, ridge_alpha=0.0)`: Solves normal equations with optional L2 penalty
  - Handles singular matrices gracefully with fallback to mean
- `BaseEstimator`: Parameter management for sklearn compatibility
  - `get_params(deep=True)`: Inspects `__init__` signature to extract only constructor parameters
  - Prevents runtime attributes from leaking into cloning
- `RegressorMixin`: Provides `score()` method (R² for regression)
- `RETISNode`: Tree node structure with model storage
- `RETIS`: Main tree implementation
  - **Hyperparameters**:
    - `max_depth=8`: Maximum tree depth (conservative default)
    - `min_samples_split=20`: Minimum samples to attempt a split
    - `min_samples_leaf=10`: Minimum samples per leaf
    - `significance_level=0.01`: F-test threshold (stricter = more pruning)
    - `min_error_reduction=0.01`: Minimum relative error reduction to accept split
    - `account_for_split_cost=True`: Adjust df for threshold search cost
    - `m_estimate=2.0`: Ridge regularization parameter (lambda)
    - `max_threshold_candidates=25`: Cap split-threshold candidates
  - **Methods**:
    - `fit(X, y)`: Two-phase: grow tree, then prune with F-test
    - `_fit_linear_model()`: Ajust linear model with optional Ridge at each node
    - `_find_best_split()`: Search for best split using linear model MSE
    - `_f_test_prune()`: Statistical significance test for splitting
    - `predict()`: Traverse tree using linear models at leaf nodes
**Key Features**: 
- **Ridge Regularization**: L2 penalty prevents coefficient explosion in leaves
- **F-Test Pruning**: Removes non-significant splits (conservative)
- **Shuffled CV Support**: Works with stratified fold splitting
- **sklearn-compatible**: Full BaseEstimator/RegressorMixin interface

#### `custom_metrics.py` (360+ lines)
**Purpose**: Complete metrics library built from scratch (no sklearn)
- **Local KFold Class**: Implements stratified k-fold splitting with optional shuffle
- **Classification Metrics (10)**:
  - `accuracy_score()`, `precision_score()`, `recall_score()`, `f1_score()`
  - `confusion_matrix()`, `classification_report()` (with dict/string output options)
  - `roc_auc_binary()`, `roc_curve()`, `precision_recall_curve()`
- **Regression Metrics (9)**:
  - `mse()`, `rmse()`, `mae()`, `r2_score()`
  - `median_absolute_error()`, `mean_absolute_percentage_error()`
  - `explained_variance_score()`, `relative_rmse()`, `relative_mae()`
- **Cross-Validation Utilities**:
  - `custom_cross_val_score()`: K-fold CV with support for R², MSE, accuracy scoring
  - `train_test_split_custom()`: Random train-test split with reproducible seed
- **ClassMetrics Class**: Static methods for all metrics
**Key Feature**: All 19+ metrics without any sklearn dependency

### Testing and Evaluation Files

#### `retis_test_all.py` (440+ lines)
**Purpose**: Main comprehensive test suite with integration demo
- **7 Core Tests**:
  1. **Basic RETIS**: Smoke test of fit/predict/score
  2. **RETIS Optimizer**: Configuration comparison with shuffled CV
  3. **Regression Evaluation**: Full pipeline with 3-fold CV and baseline comparison
  4. **Classification Evaluation**: Multi-class One-vs-Rest with metrics
  5. **sklearn Compatibility**: get_params/set_params/simple_grid_search
  6. **Performance Benchmarks**: Scaling tests on increasing dataset sizes
  7. **Edge Cases**: Single feature, small dataset, constant target
- **Integration Demo**: End-to-end regression + classification evaluation with optimizer
**Key Results**:
- All 7 tests pass with overfitting fixed
- Classification test accuracy gap: 0.95% (good generalization)
- Regression R² gap: -0.03 (excellent generalization)
- Compact trees maintained (3-5 leaves, depth 2-4)

#### `retis_regression_evaluator.py` (350+ lines)
**Purpose**: Regression evaluation with conservative demo configs
- `RETISRegressionEvaluator`: Complete evaluation pipeline
  - `calculate_all_metrics()`: MSE, RMSE, MAE, R², variance, MAPE
  - `evaluate_model()`: Full train/test evaluation with overfitting analysis
  - `cross_validate()`: K-fold CV with configurable scoring
  - `compare_with_baselines()`: Compare vs. mean/median predictors
  - `_create_evaluation_plots()`: Predicted vs. actual, residuals, error distribution
- `run_comprehensive_evaluation()`: Main demo function
  - **Conservative Demo Configs**:
    - Config 1: depth=4, split=30, leaf=20, lambda=5.0, min_split_improvement=0.01
    - Config 2: depth=6, split=30, leaf=15, lambda=3.0, min_split_improvement=0.01
    - Config 3: depth=6, split=40, leaf=20, lambda=2.0, min_split_improvement=0.02
  - Selects best config by cross-validated MSE
  - Performs 5-fold shuffled CV
**Key Feature**: Built-in overfitting analysis and reduction strategies

#### `retis_classification_evaluator.py` (460+ lines)
**Purpose**: Classification evaluation with conservative demo configs and plotting fixes
- `RETISClassifier`: One-vs-Rest wrapper for multi-class classification
  - `fit()`: Train binary RETIS models (one per class)
  - `predict_proba()`: Probability predictions with normalization
  - `predict()`: Hard class predictions
- `RETISClassificationEvaluator`: Full evaluation pipeline
  - `calculate_all_metrics()`: Accuracy, precision, recall, F1, AUC (macro/micro)
  - `evaluate_model()`: Train/test evaluation with 2D axes normalization fix
  - `cross_validate()`: K-fold CV with RETISClassifierWrapper
  - `compare_with_baselines()`: Compare vs. random and majority classifiers
  - `_create_evaluation_plots()`: Confusion matrix, classification report, ROC/PR curves
    - **Plotting Fix**: Normalizes axes to 2D array for both binary (2×2) and multiclass (1×2) layouts
- `run_comprehensive_classification_evaluation()`: Main demo function
  - **Conservative Demo Configs**: Same regularization as regression
  - Selects best config by accuracy
  - Performs 5-fold shuffled CV
**Key Features**: 
- Fixed plotting error (1D array → 2D normalization)
- `classification_report()` with `target_names` and `output_dict` support
- Built-in overfitting analysis

### Legacy/Compatibility Files

#### `retis_optimizer.py` (350+ lines)
**Purpose**: Hyperparameter optimization and comparison tools
- `StandardScalerCustom`: Feature scaling (fit/transform/fit_transform)
- `RETISOptimizer`: Grid search and model comparison
  - `preprocess_data()`: Scale train/test data with custom scaler
  - `grid_search_optimization()`: Full grid search over 7-parameter combinations
  - `manual_optimization_guide()`: Data-driven hyperparameter recommendations
  - `compare_configurations()`: Compare multiple configs with shuffled K-fold CV
    - **Key Fix**: Shuffled fold indices instead of contiguous blocks
    - **New Feature**: Optional `run_baselines=True` for baseline MSE computation
  - `evaluate_model()`: Comprehensive evaluation with multiple metrics
**Key Features**:
- **Shuffled K-Fold**: Reduces bias in CV estimates
- **Baseline Support**: Optional baseline MSE for comparisons
- **Conservative Defaults**: Sensible recommendations based on data size

#### `retis_regression_evaluator.py` (deprecated)
**Purpose**: Original regression evaluator (sklearn-dependent)
**Status**: Replaced by modern implementation with anti-overfitting

#### `retis_classification_evaluator.py` (deprecated)
**Purpose**: Original classification evaluator (sklearn-dependent)
**Status**: Replaced by modern implementation with plotting fixes

### Documentation and Demo Files

#### `FINAL_STATUS.txt` (244 lines)
**Purpose**: Project completion status and architecture summary
- Detailed breakdown of all implemented components
- Migration from sklearn to custom implementations
- Current status and achievements


### Configuration Files

#### `requirements.txt`
**Purpose**: Python dependencies
- `numpy>=1.21.0`: Core numerical computing
- `scikit-learn>=1.0.0`: Only for compatibility testing (not used in core implementation)

### Directory Structure

#### `__pycache__/` (auto-generated)
**Purpose**: Python bytecode cache
- Contains compiled .pyc files
- Auto-generated by Python interpreter

#### `.venv/` (virtual environment)
**Purpose**: Isolated Python environment
- Contains project dependencies
- Ensures reproducible environment

#### `.idea/` (IDE configuration)
**Purpose**: PyCharm/IntelliJ IDEA settings
- IDE-specific configuration files
- Not part of the codebase

## Architecture Highlights

### Anti-Overfitting Pipeline
1. **Ridge Regularization (L2)**: Integrated into linear models via `m_estimate` parameter
2. **F-Test Statistical Pruning**: Removes non-significant splits post-hoc
3. **Conservative Defaults**: Balanced hyperparameters prevent tree complexity
4. **Shuffled Cross-Validation**: Reduces bias in fold splitting
5. **Split Improvement Threshold**: Minimum relative reduction required (0.01)
6. **Threshold Capping**: Limits candidates to 25 to reduce noisy splits

### From-Scratch Philosophy
- **Zero sklearn dependencies** in core algorithms
- All metrics, cross-validation, and preprocessing custom-built
- Educational focus on understanding ML fundamentals
- Full parameter control through BaseEstimator interface

### Generalization Verification
- Regression: train R²=0.121 vs test R²=0.151 (gap=-0.03) → **Excellent**
- Classification: train acc=0.868 vs test acc=0.858 (gap=0.01) → **Excellent**
- Validation: 5-fold CV with low variance confirms stability

## Usage Flow
1. **Training**: `retis.py` - Core algorithm with all regularization
2. **Evaluation**: `retis_test_all.py` - Comprehensive 7-test suite
3. **Regression Demo**: `retis_regression_evaluator.py` - Full pipeline
4. **Classification Demo**: `retis_classification_evaluator.py` - Multi-class evaluation
5. **Metrics**: `custom_metrics.py` - 19+ evaluation metrics
6. **Optimization**: `retis_optimizer.py` - Hyperparameter search and tuning

---

## Authors

**Mohamed Reda Nkira** & **Ilias Boudrika** & **Youssef Es-saaidi**   
KDD | IATD-SI  
