import numpy as np
from typing import Dict, List, Tuple, Optional
from custom_metrics import CustomMetrics
from retis import RETIS
import urllib.request


class RETISClassifierCustom:

    def __init__(self, max_depth: int = 10, min_samples_split: int = 10,
                 min_samples_leaf: int = 5, m_estimate: float = 2.0,
                 min_mse_reduction: float = 0.01, max_threshold_candidates: int = 25,
                 use_fast_solver: bool = True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.m_estimate = m_estimate
        self.min_mse_reduction = min_mse_reduction
        self.max_threshold_candidates = max_threshold_candidates
        self.use_fast_solver = use_fast_solver

        self.models = {}
        self.classes_ = None
        self.n_classes_ = None
        self.is_binary_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RETISClassifierCustom':
       
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.is_binary_ = self.n_classes_ == 2

        if self.is_binary_:
            # Binary classification
            y_binary = (y == self.classes_[1]).astype(float)
            model = RETIS(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                m_estimate=self.m_estimate,
                min_mse_reduction=self.min_mse_reduction,
                max_threshold_candidates=self.max_threshold_candidates,
                use_fast_solver=self.use_fast_solver
            )
            model.fit(X, y_binary)
            self.models['binary'] = model
        else:
            # Multi-class: one-vs-rest
            for cls in self.classes_:
                y_ovr = (y == cls).astype(float)
                model = RETIS(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    m_estimate=self.m_estimate,
                    min_mse_reduction=self.min_mse_reduction,
                    max_threshold_candidates=self.max_threshold_candidates,
                    use_fast_solver=self.use_fast_solver
                )
                model.fit(X, y_ovr)
                self.models[cls] = model

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.is_binary_:
            prob_pos = self.models['binary'].predict(X)
            prob_pos = np.clip(prob_pos, 0, 1)
            prob_neg = 1 - prob_pos
            return np.column_stack([prob_neg, prob_pos])
        else:
            probs = []
            for cls in self.classes_:
                prob = self.models[cls].predict(X)
                prob = np.clip(prob, 0, 1)
                probs.append(prob)

            probs = np.column_stack(probs)
            # Normalize
            probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)
            return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def get_params(self, deep: bool = True) -> Dict:
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'm_estimate': self.m_estimate,
            'min_mse_reduction': self.min_mse_reduction,
            'max_threshold_candidates': self.max_threshold_candidates,
            'use_fast_solver': self.use_fast_solver
        }

    def set_params(self, **params) -> 'RETISClassifierCustom':
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class RETISClassificationEvaluatorCustom:

    def __init__(self, model: RETISClassifierCustom = None,
                 X_train: np.ndarray = None, y_train: np.ndarray = None,
                 X_test: np.ndarray = None, y_test: np.ndarray = None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.classes_ = None
        self.is_binary_ = None

    def evaluate_model(self, model: RETISClassifierCustom = None) -> Dict:
        if model is None:
            model = self.model

        if model is None:
            raise ValueError("No model provided")

        self.classes_ = model.classes_
        self.is_binary_ = model.is_binary_

        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        y_train_proba = model.predict_proba(self.X_train)
        y_test_proba = model.predict_proba(self.X_test)

        # Calculate metrics
        train_metrics = self._calculate_metrics(self.y_train, y_train_pred, y_train_proba)
        test_metrics = self._calculate_metrics(self.y_test, y_test_pred, y_test_proba)

        # Print results
        self._print_results(train_metrics, test_metrics)

        return {'train': train_metrics, 'test': test_metrics}

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_proba: np.ndarray) -> Dict[str, float]:
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = CustomMetrics.accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = CustomMetrics.precision_score(y_true, y_pred, average='macro')
        metrics['precision_micro'] = CustomMetrics.precision_score(y_true, y_pred, average='micro')
        metrics['recall_macro'] = CustomMetrics.recall_score(y_true, y_pred, average='macro')
        metrics['recall_micro'] = CustomMetrics.recall_score(y_true, y_pred, average='micro')
        metrics['f1_macro'] = CustomMetrics.f1_score(y_true, y_pred, average='macro')
        metrics['f1_micro'] = CustomMetrics.f1_score(y_true, y_pred, average='micro')

        # AUC
        try:
            if len(np.unique(y_true)) == 2:
                metrics['auc'] = CustomMetrics.roc_auc_binary(y_true, y_proba[:, 1])
        except:
            pass

        return metrics

    def _print_results(self, train_metrics: Dict, test_metrics: Dict):
        print("\n" + "="*70)
        print("🎯 RETIS CLASSIFICATION EVALUATION")
        print("="*70)

        print(f"\n📊 Training Set:")
        print(f"   Accuracy:      {train_metrics.get('accuracy', 0):.4f}")
        print(f"   Precision:     {train_metrics.get('precision_macro', 0):.4f}")
        print(f"   Recall:        {train_metrics.get('recall_macro', 0):.4f}")
        print(f"   F1 (macro):    {train_metrics.get('f1_macro', 0):.4f}")

        print(f"\n📈 Test Set:")
        print(f"   Accuracy:      {test_metrics.get('accuracy', 0):.4f}")
        print(f"   Precision:     {test_metrics.get('precision_macro', 0):.4f}")
        print(f"   Recall:        {test_metrics.get('recall_macro', 0):.4f}")
        print(f"   F1 (macro):    {test_metrics.get('f1_macro', 0):.4f}")

        overfitting = train_metrics.get('accuracy', 0) - test_metrics.get('accuracy', 0)
        print(f"\n⚠️  Overfitting gap: {overfitting:.4f}")

    def compare_with_baselines(self) -> Dict:
        retis_pred = self.model.predict(self.X_test)
        retis_acc = CustomMetrics.accuracy_score(self.y_test, retis_pred)

        # Baseline: Most frequent class
        most_frequent = np.argmax(np.bincount(self.y_train))
        baseline_pred = np.full_like(self.y_test, most_frequent)
        baseline_acc = CustomMetrics.accuracy_score(self.y_test, baseline_pred)

        results = {
            'RETIS': retis_acc,
            'Most_Frequent_Baseline': baseline_acc
        }

        print("\n🏁 Model Comparison:")
        for model_name, acc in results.items():
            print(f"   {model_name}: {acc:.4f}")

        return results


def load_adult_income() -> Tuple[np.ndarray, np.ndarray]:
    
    print("[+] Loading Adult Income dataset...")
    
    try:
        # Download dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        response = urllib.request.urlopen(url, timeout=10)
        data = response.read().decode('utf-8')
        
        # Parse CSV
        lines = data.strip().split('\n')
        X_list = []
        y_list = []
        
        # Numeric feature indices (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week)
        numeric_indices = [0, 2, 4, 10, 11, 12]
        
        for line in lines:
            if not line.strip():
                continue
            
            values = [v.strip() for v in line.split(',')]
            
            try:
                # Extract numeric features
                row = []
                for idx in numeric_indices:
                    if idx < len(values):
                        row.append(float(values[idx]))
                
                if len(row) == 6:
                    X_list.append(row)
                    # Target: 1 if >50K, 0 if <=50K
                    target = 1 if '>50K' in values[-1] else 0
                    y_list.append(target)
            except:
                continue
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        
        # Normalize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        print(f"[+] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"[+] Class distribution - Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")
        return X, y
    
    except Exception as e:
        print(f"[!] Failed to load from URL: {e}")
        print("[+] Generating synthetic Adult Income-like data...")
        # Fallback to synthetic data
        np.random.seed(42)
        n_samples = 30162
        X = np.random.randn(n_samples, 6) * np.array([15, 100000, 10, 15000, 5000, 10])
        # Create target with some correlation to features
        y = (X[:, 0] > 0) & (X[:, 5] > 0)
        y = y.astype(np.int32)
        return X, y

