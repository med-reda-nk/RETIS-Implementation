import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')

# Import custom components
from retis import RETIS
from retis_optimizer import RETISOptimizer
from custom_metrics import CustomMetrics


class RETISClassifier:

    def __init__(self, **retis_params):
       
        self.retis_params = retis_params
        self.models = {}
        self.classes_ = None
        self.n_classes_ = None
        self.is_binary_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RETISClassifier':
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.is_binary_ = self.n_classes_ == 2

        if self.is_binary_:
            # Binary classification: train one model for positive class probability
            y_binary = (y == self.classes_[1]).astype(int)
            model = RETIS(**self.retis_params)
            model.fit(X, y_binary)
            self.models['binary'] = model
        else:
            # Multi-class: one-vs-rest approach
            for i, cls in enumerate(self.classes_):
                y_ovr = (y == cls).astype(int)
                model = RETIS(**self.retis_params)
                model.fit(X, y_ovr)
                self.models[cls] = model

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        
        if self.is_binary_:
            # Binary case
            prob_pos = self.models['binary'].predict(X)
            prob_pos = np.clip(prob_pos, 0, 1)  # Ensure valid probabilities
            prob_neg = 1 - prob_pos
            return np.column_stack([prob_neg, prob_pos])
        else:
            # Multi-class case
            probs = []
            for cls in self.classes_:
                prob = self.models[cls].predict(X)
                prob = np.clip(prob, 0, 1)
                probs.append(prob)
            probs = np.column_stack(probs)

            # Normalize to ensure probabilities sum to 1
            probs = probs / probs.sum(axis=1, keepdims=True)
            return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
       
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def get_params(self, deep=True):
        return self.retis_params.copy()

    def set_params(self, **params):
        self.retis_params.update(params)
        return self


class RETISClassificationEvaluator:
    
    def __init__(self, model: RETISClassifier = None, X_train: np.ndarray = None, y_train: np.ndarray = None,
                 X_test: np.ndarray = None, y_test: np.ndarray = None):
        
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.classes_ = None
        self.is_binary_ = None
        self.metrics_train = {}
        self.metrics_test = {}

    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                            y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        
        metrics = {}

        # Basic metrics using CustomMetrics
        metrics['accuracy'] = CustomMetrics.accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = CustomMetrics.precision_score(y_true, y_pred, average='macro')
        metrics['precision_micro'] = CustomMetrics.precision_score(y_true, y_pred, average='micro')
        metrics['recall_macro'] = CustomMetrics.recall_score(y_true, y_pred, average='macro')
        metrics['recall_micro'] = CustomMetrics.recall_score(y_true, y_pred, average='micro')
        metrics['f1_macro'] = CustomMetrics.f1_score(y_true, y_pred, average='macro')
        metrics['f1_micro'] = CustomMetrics.f1_score(y_true, y_pred, average='micro')

        # AUC metrics (if probabilities available)
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification using CustomMetrics
                    metrics['auc'] = CustomMetrics.roc_auc_binary(y_true, y_proba[:, 1])
                else:
                    # Multi-class AUC (one-vs-rest) - using sklearn for now as CustomMetrics doesn't have multi-class AUC
                    y_bin = label_binarize(y_true, classes=np.unique(y_true))
                    if y_bin.shape[1] == 1:
                        y_bin = np.column_stack([1 - y_bin.ravel(), y_bin.ravel()])
                    from sklearn.metrics import roc_auc_score
                    metrics['auc_macro'] = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='macro')
                    metrics['auc_micro'] = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='micro')
            except Exception as e:
                print(f"Warning: Could not calculate AUC: {e}")

        return metrics

    def evaluate_model(self, model: RETISClassifier = None, plot: bool = True) -> Dict[str, Dict[str, float]]:
        
        if model is None:
            model = self.model

        if model is None:
            raise ValueError("No model provided for evaluation")

        self.classes_ = model.classes_
        self.is_binary_ = model.is_binary_

        # Get predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        y_train_proba = model.predict_proba(self.X_train)
        y_test_proba = model.predict_proba(self.X_test)

        # Calculate metrics
        self.metrics_train = self.calculate_all_metrics(self.y_train, y_train_pred, y_train_proba)
        self.metrics_test = self.calculate_all_metrics(self.y_test, y_test_pred, y_test_proba)

        # Print results
        self._print_evaluation_results()

        # Create plots if requested
        if plot:
            self._create_evaluation_plots(y_test_pred, y_test_proba)

        return {
            'train': self.metrics_train,
            'test': self.metrics_test
        }

    def _print_evaluation_results(self):
        print("\n" + "="*70)
        print("🎯 RETIS CLASSIFICATION MODEL EVALUATION")
        print("="*70)

        print(f"\n📊 Training Set Performance:")
        self._print_metrics(self.metrics_train)

        print(f"\n📈 Test Set Performance:")
        self._print_metrics(self.metrics_test)

        # Calculate overfitting indicators
        if 'accuracy' in self.metrics_train and 'accuracy' in self.metrics_test:
            overfitting = self.metrics_train['accuracy'] - self.metrics_test['accuracy']
            print(f"\n⚠️  Overfitting Analysis:")
            print(f"   Accuracy difference (train - test): {overfitting:.4f}")

            if overfitting > 0.1:
                print("   → Significant overfitting detected")
            elif overfitting < -0.1:
                print("   → Model may be underfitting")
            else:
                print("   → Good generalization balance")

    def _print_metrics(self, metrics: Dict[str, float]):
        print(f"   Accuracy:      {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"   Precision (macro): {metrics.get('precision_macro', 'N/A'):.4f}")
        print(f"   Precision (micro): {metrics.get('precision_micro', 'N/A'):.4f}")
        print(f"   Recall (macro):    {metrics.get('recall_macro', 'N/A'):.4f}")
        print(f"   Recall (micro):    {metrics.get('recall_micro', 'N/A'):.4f}")
        print(f"   F1 (macro):        {metrics.get('f1_macro', 'N/A'):.4f}")
        print(f"   F1 (micro):        {metrics.get('f1_micro', 'N/A'):.4f}")

        if 'auc' in metrics:
            print(f"   AUC:              {metrics['auc']:.4f}")
        if 'auc_macro' in metrics:
            print(f"   AUC (macro):       {metrics['auc_macro']:.4f}")
        if 'auc_micro' in metrics:
            print(f"   AUC (micro):       {metrics['auc_micro']:.4f}")

    def _create_evaluation_plots(self, y_test_pred: np.ndarray, y_test_proba: np.ndarray):
        try:
            if self.is_binary_:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            else:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            fig.suptitle('RETIS Classification Model Evaluation', fontsize=16)

            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_test_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.classes_, yticklabels=self.classes_, ax=axes[0, 0])
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
            axes[0, 0].set_title('Confusion Matrix')

            # Classification Report
            report = classification_report(self.y_test, y_test_pred,
                                        target_names=[str(cls) for cls in self.classes_],
                                        output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            axes[0, 1].axis('off')
            axes[0, 1].table(cellText=report_df.round(4).values,
                            colLabels=report_df.columns,
                            rowLabels=report_df.index,
                            cellLoc='center', loc='center')
            axes[0, 1].set_title('Classification Report')

            if self.is_binary_:
                # ROC Curve
                fpr, tpr, _ = roc_curve(self.y_test, y_test_proba[:, 1])
                auc_score = roc_auc_score(self.y_test, y_test_proba[:, 1])

                axes[1, 0].plot(fpr, tpr, color='darkorange', linewidth=2,
                               label=f'ROC curve (AUC = {auc_score:.4f})')
                axes[1, 0].plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
                axes[1, 0].set_xlim([0.0, 1.0])
                axes[1, 0].set_ylim([0.0, 1.05])
                axes[1, 0].set_xlabel('False Positive Rate')
                axes[1, 0].set_ylabel('True Positive Rate')
                axes[1, 0].set_title('ROC Curve')
                axes[1, 0].legend(loc="lower right")
                axes[1, 0].grid(True, alpha=0.3)

                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(self.y_test, y_test_proba[:, 1])

                axes[1, 1].plot(recall, precision, color='blue', linewidth=2)
                axes[1, 1].set_xlabel('Recall')
                axes[1, 1].set_ylabel('Precision')
                axes[1, 1].set_title('Precision-Recall Curve')
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('retis_classification_evaluation.png', dpi=300, bbox_inches='tight')
            print("📊 Evaluation plots saved as 'retis_classification_evaluation.png'")
            plt.show()

        except ImportError:
            print("\n⚠️  Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            print(f"\n⚠️  Error creating plots: {e}")

    def cross_validate(self, model: RETISClassifier = None, cv: int = 5,
                      scoring: str = 'accuracy') -> Dict[str, float]:
        
        if model is None:
            model = self.model

        print(f"\n🔄 Performing {cv}-fold cross-validation...")

        # Create a wrapper for sklearn compatibility
        class RETISClassifierWrapper:
            def __init__(self, retis_classifier):
                self.retis_classifier = retis_classifier

            def fit(self, X, y):
                return self.retis_classifier.fit(X, y)

            def predict(self, X):
                return self.retis_classifier.predict(X)

            def score(self, X, y):
                y_pred = self.predict(X)
                return accuracy_score(y, y_pred)

        wrapper = RETISClassifierWrapper(model)
        cv_scores = cross_val_score(wrapper, self.X_train, self.y_train,
                                   cv=cv, scoring='accuracy', n_jobs=-1)

        cv_results = {
            'scores': cv_scores,
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores),
            'metric': 'Accuracy'
        }

        print(f"   Accuracy scores: {cv_scores}")
        print(".4f",
              ".4f")
        return cv_results

    def compare_with_baselines(self) -> pd.DataFrame:
       
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        # Get RETIS predictions
        retis_pred = self.model.predict(self.X_test)
        retis_proba = self.model.predict_proba(self.X_test)

        results = []

        # RETIS results
        retis_metrics = self.calculate_all_metrics(self.y_test, retis_pred, retis_proba)
        results.append({
            'Model': 'RETIS',
            'Accuracy': retis_metrics.get('accuracy', 0),
            'Precision': retis_metrics.get('precision_macro', 0),
            'Recall': retis_metrics.get('recall_macro', 0),
            'F1': retis_metrics.get('f1_macro', 0),
            'AUC': retis_metrics.get('auc', retis_metrics.get('auc_macro', 0))
        })

        # Baseline classifiers
        baselines = [
            ('Random', DummyClassifier(strategy='uniform', random_state=42)),
            ('Most Frequent', DummyClassifier(strategy='most_frequent', random_state=42)),
            ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
            ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42))
        ]

        for name, clf in baselines:
            clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

            try:
                proba = clf.predict_proba(self.X_test)
            except:
                proba = None

            metrics = self.calculate_all_metrics(self.y_test, pred, proba)
            results.append({
                'Model': name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision_macro', 0),
                'Recall': metrics.get('recall_macro', 0),
                'F1': metrics.get('f1_macro', 0),
                'AUC': metrics.get('auc', metrics.get('auc_macro', 0))
            })

        df = pd.DataFrame(results)
        print("\n🏁 Model Comparison:")
        print(df.to_string(index=False, float_format='%.4f'))

        return df


def run_comprehensive_classification_evaluation(X_train: np.ndarray, y_train: np.ndarray,
                                             X_test: np.ndarray, y_test: np.ndarray,
                                             model_configs: List[Dict] = None) -> Dict:
   
    print("="*80)
    print("🔬 COMPREHENSIVE RETIS CLASSIFICATION EVALUATION")
    print("="*80)

    if model_configs is None:
        model_configs = [
            {'max_depth': 5, 'min_samples_split': 20, 'm_estimate': 2.0},
            {'max_depth': 8, 'min_samples_split': 10, 'm_estimate': 1.5},
            {'max_depth': 12, 'min_samples_split': 5, 'm_estimate': 1.0}
        ]

    # Test different configurations
    best_config = None
    best_score = -np.inf

    print("\n🔍 Testing RETIS configurations for classification...")

    for i, config in enumerate(model_configs):
        model = RETISClassifier(**config)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = accuracy_score(y_test, pred)

        print(f"   Config {i+1}: {config} → Accuracy: {score:.4f}")

        if score > best_score:
            best_score = score
            best_config = config

    print(f"\n🏆 Best configuration: {best_config} (Accuracy: {best_score:.4f})")

    # Train best model
    best_model = RETISClassifier(**best_config)
    best_model.fit(X_train, y_train)

    # Comprehensive evaluation
    evaluator = RETISClassificationEvaluator(best_model, X_train, y_train, X_test, y_test)
    eval_results = evaluator.evaluate_model(plot=True)
    cv_results = evaluator.cross_validate(cv=5)
    comparison_df = evaluator.compare_with_baselines()

    # Summary
    print(f"\n🎯 FINAL RESULTS SUMMARY:")
    print(f"   Best Configuration: {best_config}")
    print(f"   Test Accuracy: {eval_results['test'].get('accuracy', 0):.4f}")
    print(f"   Test F1 (macro): {eval_results['test'].get('f1_macro', 0):.4f}")
    if 'auc' in eval_results['test']:
        print(f"   Test AUC: {eval_results['test']['auc']:.4f}")
    print(f"   CV Score: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")

    return {
        'best_config': best_config,
        'evaluation': eval_results,
        'cv_results': cv_results,
        'comparison': comparison_df,
        'best_model': best_model
    }


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=7,
                              n_redundant=2, n_classes=3, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Run comprehensive evaluation
    results = run_comprehensive_classification_evaluation(X_train, y_train, X_test, y_test)

    print("\n✅ Comprehensive classification evaluation completed!")