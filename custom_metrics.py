import numpy as np
from typing import Dict, Tuple, Union
from sklearn.model_selection import KFold


class CustomMetrics:

    # ==================== REGRESSION METRICS ====================

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(CustomMetrics.mse(y_true, y_pred)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        if ss_tot == 0:
            return float('nan')
        return float(1 - (ss_res / ss_tot))

    @staticmethod
    def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.median(np.abs(y_true - y_pred)))

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true != 0
        if np.sum(mask) == 0:
            return float('nan')
        return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

    @staticmethod
    def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_mean = np.mean(y_true)
        numerator = np.var(y_true - y_pred)
        denominator = np.var(y_true)
        if denominator == 0:
            return float('nan')
        return float(1 - (numerator / denominator))

    @staticmethod
    def relative_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mean_y = np.mean(np.abs(y_true))
        if mean_y == 0:
            return float('inf')
        return float(CustomMetrics.rmse(y_true, y_pred) / mean_y)

    @staticmethod
    def relative_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mean_y = np.mean(np.abs(y_true))
        if mean_y == 0:
            return float('inf')
        return float(CustomMetrics.mae(y_true, y_pred) / mean_y)

    # ==================== CLASSIFICATION METRICS ====================

    @staticmethod
    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sum(y_true == y_pred) / len(y_true))

    @staticmethod
    def precision_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
        if average == 'binary':
            # Assume positive class is 1
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true != 1))
            if tp + fp == 0:
                return 0.0
            return float(tp / (tp + fp))

        elif average == 'macro':
            classes = np.unique(y_true)
            precisions = []
            for cls in classes:
                y_true_binary = (y_true == cls).astype(int)
                y_pred_binary = (y_pred == cls).astype(int)
                tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
                fp = np.sum((y_pred_binary == 1) & (y_true_binary != 1))
                if tp + fp == 0:
                    precisions.append(0.0)
                else:
                    precisions.append(tp / (tp + fp))
            return float(np.mean(precisions))

        elif average == 'micro':
            tp = np.sum(y_pred == y_true)
            total = len(y_true)
            return float(tp / total)

        return 0.0

    @staticmethod
    def recall_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
        if average == 'binary':
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fn = np.sum((y_pred != 1) & (y_true == 1))
            if tp + fn == 0:
                return 0.0
            return float(tp / (tp + fn))

        elif average == 'macro':
            classes = np.unique(y_true)
            recalls = []
            for cls in classes:
                y_true_binary = (y_true == cls).astype(int)
                y_pred_binary = (y_pred == cls).astype(int)
                tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
                fn = np.sum((y_pred_binary != 1) & (y_true_binary == 1))
                if tp + fn == 0:
                    recalls.append(0.0)
                else:
                    recalls.append(tp / (tp + fn))
            return float(np.mean(recalls))

        elif average == 'micro':
            tp = np.sum(y_pred == y_true)
            total = len(y_true)
            return float(tp / total)

        return 0.0

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
        precision = CustomMetrics.precision_score(y_true, y_pred, average=average)
        recall = CustomMetrics.recall_score(y_true, y_pred, average=average)
        if precision + recall == 0:
            return 0.0
        return float(2 * (precision * recall) / (precision + recall))

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)

        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

        return cm

    @staticmethod
    def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        classes = np.unique(y_true)
        report = {}

        for cls in classes:
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)

            tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
            fp = np.sum((y_pred_binary == 1) & (y_true_binary != 1))
            fn = np.sum((y_pred_binary != 1) & (y_true_binary == 1))
            tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            support = np.sum(y_true_binary)

            report[str(cls)] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': support
            }

        return report

    @staticmethod
    def roc_auc_binary(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        # Sort by predicted probability
        sorted_indices = np.argsort(y_proba)[::-1]
        sorted_y_true = y_true[sorted_indices]

        # Calculate TP and FP rates
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return float('nan')

        tpr_list = []
        fpr_list = []

        tp = 0
        fp = 0

        for i in range(len(sorted_y_true)):
            if sorted_y_true[i] == 1:
                tp += 1
            else:
                fp += 1

            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        # Calculate AUC using trapezoidal rule
        fpr_list = np.array([0] + fpr_list)
        tpr_list = np.array([0] + tpr_list)

        auc = 0.0
        for i in range(len(fpr_list) - 1):
            auc += (fpr_list[i + 1] - fpr_list[i]) * (tpr_list[i] + tpr_list[i + 1]) / 2

        return float(auc)

    @staticmethod
    def roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sorted_indices = np.argsort(y_proba)[::-1]
        sorted_y_true = y_true[sorted_indices]

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        fpr_list = [0.0]
        tpr_list = [0.0]
        thresholds = [np.inf]

        tp = 0
        fp = 0

        for i, idx in enumerate(sorted_indices):
            if sorted_y_true[i] == 1:
                tp += 1
            else:
                fp += 1

            if i < len(sorted_indices) - 1:
                threshold = (y_proba[idx] + y_proba[sorted_indices[i + 1]]) / 2
            else:
                threshold = y_proba[idx] - 0.001

            tpr_list.append(tp / n_pos if n_pos > 0 else 0)
            fpr_list.append(fp / n_neg if n_neg > 0 else 0)
            thresholds.append(threshold)

        return np.array(fpr_list), np.array(tpr_list), np.array(thresholds)

    @staticmethod
    def precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sorted_indices = np.argsort(y_proba)[::-1]
        sorted_y_true = y_true[sorted_indices]

        n_pos = np.sum(y_true == 1)

        precision_list = []
        recall_list = []

        tp = 0
        fp = 0

        for i in range(len(sorted_y_true)):
            if sorted_y_true[i] == 1:
                tp += 1
            else:
                fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / n_pos if n_pos > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)

        return np.array(recall_list), np.array(precision_list)


# Helper functions for cross-validation
def custom_cross_val_score(model, X: np.ndarray, y: np.ndarray, cv: int = 5,
                          scoring: str = 'r2') -> np.ndarray:
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model_copy = type(model)(**model.get_params() if hasattr(model, 'get_params') else {})
        model_copy.fit(X_train, y_train)

        # Predict
        y_pred = model_copy.predict(X_val)

        # Score
        if scoring == 'r2':
            score = CustomMetrics.r2_score(y_val, y_pred)
        elif scoring == 'mse':
            score = -CustomMetrics.mse(y_val, y_pred)  # Negative for minimization
        elif scoring == 'accuracy':
            score = CustomMetrics.accuracy_score(y_val, y_pred)
        else:
            score = CustomMetrics.r2_score(y_val, y_pred)

        scores.append(score)

    return np.array(scores)


def train_test_split_custom(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                           random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_size))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
