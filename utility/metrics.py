import numpy as np


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_log_error(y_true, y_pred):
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positive = np.sum(y_pred == 1)
    return true_positive / predicted_positive if predicted_positive else 0


def recall_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    actual_positive = np.sum(y_true == 1)
    return true_positive / actual_positive if actual_positive else 0


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0


def auc_score(y_true, y_pred):
    desc_score_indices = np.argsort(y_pred, kind="mergesort")[::-1]
    y_pred = y_pred[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_pred))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    if tps.size == 0 or fps[0] == 0:
        return 0.0

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    area = np.trapz(tpr, fpr)
    return area


def log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def r2_score(y_true, y_pred):
    total_variance = np.var(y_true)
    unexplained_variance = np.var(y_true - y_pred)
    return 1 - (unexplained_variance / total_variance)


def calculate_metrics(y_true, y_pred, metrics):
    try:
        results = {}
        for metric in metrics:
            if metric == "mse":
                results["mse"] = mean_squared_error(y_true, y_pred)
            elif metric == "mae":
                results["mae"] = mean_absolute_error(y_true, y_pred)
            elif metric == "rmse":
                results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
            elif metric == "msle":
                results["msle"] = mean_squared_log_error(y_true, y_pred)
            elif metric == "mape":
                results["mape"] = mean_absolute_percentage_error(y_true, y_pred)
            elif metric == "accuracy":
                results["accuracy"] = accuracy_score(y_true, y_pred)
            elif metric == "precision":
                results["precision"] = precision_score(y_true, y_pred)
            elif metric == "recall":
                results["recall"] = recall_score(y_true, y_pred)
            elif metric == "f1_score":
                results["f1_score"] = f1_score(y_true, y_pred)
            elif metric == "auc":
                results["auc"] = auc_score(y_true, y_pred)
            elif metric == "log_loss":
                results["log_loss"] = log_loss(y_true, y_pred)
            elif metric == "r2_score":
                results["r2_score"] = r2_score(y_true, y_pred)
            else:
                print(f"Unknown metric: {metric}")
        return results

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None
