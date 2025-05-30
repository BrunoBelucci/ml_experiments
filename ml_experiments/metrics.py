from functools import partial
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    accuracy_score,
    r2_score,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_fscore_support,
    precision_recall_curve,
)


def get_metric_fn(metric, n_classes=None):
    labels = list(range(n_classes)) if n_classes is not None else None
    # map_metric_to_func[metric] = (function, need_proba, higher_is_better)
    auc_fn = partial(roc_auc_score, multi_class="ovr", labels=labels)
    auc_fn.__name__ = "auc"
    auc_micro_fn = partial(roc_auc_score, average="micro", multi_class="ovr", labels=labels)
    auc_micro_fn.__name__ = "auc_micro"
    auc_weighted_fn = partial(roc_auc_score, average="weighted", multi_class="ovr", labels=labels)
    auc_weighted_fn.__name__ = "auc_weighted"
    log_loss_fn = partial(log_loss, labels=labels)
    log_loss_fn.__name__ = "logloss"
    balanced_accuracy_adjusted_fn = partial(balanced_accuracy_score, adjusted=True)
    balanced_accuracy_adjusted_fn.__name__ = "balanced_accuracy_adjusted"
    f1_micro_fn = partial(f1_score, average="micro", labels=labels)
    f1_micro_fn.__name__ = "f1_micro"
    f1_macro_fn = partial(f1_score, average="macro", labels=labels)
    f1_macro_fn.__name__ = "f1_macro"
    f1_weighted_fn = partial(f1_score, average="weighted", labels=labels)
    f1_weighted_fn.__name__ = "f1_weighted"
    precision_micro_fn = partial(precision_score, average="micro", labels=labels)
    precision_micro_fn.__name__ = "precision_micro"
    precision_macro_fn = partial(precision_score, average="macro", labels=labels)
    precision_macro_fn.__name__ = "precision_macro"
    precision_weighted_fn = partial(precision_score, average="weighted", labels=labels)
    precision_weighted_fn.__name__ = "precision_weighted"
    recall_micro_fn = partial(recall_score, average="micro", labels=labels)
    recall_micro_fn.__name__ = "recall_micro"
    recall_macro_fn = partial(recall_score, average="macro", labels=labels)
    recall_macro_fn.__name__ = "recall_macro"
    recall_weighted_fn = partial(recall_score, average="weighted", labels=labels)
    recall_weighted_fn.__name__ = "recall_weighted"
    average_precision_micro_fn = partial(average_precision_score, average="micro")
    average_precision_micro_fn.__name__ = "average_precision_micro"
    average_precision_macro_fn = partial(average_precision_score, average="macro")
    average_precision_macro_fn.__name__ = "average_precision_macro"
    average_precision_weighted_fn = partial(average_precision_score, average="weighted")
    average_precision_weighted_fn.__name__ = "average_precision_weighted"
    precision_recall_fscore_support_micro_fn = partial(
        precision_recall_fscore_support, average="micro", labels=labels
    )
    precision_recall_fscore_support_micro_fn.__name__ = "precision_recall_fscore_support_micro"
    precision_recall_fscore_support_macro_fn = partial(
        precision_recall_fscore_support, average="macro", labels=labels
    )
    precision_recall_fscore_support_macro_fn.__name__ = "precision_recall_fscore_support_macro"
    precision_recall_fscore_support_weighted_fn = partial(
        precision_recall_fscore_support, average="weighted", labels=labels
    )
    precision_recall_fscore_support_weighted_fn.__name__ = "precision_recall_fscore_support_weighted"
    map_metric_to_func = {
        "mse": (mean_squared_error, False, False),
        "rmse": (root_mean_squared_error, False, False),
        "r2_score": (r2_score, False, True),
        "mae": (mean_absolute_error, False, False),
        "mape": (mean_absolute_percentage_error, False, False),

        "accuracy": (accuracy_score, False, True),
        "balanced_accuracy": (balanced_accuracy_score, False, True),
        "precision_recall_curve": (precision_recall_curve, True, True),

        "auc": (auc_fn, True, True),
        "auc_micro": (auc_micro_fn, True, True),
        "auc_weighted": (auc_weighted_fn, True, True),
        "logloss": (log_loss_fn, True, False),
        "balanced_accuracy_adjusted": (balanced_accuracy_adjusted_fn, False, True),
        "f1_micro": (f1_micro_fn, False, True),
        "f1_macro": (f1_macro_fn, False, True),
        "f1_weighted": (f1_weighted_fn, False, True),
        "precision_micro": (precision_micro_fn, False, True),
        "precision_macro": (precision_macro_fn, False, True),
        "precision_weighted": (precision_weighted_fn, False, True),
        "recall_micro": (recall_micro_fn, False, True),
        "recall_macro": (recall_macro_fn, False, True),
        "recall_weighted": (recall_weighted_fn, False, True),
        "average_precision_micro": (average_precision_micro_fn, True, True),
        "average_precision_macro": (average_precision_macro_fn, True, True),
        "average_precision_weighted": (average_precision_weighted_fn, True, True),
        "precision_recall_fscore_support_micro": (precision_recall_fscore_support_micro_fn, False, True),
        "precision_recall_fscore_support_macro": (precision_recall_fscore_support_macro_fn, False, True),
        "precision_recall_fscore_support_weighted": (precision_recall_fscore_support_weighted_fn, False, True),
    }
    metric_fn, need_proba, higher_is_better = map_metric_to_func[metric]
    return metric_fn, need_proba, higher_is_better
