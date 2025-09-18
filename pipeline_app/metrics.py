import pandas as pd
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, classification_report
)

def eval_metrics(model, X_test, y_test, model_type: str):
    y_pred = model.predict(X_test)

    if model_type == "regression":
        return {
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "mse": round(mean_squared_error(y_test, y_pred), 4),
            "r2": round(r2_score(y_test, y_pred), 4),
        }
    
    elif model_type == "classification":
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        # flatten only weighted averages or per-class as needed
        return {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "report": report_dict,
            "precision": round(report_dict["weighted avg"]["precision"], 4),
            "recall": round(report_dict["weighted avg"]["recall"], 4),
            "f1": round(report_dict["weighted avg"]["f1-score"], 4),
        }


    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
