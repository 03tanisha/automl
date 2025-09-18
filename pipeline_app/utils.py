import pandas as pd

def prob_detec(y):
    if pd.api.types.is_numeric_dtype(y):
        return "regression"
    try:
        y_converted = pd.to_numeric(y, errors="coerce")
        if y_converted.notna().all(): 
            return "regression"
    except Exception:
        pass
    return "classification"
