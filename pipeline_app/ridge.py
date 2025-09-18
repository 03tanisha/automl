import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from .utils import prob_detec

def ridge_reg(df_path, target_col, **kwargs):
    df = pd.read_csv(df_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Problem type check
    if prob_detec(y) != "regression":
        raise ValueError("Ridge regression requires a numeric target column.")

    # Identify numeric & categorical columns
    num_feat = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feat = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    if num_feat:
        X[num_feat] = X[num_feat].astype(float)

    # Safe ColumnTransformer
    transformers = []
    if num_feat:
        transformers.append(("num", StandardScaler(), num_feat))
    if cat_feat:
        transformers.append(("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_feat))

    preprocessor = ColumnTransformer(transformers) if transformers else "passthrough"

    # Filter valid hyperparameters
    valid_params = ["alpha", "fit_intercept", "solver", "random_state"]
    ridge_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    if "alpha" in ridge_kwargs:
        ridge_kwargs["alpha"] = float(ridge_kwargs["alpha"])
    if "random_state" in ridge_kwargs:
        ridge_kwargs["random_state"] = int(ridge_kwargs["random_state"])

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", Ridge(**ridge_kwargs))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)

 

    return pipe,X_test,y_test  # or return pipe, X_test, y_test if needed
