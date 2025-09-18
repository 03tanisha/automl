import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from .utils import prob_detec

def linear_reg(df_path, target_col, **kwargs):
    df = pd.read_csv(df_path)

    # define X and y 
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # detect problem type
    if prob_detec(y) != "regression":
        raise ValueError("Linear Regression requires a numeric target column.")

    # numeric and categorical columns
    num_feat = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feat = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # safe ColumnTransformer
    transformers = []
    if num_feat:
        transformers.append(("num", StandardScaler(), num_feat))
    if cat_feat:
        transformers.append(("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_feat))

    preprocessor = ColumnTransformer(transformers) if transformers else "passthrough"

    # pipeline
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression())
    ])

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)

    
    return pipe, X_test,y_test # or return pipe, X_test, y_test if needed
