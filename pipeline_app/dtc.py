import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from .utils import prob_detec

def dtc_class(df_path, target_col, **kwargs):
    df = pd.read_csv(df_path)

    # Define X and y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Detect problem type
    if prob_detec(y) != 'classification':
        raise ValueError("Decision Tree Classifier requires a categorical target column.")

    # Identify numeric & categorical columns
    num_feat = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feat = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Safe ColumnTransformer
    transformers = []
    if num_feat:
        transformers.append(("num", StandardScaler(), num_feat))
    if cat_feat:
        transformers.append(("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_feat))

    preprocessor = ColumnTransformer(transformers) if transformers else "passthrough"

    # Pipeline
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", DecisionTreeClassifier(**kwargs))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)

    
    return pipe,X_test,y_test # or return pipe, X_test, y_test if needed
