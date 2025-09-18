import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from .utils import prob_detec

def dt_reg(df_path, target_col, **kwargs):
    # -------------------- LOAD DATA --------------------
    df = pd.read_csv(df_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Check if target is numeric
    if prob_detec(y) != "regression":
        raise ValueError("Decision Tree Regressor requires a numeric target column.")

    # -------------------- FEATURE TYPES --------------------
    num_feat = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feat = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    transformers = []
    if num_feat:
        transformers.append(("num", SimpleImputer(strategy="mean"), num_feat))
    if cat_feat:
        transformers.append((
            "cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_feat
        ))

    preprocessor = ColumnTransformer(transformers) if transformers else "passthrough"

    # -------------------- HYPERPARAMS --------------------
    max_depth = int(kwargs.get("max_depth", 5))
    min_samples_split = int(kwargs.get("min_samples_split", 2))
    criterion = kwargs.get("criterion", "squared_error")

    # -------------------- PIPELINE --------------------
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion
        ))
    ])

    # -------------------- TRAIN --------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)

    return pipe, X_test, y_test

# import pandas as pd
# import numpy as np
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.impute import SimpleImputer
# from .utils import prob_detec

# def dt_reg(df_path, target_col, **kwargs):
#     df = pd.read_csv(df_path)
#     X = df.drop(columns=[target_col])
#     y = df[target_col]

#     if prob_detec(y) != "regression":
#         raise ValueError("Decision Tree Regressor requires a numeric target column.")

#     num_feat = X.select_dtypes(include=[np.number]).columns.tolist()
#     cat_feat = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

#     transformers = []
#     if num_feat:
#         transformers.append(("num", SimpleImputer(strategy="mean"), num_feat))
#     if cat_feat:
#         transformers.append((
#             "cat", Pipeline([
#                 ("imputer", SimpleImputer(strategy="most_frequent")),
#                 ("encoder", OneHotEncoder(handle_unknown="ignore"))
#             ]), cat_feat
#         ))

#     preprocessor = ColumnTransformer(transformers) if transformers else "passthrough"

#     # hyperparams
#     max_depth = int(kwargs.get("max_depth", 5))
#     min_samples_split = int(kwargs.get("min_samples_split", 2))
#     criterion = kwargs.get("criterion", "squared_error")  # sklearn >=1.0

#     pipe = Pipeline([
#         ("preprocess", preprocessor),
#         ("model", DecisionTreeRegressor(
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             criterion=criterion
#         ))
#     ])

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     pipe.fit(X_train, y_train)

#     # fi
#     model = pipe.named_steps["model"]
#     preprocessor = pipe.named_steps["preprocess"]

#     # getting fi names 
#     num_names = preprocessor.transformers_[0][2]
#     cat_names = preprocessor.transformers_[1][1].get_feature_names_out(
#         preprocessor.transformers_[1][2]
#     )

#     all_features = np.concatenate([num_names, cat_names])
#     coefs = model.coef_

#     feature_importance = pd.DataFrame({
#         "feature":all_features,
#         "importance":coefs
#     })

#     return pipe

#     # return pipe, X_test, y_test
