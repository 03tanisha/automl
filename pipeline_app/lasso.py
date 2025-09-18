# import pandas as pd
# import numpy as np
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Lasso
# from sklearn.impute import SimpleImputer
# from .utils import prob_detec

# def lasso_reg(df_path, target_col, **kwargs):
#     df = pd.read_csv(df_path)
#     X = df.drop(columns=[target_col])
#     y = df[target_col]

#     # Problem type check
#     if prob_detec(y) != "regression":
#         raise ValueError("Lasso regression requires a numeric target column.")

#     # Identify numeric & categorical columns
#     num_feat = X.select_dtypes(include=[np.number]).columns.tolist()
#     cat_feat = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

#     transformers = []
#     if num_feat:
#         transformers.append((
#             "num", Pipeline([
#                 ("imputer", SimpleImputer(strategy="mean")),
#                 ("scaler", StandardScaler())
#             ]), num_feat
#         ))
#     if cat_feat:
#         transformers.append((
#             "cat", Pipeline([
#                 ("imputer", SimpleImputer(strategy="most_frequent")),
#                 ("encoder", OneHotEncoder(handle_unknown="ignore"))
#             ]), cat_feat
#         ))

#     preprocessor = ColumnTransformer(transformers) if transformers else "passthrough"

#     # Filter valid hyperparameters
#     valid_params = ["alpha", "fit_intercept", "max_iter", "tol", "random_state"]
#     lasso_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
#     if "alpha" in lasso_kwargs:
#         lasso_kwargs["alpha"] = float(lasso_kwargs["alpha"])
#     if "max_iter" in lasso_kwargs:
#         lasso_kwargs["max_iter"] = int(lasso_kwargs["max_iter"])
#     if "tol" in lasso_kwargs:
#         lasso_kwargs["tol"] = float(lasso_kwargs["tol"])
#     if "random_state" in lasso_kwargs:
#         lasso_kwargs["random_state"] = int(lasso_kwargs["random_state"])

#     pipe = Pipeline([
#         ("preprocess", preprocessor),
#         ("model", Lasso(**lasso_kwargs))
#     ])

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     pipe.fit(X_train, y_train)

#     # fi
#     model = pipe.named_steps["model"]
#     preprocessor = pipe.named_steps["preprocess"]

#     # getting fi names 
#     # num_names = preprocessor.transformers_[0][2]
#     # cat_names = preprocessor.transformers_[1][1].get_feature_names_out(
#     #     preprocessor.transformers_[1][2]
#     # )
#     num_names = np.array(num_feat) if num_feat else np.array([])
#     cat_names = np.array(preprocessor.transformers_[1][1].named_steps["encoder"].get_feature_names_out(cat_feat)) if cat_feat else np.array([])


#     all_features = np.concatenate([num_names, cat_names])
#     coefs = model.coef_

#     feature_importance = pd.DataFrame({
#         "feature":all_features,
#         "importance":coefs
#     })

#     return pipe

#     # return pipe, X_test, y_test
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from .utils import prob_detec

def lasso_reg(df_path, target_col, **kwargs):
    df = pd.read_csv(df_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Problem type check
    if prob_detec(y) != "regression":
        raise ValueError("Lasso regression requires a numeric target column.")

    # Identify numeric & categorical columns
    num_feat = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feat = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    transformers = []
    if num_feat:
        transformers.append((
            "num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), num_feat
        ))
    if cat_feat:
        transformers.append((
            "cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_feat
        ))

    preprocessor = ColumnTransformer(transformers) if transformers else "passthrough"

    # Filter valid hyperparameters
    valid_params = ["alpha", "fit_intercept", "max_iter", "tol", "random_state"]
    lasso_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    if "alpha" in lasso_kwargs:
        lasso_kwargs["alpha"] = float(lasso_kwargs["alpha"])
    if "max_iter" in lasso_kwargs:
        lasso_kwargs["max_iter"] = int(lasso_kwargs["max_iter"])
    if "tol" in lasso_kwargs:
        lasso_kwargs["tol"] = float(lasso_kwargs["tol"])
    if "random_state" in lasso_kwargs:
        lasso_kwargs["random_state"] = int(lasso_kwargs["random_state"])

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", Lasso(**lasso_kwargs))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)



    return pipe,X_test,y_test # or return pipe, X_test, y_test if needed
