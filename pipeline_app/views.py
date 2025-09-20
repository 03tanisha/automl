import pandas as pd
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, FileResponse
import pandas as pd
import io
import os
import json
import numpy as np
import pickle
from chardet import detect
from django.conf import settings
from .dtc import dtc_class
from .linear import linear_reg
from .ridge import ridge_reg
from .dtr import dt_reg
from .lasso import lasso_reg
from .metrics import eval_metrics
from .charts import generate_charts  
from .corr import corr_chart
from pathlib import Path


# def safe_json(data):
#     """Recursively convert numpy types to native Python for JSON serialization."""
#     if isinstance(data, dict):
#         return {k: safe_json(v) for k, v in data.items()}
#     elif isinstance(data, list):
#         return [safe_json(v) for v in data]
#     elif isinstance(data, (np.int64, np.int32)):
#         return int(data)
#     elif isinstance(data, (np.float64, np.float32)):
#         return float(data)
#     else:
#         return data


def sanitize_params(params):
    safe = {}
    for key, val in params.items():
        # convert string numbers to int/float
        if isinstance(val, str) and val.replace('.', '', 1).isdigit():
            if '.' in val:
                safe[key] = float(val)
            else:
                safe[key] = int(val)
        else:
            safe[key] = val
    return safe


MODEL_FOLDER = Path(settings.BASE_DIR) / "media" / "models"
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

@csrf_exempt
def home(request):
    message = None
    uploaded_file_name = None
    num_rows = 0
    num_cols = 0
    cat_cols = []
    num_cols_list = []
    date_cols = []
    num_cat = 0
    num_num = 0
    num_date = 0
    sum_stats = []
    col_stats = []
    chart_groups = []
    all_cols = []
    file_path = None

    # Check already trained models
    trained_models = []
    for model in ["linear", "ridge", "lasso", "dtr", "dtc", "rfr", "rfc", "xgb"]:
        model_file = os.path.join(MODEL_FOLDER, f"{model}_trained.pkl")
        if os.path.exists(model_file):
            trained_models.append(model)
    if trained_models:
        message = f"Already trained: {', '.join(trained_models)}"

    if request.method != "POST":
        chart_groups.append({
            "col": "Correlation Heatmap",
            "charts": ["<p>Upload a CSV to see correlation matrix</p>"]
        })

    if request.method == "POST":
        if "csv_file" in request.FILES:
            csv_file = request.FILES["csv_file"]

            if not csv_file:
                message = "No file uploaded"
            elif csv_file.size >= 100 * 1024 * 1024 :
                message = "File too large (Max 100 MB allowed)"
            else:
                uploaded_file_name = csv_file.name
                raw = csv_file.read()
                enc = detect(raw)["encoding"] or "utf-8"

                try:
                    decoded = raw.decode(enc)
                except UnicodeDecodeError:
                    decoded = raw.decode("ISO-8859-1")

                df = pd.read_csv(io.StringIO(decoded), on_bad_lines="skip")

                tmp_path = os.path.join(settings.BASE_DIR, "media", uploaded_file_name)
                df.to_csv(tmp_path, index=False)
                file_path = tmp_path

                num_rows, num_cols = df.shape
                cat_cols = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
                num_cols_list = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
                num_cat, num_num, num_date = len(cat_cols), len(num_cols_list), len(date_cols)
                all_cols = num_cols_list + cat_cols

                for col in df.columns:
                    col_stats.append({
                        "column": col,
                        "nulls": int(df[col].isnull().sum()),
                        "unique": int(df[col].nunique())
                    })

                for col in num_cols_list:
                    sum_stats.append({
                        "column": col,
                        "mean": round(df[col].mean(), 2),
                        "median": round(df[col].median(), 2),
                        "min": round(df[col].min(), 2),
                        "max": round(df[col].max(), 2),
                        "std": round(df[col].std(), 2),
                    })

                for col in all_cols:
                    chart_groups.append(generate_charts(df, col))

                chart_groups.append({
                    "col": "Correlation Heatmap",
                    "charts": [corr_chart(df)]
                })

                skip_cleaning = request.POST.get("skip_cleaning")
                if not skip_cleaning:
                    df.dropna(inplace=True)
                    message = "CSV Uploaded and Cleaned Successfully!"
                else:
                    message = "CSV Uploaded Successfully!"


                # return JsonResponse(safe_json({
                #     "message": message,
                #     "uploaded_file_name": uploaded_file_name,
                #     "num_rows": num_rows,
                #     "num_cols": num_cols,
                #     "cat_cols": cat_cols,
                #     "num_cols_list": num_cols_list,
                #     "date_cols": date_cols,
                #     "num_cat": num_cat,
                #     "num_num": num_num,
                #     "num_date": num_date,
                #     "col_stats": col_stats,
                #     "sum_stats": sum_stats,
                #     "file_path": file_path,
                # }))

        elif request.content_type == "application/json":
            try:
                data = json.loads(request.body)
                action = data.get("action")

                if action == "train":
                    target_col = data.get("target")
                    model_name = data.get("model")
                    file_path = data.get("file_path")
                    hyperparams = data.get("hyperparams", {})

                    hyperparams = sanitize_params(hyperparams)

                    if not target_col or not model_name or not file_path:
                        return JsonResponse({"message": "Missing data for training"}, status=400)

                    df = pd.read_csv(file_path)
                    X = df.drop(columns=[target_col])
                    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_cols = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

                    print("DEBUG: model_name received from frontend =", model_name)

                    if not numeric_cols and model_name in ["linear", "ridge", "lasso"]:
                        return JsonResponse({"message": "Error: No numeric columns available for regression."}, status=400)
                    if not numeric_cols and not categorical_cols:
                        return JsonResponse({"message": "Error: No features available to train the model."}, status=400)
                    
                                        # ✅ FIXED: hyperparam casting
                    safe_hyperparams = {}
                    for k, v in hyperparams.items():
                        key = k.replace("param_", "")
                        
                        # map frontend 'min_split' to sklearn's 'min_samples_split'
                        if key == "min_split":  
                            safe_hyperparams["min_samples_split"] = int(v)
                        elif key in ["max_depth", "n_estimators"]:
                            safe_hyperparams[key] = int(v)
                        elif isinstance(v, str):
                            try:
                                safe_hyperparams[key] = float(v)
                            except ValueError:
                                safe_hyperparams[key] = v
                        else:
                            safe_hyperparams[key] = v



                    # if model_name == "linear":
                    #     pipe = linear_reg(file_path, target_col)
                    # elif model_name == "ridge":
                    #     pipe = ridge_reg(file_path, target_col, **safe_hyperparams)
                    # elif model_name == "dtr":
                    #     pipe = dt_reg(file_path, target_col, **safe_hyperparams)
                    # elif model_name == "lasso":
                    #     pipe = lasso_reg(file_path, target_col, **safe_hyperparams)
                    # elif model_name == "dtc":
                    #     pipe = dtc_class(file_path, target_col, **safe_hyperparams)
                    # else:
                    #     return JsonResponse({"message": "Unknown model selected"}, status=400)





                    if model_name == "linear":
                        pipe, X_test_model,y_test_model = linear_reg(file_path, target_col, **safe_hyperparams)
                    elif model_name == "ridge":
                        pipe,X_test_model,y_test_model= ridge_reg(file_path, target_col, **safe_hyperparams)
                    elif model_name == "dtr":
                        pipe,X_test_model,y_test_model= dt_reg(file_path, target_col, **safe_hyperparams)
                    elif model_name == "lasso":
                        pipe,X_test_model,y_test_model= lasso_reg(file_path, target_col, **safe_hyperparams)
                    elif model_name == "dtc":
                        pipe,X_test_model,y_test_model= dtc_class(file_path, target_col, **safe_hyperparams)
                    else:
                        return JsonResponse({"message": "Unknown model selected"}, status=400)

                    model_file = os.path.join(MODEL_FOLDER, f"{model_name}_trained.pkl")
                    with open(model_file, "wb") as f:
                        pickle.dump(pipe, f)

                    request.session['last_trained_model'] = {
                        'name': model_name,
                        'file_path': file_path,
                        'target_col': target_col,
                        'type': 'regression' if model_name in ['linear','ridge','lasso','dtr','rfr'] else 'classification',
                        'model_file': model_file
                    }

                    message = f"{model_name.upper()} model trained successfully!"

                    # ---------- IMMEDIATE EVALUATION ----------
                    y_test = df[target_col]
                    X_test = df.drop(columns=[target_col])
                    model_type = 'regression' if model_name in ['linear','ridge','lasso','dtr','rfr'] else 'classification'
                    metrics = eval_metrics(pipe, X_test_model, y_test_model, model_type)

                  

                    return JsonResponse({
                        "message": message,
                        "model_file": model_file,
                        "results": metrics,
                        "model_name": model_name,
                        "type": model_type,

                    })


                # if action == "train":
                #     target_col = data.get("target")
                #     model_name = data.get("model")
                #     file_path = data.get("file_path")
                #     hyperparams = data.get("hyperparams", {})

                #     hyperparams = sanitize_params(hyperparams)

                #     if not target_col or not model_name or not file_path:
                #         return JsonResponse({"message": "Missing data for training"}, status=400)

                #     df = pd.read_csv(file_path)
                #     X = df.drop(columns=[target_col])
                #     numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                #     categorical_cols = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

                #     if not numeric_cols and model_name in ["linear", "ridge", "lasso"]:
                #         return JsonResponse({"message": "Error: No numeric columns available for regression."}, status=400)
                #     if not numeric_cols and not categorical_cols:
                #         return JsonResponse({"message": "Error: No features available to train the model."}, status=400)

                #     safe_hyperparams = {}
                #     for k, v in hyperparams.items():
                #         key = k.replace("param_", "")
                #         try:
                #             safe_hyperparams[key] = float(v)
                #         except ValueError:
                #             safe_hyperparams[key] = v

                #     if model_name == "linear":
                #         pipe, _, _ = linear_reg(file_path, target_col, **safe_hyperparams)
                #     elif model_name == "ridge":
                #         pipe, _, _ = ridge_reg(file_path, target_col, **safe_hyperparams)
                #     elif model_name == "dtr":
                #         pipe, _, _ = dt_reg(file_path, target_col, **safe_hyperparams)
                #     elif model_name == "lasso":
                #         pipe, _, _ = lasso_reg(file_path, target_col, **safe_hyperparams)
                #     elif model_name == "dtc":
                #         pipe, _, _ = dtc_class(file_path, target_col, **safe_hyperparams)
                #     else:
                #         return JsonResponse({"message": "Unknown model selected"}, status=400)

                #     model_file = os.path.join(MODEL_FOLDER, f"{model_name}_trained.pkl")
                #     with open(model_file, "wb") as f:
                #         pickle.dump(pipe, f)

                #     request.session['last_trained_model'] = {
                #         'name': model_name,
                #         'file_path': file_path,
                #         'target_col': target_col,
                #         'type': 'regression' if model_name in ['linear','ridge','lasso', 'dtr','rfr'] else 'classification',
                #         'model_file': model_file
                #     }

                #     message = f"{model_name.upper()} model trained successfully!"

                #     # ---------- IMMEDIATE EVALUATION ----------
                #                         # Evaluate immediately after training
                #     y_test = df[target_col]
                #     X_test = df.drop(columns=[target_col])
                #     model_type = 'regression' if model_name in ['linear','ridge','lasso','dtr','rfr'] else 'classification'
                #     metrics = eval_metrics(pipe, X_test, y_test, model_type)

                #     return JsonResponse({
                #         "message": message,
                #         "model_file": model_file,
                #         "results": metrics,
                #         "model_name": model_name,
                #         "type": model_type
                #     })


                elif action == "evaluate":
                    last_model = request.session.get("last_trained_model")
                    if not last_model:
                        return JsonResponse({"message": "No trained model found"}, status=400)

                    file_path = last_model["file_path"]
                    model_name = last_model["name"]
                    target_col = last_model["target_col"]
                    model_file = last_model["model_file"]
                    model_type = last_model["type"]

                    df = pd.read_csv(file_path)
                    X_test = df.drop(columns=[target_col])
                    y_test = df[target_col]

                    if not os.path.exists(model_file):
                        return JsonResponse({"message": "Trained model file missing"}, status=400)

                    with open(model_file, "rb") as f:
                        trained_model = pickle.load(f)

                    metrics = eval_metrics(trained_model, X_test, y_test, model_type)
                    return JsonResponse({"results": metrics, "model_name": model_name, "type": model_type})

                else:
                    return JsonResponse({"message": "Unknown action"}, status=400)
            except Exception as e:
                import traceback
                print("❌ Training error:", e)
                traceback.print_exc()   # full traceback in terminal
                return JsonResponse({"message": f"Error during AJAX operation: {str(e)}"}, status=500)


            # except Exception as e:
            #     return JsonResponse({"message": f"Error during AJAX operation: {str(e)}"}, status=500)

    return render(request, "app/overview.html", {
        "message": message,
        "uploaded_file_name": uploaded_file_name,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_cols_list": num_cols_list,
        "all_cols": all_cols,
        "date_cols": date_cols,
        "num_cat": num_cat,
        "num_num": num_num,
        "num_date": num_date,
        "sum_stats": sum_stats,
        "col_stats": col_stats,
        "chart_groups": chart_groups,
        "file_path": file_path,
        "columns": all_cols,
        "trained_models": trained_models
    })


@csrf_exempt
def predict(request):
    import json
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            last_model = request.session.get("last_trained_model")
            if not last_model:
                return JsonResponse({"error": "No trained model found"}, status=400)

            # Load trained model
            model_file = last_model["model_file"]
            with open(model_file, "rb") as f:
                trained_model = pickle.load(f)

            # Expecting features as a list in JSON
            features = data.get("features")
            if not features:
                return JsonResponse({"error": "No features provided"}, status=400)

            # Convert to numpy array and predict
            input_array = np.array([features])
            prediction = trained_model.predict(input_array)

            return JsonResponse({"prediction": prediction.tolist()})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "POST request required"}, status=400)
