import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    # Fallback for Python builds without reconfigure (very old Pythons)
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
import pickle
import mlflow
mlflow.search_runs()


# Set MLflow tracking URI once at the start
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Price Prediction_(Regression)")

# Load data
train_data = pd.read_csv(r"D:\Click_stream\train_data.csv")
test_data = pd.read_csv(r"D:\Click_stream\test_data.csv")

from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
train_data['page2_clothing_model'] = le1.fit_transform(train_data['page2_clothing_model'])

le2 = LabelEncoder()
test_data['page2_clothing_model'] = le2.fit_transform(test_data['page2_clothing_model'])
from sklearn.preprocessing import LabelEncoder
import pickle

label_encoder = LabelEncoder()
label_encoder.fit(train_data['page2_clothing_model'])  # whatever column you used

# Save correctly
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

# Load correctly
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

from sklearn.preprocessing import StandardScaler
train_features = train_data[['page1_main_category', 'page2_clothing_model', 'colour']]
train_target = train_data['price']

test_features = test_data[['page1_main_category', 'page2_clothing_model', 'colour']]
test_target = test_data['price']

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV


model_params = {
    "Linear Regressor": (LinearRegression(), {}),
    
    "Ridge Regressor": (Ridge(), {
        "alpha": [0.01, 0.1, 1, 10, 100]
    }),
    
    "Lasso Regressor": (Lasso(), {
        "alpha": [0.01, 0.1, 1, 10, 100]
    }),
    
    "Gradient Boosting Regressor": (GradientBoostingRegressor(), {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    }),
    
    "Random Forest Regressor": (RandomForestRegressor(), {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    })
}

reports = []

for name, (model, param_grid) in model_params.items():
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
        grid_search.fit(train_features, train_target)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        best_model = model
        best_model.fit(train_features, train_target)
        best_params = "Default Parameters"

    predictions = best_model.predict(test_features)
    mae = mean_absolute_error(test_target, predictions)
    r2 = r2_score(test_target, predictions)

    reports.append((name, best_model, best_params, mae, r2))


for name, model, best_params, mae, r2 in reports:
    print(f"Model: {name}")
    print(f"Best Parameters: {best_params}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("\n")


import mlflow
import mlflow.sklearn
import mlflow.pyfunc

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Price Prediction_(Regression)")

for name, model, best_params, mae, r2 in reports:
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        
        if name == "Linear Regressor":
            mlflow.sklearn.log_model(model, "linear_model")
        elif name == "Ridge Regressor":
            mlflow.sklearn.log_model(model, "ridge_model")
        elif name == "Lasso Regressor":
            mlflow.sklearn.log_model(model, "lasso_model")
        elif name == "Gradient Boosting Regressor":
            mlflow.sklearn.log_model(model, "gradient_boosting_model")
        elif name == "Random Forest Regressor":
            mlflow.sklearn.log_model(model, "random_forest_model")
        else:
            pass



model_name ='Gradient Boosting Regressor'
run_id = '5783d722253b4ec0afd4ec5574c3d78e'
model_uri = f'runs:/{run_id}/gradient_boosting_model'

mlflow.sklearn.load_model(f"runs:/{run_id}/gradient_boosting_model")

with mlflow.start_run(run_id=run_id):
    mlflow.register_model(model_uri= model_uri , name= model_name)        


mlflow.set_tracking_uri("http://127.0.0.1:5000")
model_name = "Gradient Boosting Regressor"
model_version = "1"
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

import pickle

with open('le1_clothing_model.pkl', 'wb') as f:
    pickle.dump(le1, f)

with open('le2_clothing_model.pkl', 'wb') as f:
    pickle.dump(le2, f)

with open('regression_standard_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le2, f)    