import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Set MLflow tracking URI once at the start


# Load data
train_data = pd.read_csv(r"D:\Click_stream\train_data.csv")
test_data = pd.read_csv(r"D:\Click_stream\test_data.csv")

# Load label encoders
le1 = pickle.load(open(r"D:\Click_stream\le1_clothing_model.pkl", "rb"))
le2 = pickle.load(open(r"D:\Click_stream\le2_clothing_model.pkl", "rb"))

train_data['page2_clothing_model'] = le1.transform(train_data['page2_clothing_model'])
test_data['page2_clothing_model'] = le2.transform(test_data['page2_clothing_model'])

from sklearn.preprocessing import StandardScaler
train_features = train_data[['page1_main_category', 'page2_clothing_model', 'colour', 'order', 'price', 'location', 'model_photography']]
train_target = train_data['price_2']

test_features = test_data[['page1_main_category', 'page2_clothing_model', 'colour', 'order', 'price', 'location', 'model_photography']]
test_target = test_data['price_2']

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

model_params = {
    "Logistic_Regression": (LogisticRegression(), {
        "C": [0.01, 0.1, 1, 10, 100],  
        "solver": ["liblinear", "lbfgs"] 
    }),
    
    "Random_Forest": (RandomForestClassifier(), {
        "n_estimators": [50, 100, 200],  
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10] 
    }),
    
    "Decision_Tree": (DecisionTreeClassifier(), {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"]  
    })
}

reports = []

for name, (model, param_grid) in model_params.items():
    if param_grid: 
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid_search.fit(train_features, train_target)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        best_model = model
        best_model.fit(train_features, train_target)
        best_params = "Default Parameters"

    predictions = best_model.predict(test_features)
    accuracy = accuracy_score(test_target, predictions)
    report = classification_report(test_target, predictions)
    confusion = confusion_matrix(test_target, predictions)

    reports.append((name, best_model, best_params, accuracy, report, confusion))

for name, model, best_params, accuracy, report, confusion in reports:
    print(f"Model: {name}")
    print(f"Best Parameters: {best_params}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{confusion}\n")    


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Purchase_Classification_Models")

for name, model, best_params, accuracy, report, confusion in reports:
    with mlflow.start_run(run_name=name) as run:
        mlflow.sklearn.log_model(model, f"{name}_model")
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")
        mlflow.log_text(str(confusion), "confusion_matrix.txt")



model_name ='Random Forest Classifier'
run_id = '6e1ce31c45d24f3c893294bc2cd81e31'
model_uri = f'runs:/{run_id}/Random_Forest_model'

with mlflow.start_run(run_id=run_id):
    mlflow.register_model(model_uri= model_uri , name= model_name)


mlflow.set_tracking_uri("http://127.0.0.1:5000")
model_name = "Random Forest Classifier"
model_version = "1"
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)


import pickle

with open("random_forest_classifier_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("classification_standard_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le2, f)    